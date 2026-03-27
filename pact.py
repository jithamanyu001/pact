# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import warnings
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


def create_iteration_aware_bidirectional_mask(
    input_ids,
    original_attention_mask,
    latent_token_id,
    start_latent_id,
    end_latent_id,
    n_latents_per_iteration,
    device,
    dtype=torch.float32,
):
    """
    Create attention mask with:
    - Bidirectional attention WITHIN each iteration's latent tokens
    - Causal attention ACROSS iterations (later latents can attend to earlier, not vice versa)
    - Causal attention for all non-latent tokens
    
    Args:
        input_ids: [batch, seq_len]
        original_attention_mask: [batch, seq_len]
        latent_token_id: ID of <|latent|> token
        start_latent_id: ID of <|start-latent|> token
        end_latent_id: ID of <|end-latent|> token
        n_latents_per_iteration: c_thought (number of latents generated per iteration)
        device: torch device
        dtype: torch dtype
    
    Returns:
        attention_mask: [batch, 1, seq_len, seq_len] with 0 for attend, -inf for mask
    """
    batch_size, seq_len = input_ids.shape
    
    # Start with causal mask (lower triangular allows attention to past)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    
    # Convert to attention mask format (0 = attend, -inf = mask)
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
    attention_mask = attention_mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    if original_attention_mask is not None:
        padding_bool_mask = (1.0 - original_attention_mask).bool()
        padding_bool_mask = padding_bool_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.masked_fill(padding_bool_mask, float('-inf'))
    # For each sample, enable bidirectional attention within each iteration's latents
    for batch_idx in range(batch_size):
        # Find all latent token positions
        latent_positions = (input_ids[batch_idx] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
        
        if len(latent_positions) == 0:
            continue
        
        # Group latents by iteration
        num_latents = len(latent_positions)
        num_iterations = (num_latents + n_latents_per_iteration - 1) // n_latents_per_iteration
        
        for iteration in range(num_iterations):
            start_idx = iteration * n_latents_per_iteration
            end_idx = min((iteration + 1) * n_latents_per_iteration, num_latents)
            
            # Get positions for this iteration's latents
            iter_positions = latent_positions[start_idx:end_idx]
            
            # Enable bidirectional attention within this iteration
            for i in iter_positions:
                for j in iter_positions:
                    attention_mask[batch_idx, 0, i, j] = 0.0
    
    return attention_mask


class LowRankProjectionHead(nn.Module):
    def __init__(self, hidden_size, rank, std_init=0.02):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std_init)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return self.up_proj(self.down_proj(x))


class Pact(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        n_latents=5,
        projection_rank=128,
        num_attention_heads=4,
        num_context_tokens=1,
        use_fixed_latents=False,
        max_latent_stage=1,
    ):

        super(Pact, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.n_latents = n_latents  # c_thought: latents per iteration
        self.num_context_tokens = num_context_tokens
        
        assert self.num_context_tokens >= 0, "num_context_tokens must be greater than 0"
        # Tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
            hidden_size = self.base_causallm.config.n_embd
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            hidden_size = self.base_causallm.config.hidden_size

        self.hidden_size = hidden_size
        self.use_fixed_latents = use_fixed_latents

        if self.use_fixed_latents:
            # Fixed learnable embeddings: one unique vector per latent position
            total_latent_positions = max_latent_stage * n_latents
            self.fixed_latent_embeddings = nn.Parameter(
                torch.randn(total_latent_positions, hidden_size)
            )
            nn.init.normal_(self.fixed_latent_embeddings, mean=0.0, std=0.02)
        else:
            # Original PaCT: codebook queries + cross-attention
            self.latent_queries = nn.Parameter(torch.randn(n_latents, hidden_size))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                batch_first=True
            )
            nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)

    def _get_latent_info(self, input_ids):
        """
        Get latent token information for the batch.
        
        Returns:
            latent_mask: [batch, seq_len] bool tensor, True where latent tokens are
            latent_counts: [batch] tensor with count of latents per sample
            min_latent_pos: int, minimum position of first latent across batch (or seq_len if none)
            has_latents: [batch] bool tensor, True if sample has any latents
        """
        batch_size, seq_len = input_ids.shape
        
        latent_mask = (input_ids == self.latent_token_id)  # [batch, seq_len]
        latent_counts = latent_mask.sum(dim=1)  # [batch]
        has_latents = latent_counts > 0  # [batch]
        
        # Find minimum latent position across batch
        # For samples without latents, set their "first latent pos" to seq_len
        latent_positions = latent_mask.float() * torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        latent_positions = latent_positions + (~latent_mask).float() * seq_len  # non-latents get seq_len
        first_latent_per_sample = latent_positions.min(dim=1).values  # [batch]
        
        # Min across samples that have latents
        if has_latents.any():
            min_latent_pos = first_latent_per_sample[has_latents].min().long().item()
        else:
            min_latent_pos = seq_len
        
        return latent_mask, latent_counts, min_latent_pos, has_latents

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        """
        Forward pass with iterative latent generation using codebook.
        
        For scheduled_stage S, we do S iterations:
        - Each iteration generates n_latents (c_thought) latent embeddings
        - Uses codebook queries cross-attending to last hidden state
        - Injects latents into placeholder positions
        - Runs through transformer to get next hidden state for next iteration
        """
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embedding(input_ids)
        
        # Get latent information
        latent_mask, latent_counts, min_latent_pos, has_latents = self._get_latent_info(input_ids)
        
        # If no latents in entire batch, run standard forward pass
        if not has_latents.any():
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

        # Create iteration-aware bidirectional attention mask
        custom_attention_mask = create_iteration_aware_bidirectional_mask(
            input_ids=input_ids,
            original_attention_mask=attention_mask,
            latent_token_id=self.latent_token_id,
            start_latent_id=self.start_latent_id,
            end_latent_id=self.end_latent_id,
            n_latents_per_iteration=self.n_latents,
            device=input_ids.device,
            dtype=inputs_embeds.dtype,
        )

        # Determine number of iterations needed
        max_latent_count = latent_counts.max().item()
        num_iterations = (max_latent_count + self.n_latents - 1) // self.n_latents  # ceil division
        
        # Prepare working copy of embeddings for latent injection
        working_embeds = inputs_embeds.clone()
        
        # Pass 1: Run prefix (up to min_latent_pos) to get initial KV cache and hidden states
        outputs_pre = self.base_causallm(
            inputs_embeds=inputs_embeds[:, :min_latent_pos, :],
            attention_mask=custom_attention_mask[:, :, :min_latent_pos, :min_latent_pos],
            position_ids=position_ids[:, :min_latent_pos],
            output_hidden_states=True,
            use_cache=True
        )
        
        past_key_values = outputs_pre.past_key_values
        all_logits = [outputs_pre.logits]  # Collect logits from all passes
        hidden_states = outputs_pre.hidden_states[-1]
        # Get last hidden state from prefix as initial seed for codebook
        # Shape: [batch, 1, hidden_size]
        last_hidden = hidden_states[:, -self.num_context_tokens:, :]
        residual_information = hidden_states[:,-1: , :]

        # Track current position for KV cache
        current_pos = min_latent_pos
        # Iteratively generate and process latents
        for iteration in range(num_iterations):
            # Determine which samples are active in this iteration
            # A sample is active if it has latents at positions for this iteration
            iter_start_idx = iteration * self.n_latents
            active_mask = latent_counts > iter_start_idx  # [batch]
            
            if not active_mask.any():
                break
            
            # Generate n_latents latent embeddings
            if self.use_fixed_latents:
                # Fixed learnable embeddings: look up by position index
                end_emb_idx = min(iter_start_idx + self.n_latents,
                                  self.fixed_latent_embeddings.size(0))
                gen_latents = self.fixed_latent_embeddings[iter_start_idx:end_emb_idx]
                gen_latents = gen_latents.unsqueeze(0).expand(batch_size, -1, -1)
                if gen_latents.size(1) < self.n_latents:
                    pad = torch.zeros(batch_size, self.n_latents - gen_latents.size(1),
                                      self.hidden_size, device=gen_latents.device, dtype=gen_latents.dtype)
                    gen_latents = torch.cat([gen_latents, pad], dim=1)
            else:
                # PaCT: codebook cross-attention
                # Query: codebook [n_latents, hidden] -> [batch, n_latents, hidden]
                queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
                # Key/Value: last hidden state [batch, 1, hidden]
                gen_latents, _ = self.cross_attn(queries, last_hidden, last_hidden)
                gen_latents  = gen_latents + residual_information
            # gen_latents: [batch, n_latents, hidden]

            # Inject generated latents into working_embeds at correct positions
            # Only inject where there are actual latent placeholders
            for j in range(self.n_latents):
                global_latent_idx = iter_start_idx + j
                target_pos = min_latent_pos + global_latent_idx

                if target_pos >= seq_len:
                    break

                # Check which samples have a latent at this position
                inject_mask = latent_mask[:, target_pos]  # [batch]

                if inject_mask.any():
                    # Inject for samples that have latent placeholder at this position
                    working_embeds[inject_mask, target_pos, :] = gen_latents[inject_mask, j, :]
            
            # Determine the chunk of positions to process in this iteration
            iter_end_idx = min((iteration + 1) * self.n_latents, max_latent_count)
            chunk_start = current_pos
            chunk_end = min_latent_pos + iter_end_idx
            
            if chunk_end > seq_len:
                chunk_end = seq_len
            
            if chunk_start >= chunk_end:
                break
            
            # Get the embeddings for this chunk (with injected latents)
            chunk_embeds = working_embeds[:, chunk_start:chunk_end, :]
            
            # Run this chunk through transformer with KV cache
            outputs_iter = self.base_causallm(
                inputs_embeds=chunk_embeds,
                attention_mask=custom_attention_mask[:, :, chunk_start:chunk_end, :chunk_end],
                position_ids=position_ids[:, chunk_start:chunk_end],
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            outputs_iter_hidden_states = outputs_iter.hidden_states[-1]
            hidden_states = torch.cat([hidden_states, outputs_iter_hidden_states], dim=1)
            past_key_values = outputs_iter.past_key_values
            all_logits.append(outputs_iter.logits)
            
            # Update last_hidden for next iteration
            # Use the last position's hidden state from this chunk
            last_hidden = hidden_states[:, -self.num_context_tokens:, :]
            
            current_pos = chunk_end
        
        # Final pass: Process any remaining tokens after latents (suffix)
        if current_pos < seq_len:
            suffix_embeds = working_embeds[:, current_pos:, :]
            
            outputs_suffix = self.base_causallm(
                inputs_embeds=suffix_embeds,
                attention_mask=custom_attention_mask[:, :, current_pos:, :],
                position_ids=position_ids[:, current_pos:],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            all_logits.append(outputs_suffix.logits)
        
        # Concatenate all logits
        logits = torch.cat(all_logits, dim=1)
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=working_embeds, logits=logits)

    def train(self, mode=True):
        self.base_causallm.train(mode)
        return self

    def eval(self):
        self.base_causallm.eval()
        return self

    def _generate_latents_iteratively(self, input_ids, attention_mask, inputs_embeds, position_ids):
        """
        Generate latent embeddings iteratively using codebook, vectorized across batch.
        
        Returns:
            working_embeds: embeddings with latents injected
            past_key_values: KV cache after processing all latents
            current_pos: position after all latents
            custom_attention_mask: the attention mask used
        """
        batch_size, seq_len = input_ids.shape
        
        # Get latent information
        latent_mask, latent_counts, min_latent_pos, has_latents = self._get_latent_info(input_ids)
        
        # Create iteration-aware bidirectional attention mask
        custom_attention_mask = create_iteration_aware_bidirectional_mask(
            input_ids=input_ids,
            original_attention_mask=attention_mask,
            latent_token_id=self.latent_token_id,
            start_latent_id=self.start_latent_id,
            end_latent_id=self.end_latent_id,
            n_latents_per_iteration=self.n_latents,
            device=input_ids.device,
            dtype=inputs_embeds.dtype,
        )
        
        # If no latents, just return original embeddings
        if not has_latents.any():
            return inputs_embeds, None, 0, custom_attention_mask, False
        
        # Determine number of iterations needed
        max_latent_count = latent_counts.max().item()
        num_iterations = (max_latent_count + self.n_latents - 1) // self.n_latents
        
        # Prepare working copy of embeddings
        working_embeds = inputs_embeds.clone()
        
        # Run prefix to get initial KV cache
        outputs_pre = self.base_causallm(
            inputs_embeds=inputs_embeds[:, :min_latent_pos, :],
            attention_mask=custom_attention_mask[:, :, :min_latent_pos, :min_latent_pos],
            position_ids=position_ids[:, :min_latent_pos],
            output_hidden_states=True,
            use_cache=True
        )
        
        past_key_values = outputs_pre.past_key_values
        last_hidden = outputs_pre.hidden_states[-1][:, -self.num_context_tokens:, :]
        residual_information = outputs_pre.hidden_states[-1][:,-1: , :]
        current_pos = min_latent_pos
        
        # Iteratively generate latents
        for iteration in range(num_iterations):
            iter_start_idx = iteration * self.n_latents
            active_mask = latent_counts > iter_start_idx
            
            if not active_mask.any():
                break
            
            # Generate latents
            if self.use_fixed_latents:
                end_emb_idx = min(iter_start_idx + self.n_latents,
                                  self.fixed_latent_embeddings.size(0))
                gen_latents = self.fixed_latent_embeddings[iter_start_idx:end_emb_idx]
                gen_latents = gen_latents.unsqueeze(0).expand(batch_size, -1, -1)
                if gen_latents.size(1) < self.n_latents:
                    pad = torch.zeros(batch_size, self.n_latents - gen_latents.size(1),
                                      self.hidden_size, device=gen_latents.device, dtype=gen_latents.dtype)
                    gen_latents = torch.cat([gen_latents, pad], dim=1)
            else:
                queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
                gen_latents, _ = self.cross_attn(queries, last_hidden, last_hidden)
                gen_latents  = gen_latents + residual_information
            # Inject latents
            for j in range(self.n_latents):
                global_latent_idx = iter_start_idx + j
                target_pos = min_latent_pos + global_latent_idx
                
                if target_pos >= seq_len:
                    break
                
                inject_mask = latent_mask[:, target_pos]
                if inject_mask.any():
                    working_embeds[inject_mask, target_pos, :] = gen_latents[inject_mask, j, :]
            
            # Process this chunk
            iter_end_idx = min((iteration + 1) * self.n_latents, max_latent_count)
            chunk_start = current_pos
            chunk_end = min_latent_pos + iter_end_idx
            
            if chunk_end > seq_len:
                chunk_end = seq_len
            
            if chunk_start >= chunk_end:
                break
            
            chunk_embeds = working_embeds[:, chunk_start:chunk_end, :]
            
            outputs_iter = self.base_causallm(
                inputs_embeds=chunk_embeds,
                attention_mask=custom_attention_mask[:, :, chunk_start:chunk_end, :chunk_end],
                position_ids=position_ids[:, chunk_start:chunk_end],
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            
            past_key_values = outputs_iter.past_key_values
            last_hidden = outputs_iter.hidden_states[-1][:, -self.num_context_tokens:, :]
            current_pos = chunk_end
            self.gen_forward_cnt += 1
        
        return working_embeds, past_key_values, current_pos, custom_attention_mask, True

    def generate(
        self,
        input_ids,
        attention_mask,  # Not used directly, we build custom mask
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        """
        Generate tokens with iterative latent generation, vectorized across batch.
        
        1. Process prefix and iteratively generate latents using codebook
        2. Process any suffix after latents  
        3. Autoregressively generate new tokens
        """
        self.gen_forward_cnt = 0
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        inputs_embeds = self.embedding(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Generate latents iteratively (vectorized)
        working_embeds, past_key_values, current_pos, custom_attention_mask, had_latents = \
            self._generate_latents_iteratively(input_ids, attention_mask, inputs_embeds, position_ids)
        
        # Process any remaining suffix after latents
        if had_latents and current_pos < seq_len:
            suffix_embeds = working_embeds[:, current_pos:, :]
            
            outputs_suffix = self.base_causallm(
                inputs_embeds=suffix_embeds,
                attention_mask=custom_attention_mask[:, :, current_pos:, :],
                position_ids=position_ids[:, current_pos:],
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            past_key_values = outputs_suffix.past_key_values
            logits = outputs_suffix.logits
            self.gen_forward_cnt += 1
        elif had_latents:
            # All tokens were latents, need to get logits
            # Run a dummy forward to get the final logits (already have KV cache)
            # Actually the last iteration already gave us logits
            # We need to do one more forward with the last position
            last_embed = working_embeds[:, -1:, :]
            outputs_last = self.base_causallm(
                inputs_embeds=last_embed,
                position_ids=position_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs_last.past_key_values
            logits = outputs_last.logits
            self.gen_forward_cnt += 1
        else:
            # No latents, just run the full sequence
            outputs = self.base_causallm(
                inputs_embeds=working_embeds,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            self.gen_forward_cnt += 1
        
        # Get first new token (vectorized across batch)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)  # [batch]
        
        # Initialize generated tokens list
        generated_tokens = [input_ids]
        generated_tokens.append(next_tokens.unsqueeze(1))
        
        # Track which sequences are finished
        finished = (next_tokens == self.eos_token_id)
        
        # Current position for position_ids
        cur_pos = seq_len
        
        # Autoregressive generation
        for step in range(max_new_tokens - 1):
            if finished.all():
                break
            
            # Get embeddings for new tokens
            new_token_embeds = self.embedding(next_tokens).unsqueeze(1)  # [batch, 1, hidden]
            new_position_ids = torch.full((batch_size, 1), cur_pos, device=device, dtype=torch.long)
            
            # Forward pass with KV cache
            outputs = self.base_causallm(
                inputs_embeds=new_token_embeds,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            self.gen_forward_cnt += 1
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            
            # Get next tokens
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)  # [batch]
            
            # For finished sequences, replace with pad token (or keep eos)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, self.eos_token_id), next_tokens)
            
            # Update finished mask
            finished = finished | (next_tokens == self.eos_token_id)
            
            # Append to generated tokens
            generated_tokens.append(next_tokens.unsqueeze(1))
            cur_pos += 1
        
        # Handle synced_gpus for FSDP
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                dummy_embeds = self.embedding(next_tokens).unsqueeze(1)
                dummy_pos = torch.full((batch_size, 1), cur_pos, device=device, dtype=torch.long)
                _ = self.base_causallm(
                    inputs_embeds=dummy_embeds,
                    position_ids=dummy_pos,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                cur_pos += 1
        
        # Concatenate all tokens
        all_tokens = torch.cat(generated_tokens, dim=1)
        
        if output_embedding:
            return all_tokens, working_embeds
        else:
            return all_tokens
if __name__ == "__main__":

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="cpu")

    input_ids = torch.tensor([[0, 901, 900, 900, 902, 1, 1, 1, 1, 1],[0, 901, 900, 900, 902, 1, 1, 1, 1, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    labels = input_ids.clone()

    # Test standard PaCT
    pact = Pact(
        base_causallm=model,
        latent_token_id=900,
        start_latent_id=901,
        end_latent_id=902,
        eos_token_id=1,
        n_latents=5,
        num_attention_heads=4,
    )
    outputs = pact(input_ids, attention_mask, labels, position_ids)
    print(f"PaCT loss: {outputs.loss.item()}")

    # Test fixed latent embeddings
    pact_fixed = Pact(
        base_causallm=model,
        latent_token_id=900,
        start_latent_id=901,
        end_latent_id=902,
        eos_token_id=1,
        n_latents=5,
        num_attention_heads=4,
        use_fixed_latents=True,
        max_latent_stage=2,
    )
    outputs_fixed = pact_fixed(input_ids, attention_mask, labels, position_ids)
    print(f"Fixed latents loss: {outputs_fixed.loss.item()}")
    print("Both modes work correctly!")
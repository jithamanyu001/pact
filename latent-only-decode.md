# Latent-Only Decoding

Tests whether PaCT's latent states encode enough reasoning information to decode the answer **without access to the original question tokens**.

---

## What it does

During normal PaCT inference the answer tokens attend to both the question and the latent states. Latent-only decoding removes the question from the decoding context:

1. **Latents are generated normally** — they see the full question, so they have the chance to compress relevant information.
2. **KV cache is discarded** — the cache built during latent generation contains question token representations.
3. **Latent region is re-run from scratch** — only the `<start-latent>`, latent tokens, and `<end-latent>` are passed through the transformer with fresh position IDs starting from 0. This builds a new KV cache with zero question information.
4. **Autoregressive decoding proceeds** using only this latent-only KV cache.

If the model can still produce correct answers, it proves the latents have genuinely compressed the reasoning — they are not inert placeholders.

```
Normal PaCT:     [question] [<start>  latents  <end>] → answer attends to everything
Latent-only:     [question] [<start>  latents  <end>] → discard question KV cache
                            [<start>  latents  <end>] → answer attends only to this
```

---

## Where the logic lives

**`pact.py`** — `generate()` method:

```python
if self.latent_only_decode and had_latents:
    # Slice the latent region: <start-latent> + latents + <end-latent>
    start_latent_pos = max(min_latent_pos - 1, 0)
    latent_region = working_embeds[:, start_latent_pos:, :]

    # Fresh position IDs (0, 1, 2, ...) — no reference to question positions
    latent_position_ids = torch.arange(region_len, device=device)

    # Rebuild attention mask for just the latent region
    latent_attn_mask = create_iteration_aware_bidirectional_mask(...)

    # Run from scratch — builds a new KV cache without question tokens
    outputs_latent_only = self.base_causallm(
        inputs_embeds=latent_region,
        attention_mask=latent_attn_mask,
        position_ids=latent_position_ids,
        use_cache=True,
    )
    past_key_values = outputs_latent_only.past_key_values
```

**`run.py`** — eval loop saves per-example outputs (question, ground truth, prediction, correctness, raw generation) to JSON for inspection.

---

## How to run

1. Set `load_model_path` in the config to your trained PaCT checkpoint:

```bash
# Edit args/gsm_latent_only_decode.yaml
load_model_path: path/to/your/pact/checkpoint
```

2. Create the output directory:

```bash
mkdir -p eval_outputs
```

3. Run evaluation:

```bash
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py args/gsm_latent_only_decode.yaml
```

4. Inspect outputs:

```bash
# Each GPU rank saves its own shard
cat eval_outputs/gsm_latent_only_decode_rank0.json | python -m json.tool | head -50
```

Output JSON per example:
```json
{
  "idx": 0,
  "question": "Janet has 3 apples...",
  "gt_answer": "5",
  "predicted_answer": "5",
  "correct": true,
  "full_output": "The answer is 5",
  "raw_output": "<start-latent>...<end-latent> The answer is 5"
}
```

---

## Config

`args/gsm_latent_only_decode.yaml` — key settings:

| Setting | Value | Purpose |
|---|---|---|
| `latent_only_decode` | `True` | Enables the latent-only decode path |
| `save_eval_outputs` | `True` | Saves per-example predictions to JSON |
| `eval_output_path` | `eval_outputs/gsm_latent_only_decode.json` | Output file (rank suffix added automatically) |
| `only_eval` | `True` | Skip training, eval only |

---

## Expected results for rebuttal

| Condition | Expected Accuracy |
|---|---|
| Normal PaCT (question + latents) | ~34.4% |
| Latent-only decode (latents only) | ~20-30% |
| Fixed latent placeholder | ~22% |

If latent-only decoding outperforms the fixed-latent baseline, it demonstrates that PaCT's latents carry reasoning content beyond what a placeholder could provide. Even partial accuracy recovery is strong evidence — the latents encode enough of the question and reasoning to generate answers without ever seeing the original prompt during decoding.

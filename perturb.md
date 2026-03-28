# Perturbation Analysis

Tests whether PaCT's latent embeddings are causally active (not inert placeholders).
All perturbations are **inference-only** — `forward()` and training are untouched.

---

## Modes

| Mode | What happens to `gen_latents` |
|------|-------------------------------|
| `none` | No change. Standard PaCT. |
| `noise` | Add `N(0, sigma^2 * I)`. Sweep sigma = 0.01 → 1.0 to see monotonic accuracy drop. |
| `zero` | Replace with zeros. Tests if latent content is necessary at all. |
| `random` | Replace with random vector scaled to same norm. Tests if the specific direction matters. |
| `swap` | Each example gets the previous example's latents. Tests if latents are input-specific. |

Expected result: accuracy should degrade as perturbation increases. If it does, latents are causally active (unlike Coconut per Zhang et al.).

---

## How swap works

Full-latent swap: B generates all its latents normally, then all positions are replaced with A's latents at the end.

`_swap_buffer` stores the list of per-stage latents from the previous example.
`_swap_current_latents` accumulates the current example's real latents during generation.

**Example** (3 stages, eval runs sequentially one example at a time):

```
Question A:
  Loop: generate A's latents normally across all 3 stages
        save [A_stage0, A_stage1, A_stage2]
  _swap_buffer = None → no swap, A uses its own latents
  After: _swap_buffer = [A_stage0, A_stage1, A_stage2]

Question B:
  Loop: generate B's latents normally across all 3 stages
        save [B_stage0, B_stage1, B_stage2]
  After loop: _swap_buffer exists →
    1. Overwrite all latent positions in working_embeds with A's latents
    2. Re-run transformer over prefix+latents to rebuild KV cache
  After: _swap_buffer = [B_stage0, B_stage1, B_stage2]

Question C:
  Generates normally, then gets B's full latents swapped in.
```

B's saved latents are "clean" — generated conditioned on B's own previous stages, not contaminated by A's latents. Works because eval uses `batch_size=1` and the same `Pact` object persists across calls.

---

## Where perturbation is applied

`_generate_latents_iteratively()` in `pact.py` — the path used by `generate()` (eval only).

```python
# Inside the loop (per iteration):
if use_swap:
    self._swap_current_latents.append(gen_latents.clone())  # save, don't modify
else:
    gen_latents = self._perturb_latents(gen_latents)         # noise/zero/random

# After the loop (swap only):
if use_swap and self._swap_buffer is not None:
    # Overwrite ALL latent positions with previous example's latents
    for iteration, swapped_latents in enumerate(self._swap_buffer):
        working_embeds[..., latent_positions, :] = swapped_latents[...]
    # Re-run transformer to rebuild KV cache
    outputs_rerun = self.base_causallm(working_embeds[:, :current_pos, :], ...)
    past_key_values = outputs_rerun.past_key_values
```

---

## Config files

All in `args/`. Set `load_model_path` to your trained PaCT checkpoint before running.

| File | Mode | Sigma |
|------|------|-------|
| `gsm_perturb_zero.yaml` | zero | - |
| `gsm_perturb_noise_0.01.yaml` | noise | 0.01 |
| `gsm_perturb_noise_0.05.yaml` | noise | 0.05 |
| `gsm_perturb_noise_0.1.yaml` | noise | 0.1 |
| `gsm_perturb_noise_0.5.yaml` | noise | 0.5 |
| `gsm_perturb_noise_1.0.yaml` | noise | 1.0 |
| `gsm_perturb_random.yaml` | random | - |
| `gsm_perturb_swap.yaml` | swap | - |

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_perturb_zero.yaml
```

---

## Expected result table for rebuttal

| Perturbation | Expected Accuracy |
|---|---|
| None (PaCT baseline) | ~34.4% |
| Noise σ=0.01 | ~34% |
| Noise σ=0.1 | ~28-32% |
| Noise σ=1.0 | ~16-20% |
| Zero | ~16-22% |
| Random | ~16-22% |
| Swap | ~18-24% |

A monotonic drop with noise + near-baseline accuracy under zero/random/swap directly refutes the "causally inert" claim from Zhang et al.

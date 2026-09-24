# paper-to-torch

Practice turning the equations in ML papers into correct, vectorized PyTorch, and understanding what they do.

Each notebook takes real equations from a real paper and walks through the same loop:

1. **Read the equation.** Decode the notation, write down every symbol's shape, and find the free and summed indices.
2. **Implement it.** Write the literal version first, then the vectorized one.
3. **Test it.** Compare against a reference, check the paper's own claims, and try extreme inputs.
4. **Understand it.** Run an experiment that shows *why* the equation has the form it does.

## Setup

```bash
git clone https://github.com/sidkiblawi/paper-to-torch.git
cd paper-to-torch
uv sync
uv run jupyter lab
```

Work in `exercises/`. Every `raise NotImplementedError` is yours to fill in, and the cell after it tests your code with ✅/❌ output. `solutions/` has the same notebooks, completed, plus answers to the reflection questions. Try an exercise properly before you open the solution.

## Curriculum

| # | Notebook | Paper(s) | Main skill |
|---|---|---|---|
| 00 | [Reading equations](exercises/00_reading_equations.ipynb) | — | The recipe: notation → shapes → indices → einsum / broadcasting |
| 01 | [Softmax & cross-entropy](exercises/01_softmax_crossentropy.ipynb) | Szegedy et al. 2016 (label smoothing) | Numerical stability; log-space; deriving a gradient |
| 02 | [Normalization](exercises/02_normalization.ipynb) | BatchNorm, LayerNorm, RMSNorm | *Which axis* does $\sum_i$ run over? |
| 03 | [Attention](exercises/03_attention.ipynb) | Vaswani et al. 2017 | Hidden batch/head dims, masks, split/merge heads, testing footnote 4 |
| 04 | [Positional encodings](exercises/04_positional_encodings.ipynb) | Vaswani 2017; RoFormer (RoPE) | Index arithmetic; matrix form → efficient form; layout conventions |
| 05 | [Activations & FFN](exercises/05_activations_ffn.ipynb) | GELU, Swish, GLU variants | Special functions; row-vector papers; parameter-count arguments |
| 06 | [KL, VAE, CLIP](exercises/06_divergences_vae_clip.ipynb) | Kingma & Welling 2013; CLIP; InfoNCE | Expectations: closed form vs. Monte Carlo; reading pseudocode |
| 07 | [Optimizers](exercises/07_optimizers.ipynb) | Momentum, Adam, AdamW | Algorithm boxes → stateful code; "equivalent" parameterizations |
| 08 | [LoRA](exercises/08_lora.ipynb) | Hu et al. 2021 | Equation + prose → `nn.Module`; frozen params; merging |
| 09 | [DDPM](exercises/09_ddpm.ipynb) | Ho et al. 2020 | 1- vs 0-indexing; per-example broadcasting; oracle tests; a full training run |
| 10 | [DPO](exercises/10_dpo.ipynb) | Rafailov et al. 2023 | Sequence log-probs (shift + mask); checking a paper's theorem empirically |
| 11 | [MoE capstone](exercises/11_moe_capstone.ipynb) | Shazeer 2017; Switch Transformer | Dense reference → sparse implementation; partly non-differentiable losses |
| 12 | [Inside `torch.nn`](exercises/12_inside_nn_module.ipynb) | Kaiming init; BatchNorm; Dropout | Rebuild `nn.Linear`, `BatchNorm1d`, `Dropout`, `Embedding` exactly; parameters vs. buffers; hooks |
| 13 | [Inside the built-ins](exercises/13_inside_functional_autograd_optim.ipynb) | STE / VQ-VAE; SGDR | Full `F.cross_entropy` API; `autograd.Function`; `torch.optim.Optimizer` internals and schedulers |
| 99 | [Template: your own paper](exercises/99_template_your_own_paper.ipynb) | any | A reusable worksheet for practicing on new papers |

Notebooks 00 and 01 are prerequisites for the rest. After those, 02–08 can be done in any order. Notebooks 09–11 are longer and assume the earlier ones. Notebooks 12–13 open up the PyTorch built-ins the other notebooks use as reference answers. You can do them any time after 01 (and 13 after 07). Notebook 12 is good preparation for 08.

## The recipe (short version)

| Step | Ask yourself |
|---|---|
| **Symbols** | Which symbols are inputs, which are learned, and which are hyperparameters? Where does the paper define each one? |
| **Shapes** | What shape is each symbol? Which batch, sequence, or head dimension is the paper leaving out? |
| **Indices** | Which indices appear on the left-hand side (those form the output shape)? Which appear only on the right (those are reductions)? |
| **Conventions** | Does the paper write $Wx$ or $xW$? Is $t$ 0- or 1-indexed? In a mask, does `True` mean keep or block? |
| **Hazards** | Is there an `exp`, `log`, division, or subtraction of near-equal numbers that could overflow, underflow, or cancel? |
| **Literal → vectorized** | Did you write loops first, then vectorize, and test the two against each other? |
| **Test the claims** | Can you check an invariance, a closed form, a behavior at init, or an equivalence the paper states? |

## Useful habits

- Put a `# (B, T, D)` shape comment on every line.
- **Never divide probabilities.** Subtract log-probs instead.
- When a paper provides pseudocode (CLIP, DDPM, Adam), it's often clearer than the equations. Implement from it and use the equations to check your work.
- When you're unsure about a closed-form derivation, estimate the same quantity by Monte Carlo and compare.

## Repo layout

```
exercises/     ← work here
solutions/     ← completed notebooks + answers
p2t/           ← check(), check_grad(), check_shape(), seed() used by every notebook
_build/        ← source for the notebooks (one file generates both versions)
```

To regenerate the notebooks after editing `_build/src/*.txt`, and to confirm every solution runs:

```bash
uv run python _build/build.py --execute
```

Note that the build overwrites `exercises/`. Do your own work in a copy, or commit it first.

# autoresearch-mlx

Autonomous LLM pretraining research on Apple Silicon, using [MLX](https://github.com/ml-explore/mlx).

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for Apple Silicon (M1/M2/M3/M4) with unified memory.

## What is this?

This project lets an AI agent autonomously run pretraining experiments on a small GPT model, trying to minimize validation bits-per-byte (val_bpb) within a fixed 5-minute training budget. The agent modifies `train.py`, commits, runs the experiment, evaluates the result, and either keeps or discards each change — looping indefinitely until stopped.

The entire pipeline runs locally on Apple Silicon hardware using Apple's [MLX framework](https://github.com/ml-explore/mlx), with no GPU server required.

## How it works

```
┌─────────────────────────────────────────────────┐
│                 EXPERIMENT LOOP                  │
│                                                  │
│  1. Agent proposes a change to train.py          │
│  2. Git commit the change                        │
│  3. Run training (5-minute wall clock budget)    │
│  4. Evaluate val_bpb on held-out data            │
│  5. If improved → keep. If worse → git reset.    │
│  6. Log result to results.tsv                    │
│  7. Repeat forever.                              │
└─────────────────────────────────────────────────┘
```

The agent can modify anything in `train.py` — model architecture, hyperparameters, optimizer settings, the training loop itself — as long as the code runs and finishes within the time budget.

## Architecture

The default model is a small GPT (decoder-only transformer) with:

| Parameter | Default |
|---|---|
| Layers | 8 |
| Attention heads | 6 |
| KV heads | 6 (full MHA, set lower for GQA) |
| Embedding dim | 384 |
| Context length | 2048 |
| Vocab size | 8192 (BPE) |
| Parameters | ~10M |

Key architecture choices:
- **RMSNorm** with `mx.fast.rms_norm` (optimised Metal kernel)
- **Rotary Position Embeddings (RoPE)** for position encoding
- **SwiGLU MLP** (SiLU-gated linear unit)
- **Grouped Query Attention** support via configurable `n_kv_head`
- **Scaled dot-product attention** via `mx.fast.scaled_dot_product_attention` (Metal kernel)

## Dataset

Training data comes from [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) on Hugging Face — a shuffled mix of web text stored as Parquet shards. A custom BPE tokenizer (vocab size 8192) is trained on the data using [rustbpe](https://github.com/karpathy/rustbpe).

The dataloader uses **BOS-aligned best-fit packing** to maximise token utilisation per sequence.

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) — tested on M4 Mac Mini with 16GB unified memory
- **Python**: 3.10+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

### Dependencies

| Package | Purpose |
|---|---|
| `mlx >= 0.22.0` | Apple's ML framework for Apple Silicon |
| `numpy >= 2.0.0` | Array operations |
| `pyarrow >= 17.0.0` | Reading Parquet data shards |
| `requests >= 2.32.0` | Downloading data from Hugging Face |
| `tiktoken >= 0.8.0` | BPE tokenizer encoding/decoding |
| `rustbpe >= 0.1.0` | Fast BPE tokenizer training |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ElixirLabsUK/autoresearch-mlx.git
cd autoresearch-mlx
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Prepare data and tokenizer

```bash
uv run prepare.py
```

This downloads training shards from Hugging Face and trains a BPE tokenizer. By default it downloads 10 shards (~500MB). You can adjust:

```bash
# Download only 4 shards (smaller, for quick testing)
uv run prepare.py --num-shards 4

# Download all 6542 shards (full dataset)
uv run prepare.py --num-shards -1
```

Data and tokenizer are cached in `~/.cache/autoresearch/`.

### 4. Run a training experiment

```bash
uv run train.py
```

Training runs for exactly 5 minutes (wall clock, excluding warmup compilation steps), then evaluates on held-out data and prints a summary:

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    325.9
peak_memory_mb:   4500.2
total_tokens_M:   12.3
num_steps:        150
num_params_M:     10.5
depth:            8
```

## Project structure

```
autoresearch-mlx/
├── prepare.py     # Data download, tokenizer training, dataloader, evaluation (read-only)
├── train.py       # Model architecture + training loop (the file you modify)
├── program.md     # Agent instructions for the experiment loop
├── pyproject.toml # Project metadata and dependencies
├── results.tsv    # Experiment results log (not committed)
└── run.log        # Latest training run output (not committed)
```

### File roles

- **`prepare.py`** — Fixed infrastructure. Downloads data shards, trains the BPE tokenizer, provides the dataloader and evaluation function (`evaluate_bpb`). **Do not modify.**
- **`train.py`** — The experiment file. Contains the GPT model definition, hyperparameters, optimizer setup, and training loop. **This is the only file the agent (or you) should edit.**
- **`program.md`** — Instructions for the AI agent on how to run the experiment loop.
- **`results.tsv`** — Tab-separated log of all experiments. Columns: `commit`, `val_bpb`, `memory_gb`, `status`, `description`.

## Running experiments

### Manual experimentation

Edit `train.py` to change the model or training setup, then run:

```bash
uv run train.py > run.log 2>&1
grep "^val_bpb:" run.log
```

### Autonomous agent experimentation

The project is designed to be driven by an AI coding agent (e.g. Claude Code). Point the agent at `program.md` for full instructions. The agent will:

1. Create an experiment branch (`autoresearch/<tag>`)
2. Run the baseline
3. Loop: modify `train.py` → commit → train → evaluate → keep/discard
4. Log all results to `results.tsv`

### Results tracking

Each experiment is logged to `results.tsv`:

```
commit    val_bpb     memory_gb   status    description
a1b2c3d   1.234567    4.4         keep      baseline
b2c3d4e   1.220100    4.5         keep      increase LR to 6e-4
c3d4e5f   1.250000    4.4         discard   switch to GeLU
d4e5f6g   0.000000    0.0         crash     double model width (OOM)
```

## Hyperparameters

All tunable hyperparameters are defined at the top of `train.py`:

```python
# Model architecture
DEPTH = 8               # number of transformer layers
N_HEAD = 6              # attention heads
N_KV_HEAD = 6           # key/value heads (set < N_HEAD for GQA)
N_EMBD = 384            # embedding dimension

# Optimization
BATCH_SIZE = 4           # batch size (small for 16GB unified memory)
LEARNING_RATE = 3e-4     # peak learning rate
WEIGHT_DECAY = 0.1       # AdamW weight decay
WARMUP_RATIO = 0.05      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5     # fraction for LR cooldown
FINAL_LR_FRAC = 0.1      # final LR as fraction of peak
GRAD_ACCUM_STEPS = 8     # gradient accumulation steps
```

### Learning rate schedule

The LR follows a warmup-hold-cooldown schedule based on wall clock progress:

1. **Warmup** (0% → 5%): Linear ramp from 0 to peak LR
2. **Hold** (5% → 50%): Constant at peak LR
3. **Cooldown** (50% → 100%): Linear decay to `FINAL_LR_FRAC × peak LR`

## Apple Silicon performance notes

- MLX uses **lazy evaluation** — tensors are not computed until `mx.eval()` is called
- **Unified memory** means no CPU↔GPU transfer overhead, but you share memory with the OS. Keep peak usage under ~10GB on 16GB machines.
- `mx.fast.rms_norm` and `mx.fast.scaled_dot_product_attention` are **optimised Metal kernels** — always prefer these over manual implementations
- Gradient checkpointing is not available in MLX — manage memory via model size and batch size
- Expect **10–50x slower** than H100. The 5-minute budget still applies; you get fewer steps but relative comparisons between experiments remain valid.
- Smaller models with more training steps often outperform larger models with fewer steps on limited hardware.

## Evaluation metric

**Bits per byte (BPB)** — a vocabulary-size-independent metric computed on held-out validation data. Lower is better. This is the single number that determines whether an experiment is a success.

BPB is calculated by:
1. Computing cross-entropy loss per token on validation data
2. Weighting by the UTF-8 byte length of each token
3. Converting from nats to bits (dividing by ln(2))

This makes results comparable across different tokenizer vocab sizes.

## Credits

- Original project: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy
- Dataset: [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- ML framework: [MLX](https://github.com/ml-explore/mlx) by Apple
- Tokenizer training: [rustbpe](https://github.com/karpathy/rustbpe) by Andrej Karpathy

## License

MIT

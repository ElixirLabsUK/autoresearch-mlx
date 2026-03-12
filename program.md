# autoresearch-mlx

Autonomous LLM pretraining research on Apple Silicon, using MLX.

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for Apple Silicon (M1/M2/M3/M4).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Platform

- **Hardware**: Apple Silicon (M1/M2/M3/M4) with unified memory
- **Framework**: MLX (Apple's ML framework)
- **Memory**: Shared CPU/GPU unified memory — no VRAM distinction. On 16GB machines, keep peak usage under ~10GB to leave room for the OS.
- **Performance**: Expect ~10-50x slower than H100. The 5-minute budget still applies — you'll get fewer steps but the comparison within experiments is still valid.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Architecture, optimizer, hyperparameters, training loop, batch size, model size — all fair game.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness (`evaluate_bpb` in `prepare.py`).

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, everything is fair game. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but keep peak usage reasonable for the machine (under ~10GB on 16GB machines).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep.

**The first run**: Always establish the baseline first by running train.py as is.

## Output format

The script prints a summary like:

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

Extract the key metric: `grep "^val_bpb:" run.log`

## Logging results

Log each experiment to `results.tsv` (tab-separated).

Header and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_memory_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	4.4	keep	baseline
b2c3d4e	1.220100	4.5	keep	increase LR to 6e-4
c3d4e5f	1.250000	4.4	discard	switch to GeLU
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Tune `train.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_memory_mb:" run.log`
6. If grep is empty, run crashed. `tail -n 50 run.log` for the trace and attempt a fix.
7. Record results in the tsv (do NOT commit results.tsv)
8. If val_bpb improved (lower), keep the commit and advance
9. If val_bpb is equal or worse, git reset back

**Timeout**: Each experiment should take ~5 minutes + eval overhead. Kill and discard if it exceeds 10 minutes.

**Crashes**: Fix typos/imports and re-run. If the idea is fundamentally broken, log as crash and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining approaches, try radical changes, re-read the code. The loop runs until manually stopped.

## Apple Silicon tips for the agent

- MLX uses lazy evaluation — call `mx.eval()` to force computation
- Unified memory means no CPU↔GPU transfer cost
- SwiGLU MLP is typically better than ReLU² on MLX
- Smaller models with more steps often beat larger models with fewer steps on limited hardware
- `mx.fast.rms_norm` and `mx.fast.scaled_dot_product_attention` are optimized Metal kernels — prefer them
- Gradient checkpointing isn't available in MLX — manage memory via model size and batch size

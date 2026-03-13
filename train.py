"""
Autoresearch pretraining script for Apple Silicon. Single-file, MLX.
Usage: uv run train.py
"""

import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model (with Value Embeddings, squared ReLU, logit capping)
# ---------------------------------------------------------------------------

def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 384


class RotaryEmbedding:
    def __init__(self, head_dim, max_seq_len=8192, base=10000.0):
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        t = mx.arange(max_seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        self.cos = mx.cos(freqs)
        self.sin = mx.sin(freqs)
        mx.eval(self.cos, self.sin)

    def __call__(self, x, offset=0):
        T = x.shape[1]
        cos = self.cos[offset:offset + T][None, :, None, :]
        sin = self.sin[offset:offset + T][None, :, None, :]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return mx.concatenate([y1, y2], axis=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Value embedding gate (32 channels from input → n_kv_head gates)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

        self.scale = self.head_dim ** -0.5

    def __call__(self, x, rotary, ve=None):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value embeddings: gated additive
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Apply rotary embeddings then QK norm (post-RoPE norm, proven better)
        q = rotary(q)
        k = rotary(k)
        q = norm(q)
        k = norm(k)

        # Transpose for attention: (B, N, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    """Squared ReLU MLP — simpler and fewer params than SwiGLU."""
    def __init__(self, config):
        super().__init__()
        hidden = 3 * config.n_embd  # 3x (not 4x) since no gate projection
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def __call__(self, x, rotary, ve=None):
        x = x + self.attn(norm(x), rotary, ve)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.n_embd // config.n_head, config.sequence_len * 2)

        # Per-layer value embeddings
        kv_dim = config.n_kv_head * (config.n_embd // config.n_head)
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
        }

        # Residual scaling (learnable per-layer)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((config.n_layer,), 0.1, dtype=mx.float32)

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        # Embeddings
        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for block in self.blocks:
            # Attention: uniform init, zero output projection
            block.attn.c_q.weight = mx.random.uniform(-scale, scale, block.attn.c_q.weight.shape).astype(mx.bfloat16)
            block.attn.c_k.weight = mx.random.uniform(-scale, scale, block.attn.c_k.weight.shape).astype(mx.bfloat16)
            block.attn.c_v.weight = mx.random.uniform(-scale, scale, block.attn.c_v.weight.shape).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)
            # VE gate: zero init (starts with no VE contribution, learns to use it)
            block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(mx.bfloat16)
            # MLP: uniform init, zero output projection
            block.mlp.c_fc.weight = mx.random.uniform(-scale, scale, block.mlp.c_fc.weight.shape).astype(mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight).astype(mx.bfloat16)

        # Value embeddings: uniform init
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-scale, scale, ve.weight.shape).astype(mx.bfloat16)

        # Residual lambdas
        self.resid_lambdas = mx.ones((self.config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1, dtype=mx.float32)

    def __call__(self, idx, targets=None):
        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx)
            x = block(x, self.rotary, ve)
        x = norm(x)
        logits = self.lm_head(x)
        # Logit capping — prevents explosion
        logits = 15.0 * mx.tanh(logits / 15.0)
        return logits

    def num_params(self):
        from mlx.utils import tree_flatten
        leaves = tree_flatten(self.parameters())
        return sum(p.size for _, p in leaves)

# ---------------------------------------------------------------------------
# Per-param-group AdamW (ported from solo run)
# ---------------------------------------------------------------------------

class AdamW:
    def __init__(self, model, matrix_lr, embedding_lr, unembedding_lr, scalar_lr,
                 weight_decay, adam_betas):
        self.param_config = {}
        self.adam_state = {}

        dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5

        for path, param in tree_flatten(model.parameters()):
            if "blocks" in path and param.ndim == 2:
                cfg = {"lr": matrix_lr, "betas": adam_betas, "eps": 1e-10,
                       "weight_decay": weight_decay}
            elif "wte" in path:
                cfg = {"lr": embedding_lr * dmodel_lr_scale, "betas": adam_betas,
                       "eps": 1e-10, "weight_decay": 0.0}
            elif "value_embeds" in path:
                cfg = {"lr": embedding_lr * dmodel_lr_scale, "betas": adam_betas,
                       "eps": 1e-10, "weight_decay": 0.0}
            elif "lm_head" in path:
                cfg = {"lr": unembedding_lr * dmodel_lr_scale, "betas": adam_betas,
                       "eps": 1e-10, "weight_decay": 0.0}
            elif "resid_lambdas" in path:
                cfg = {"lr": scalar_lr * 0.01, "betas": adam_betas,
                       "eps": 1e-10, "weight_decay": 0.0}
            elif "x0_lambdas" in path:
                cfg = {"lr": scalar_lr, "betas": (0.96, 0.95),
                       "eps": 1e-10, "weight_decay": 0.0}
            else:
                cfg = {"lr": unembedding_lr * dmodel_lr_scale, "betas": adam_betas,
                       "eps": 1e-10, "weight_decay": 0.0}
            self.param_config[path] = cfg

        self.initial_lrs = {p: c["lr"] for p, c in self.param_config.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        wd = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        s = self.adam_state[path]
        s["t"] += 1
        s["m"] = beta1 * s["m"] + (1 - beta1) * grad_f32
        s["v"] = beta2 * s["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** s["t"]
        bias2 = 1 - beta2 ** s["t"]
        denom = mx.sqrt(s["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (s["m"] / denom)
        return param_f32.astype(param.dtype)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            new_param = self._step(path, grad, flat_params[path], self.param_config[path])
            self._set_path_value(model, path, new_param)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier

    @property
    def state(self):
        arrays = []
        for s in self.adam_state.values():
            arrays.extend([s["m"], s["v"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 4
N_HEAD = 2
N_KV_HEAD = 1
N_EMBD = 256

# Optimization (per-param-group LRs from solo run)
BATCH_SIZE = 8
MATRIX_LR = 0.04
EMBEDDING_LR = 1.5
UNEMBEDDING_LR = 0.004
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.04
ADAM_BETAS = (0.65, 0.9)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.2
FINAL_LR_FRAC = 0.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_kv_head=N_KV_HEAD,
    n_embd=N_EMBD,
)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

num_params = model.num_params()
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

total_batch_tokens = BATCH_SIZE * MAX_SEQ_LEN
print(f"Tokens per optimizer step: {total_batch_tokens:,}")

# Optimizer (per-param-group)
optimizer = AdamW(
    model,
    matrix_lr=MATRIX_LR,
    embedding_lr=EMBEDDING_LR,
    unembedding_lr=UNEMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
)

# Loss function
def loss_fn(model, x, y):
    logits = model(x)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = y.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))

loss_and_grad = nn.value_and_grad(model, loss_fn)

# LR schedule
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

train_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")

print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
warmup_steps = 2

while True:
    t0 = time.time()

    x, y, epoch = next(train_loader)
    loss, grads = loss_and_grad(model, x, y)
    mx.eval(loss, grads)

    # Update LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_mult = get_lr_multiplier(progress)
    optimizer.set_lr_multiplier(lr_mult)

    # Apply gradients
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    avg_loss = loss.item()

    # Fast fail
    if math.isnan(avg_loss) or avg_loss > 100:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > warmup_steps:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * avg_loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_tokens / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    lr_display = MATRIX_LR * lr_mult
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {lr_display:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    step += 1

    if step > warmup_steps and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * total_batch_tokens

# Final eval
print("Evaluating...")
val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)

# Final summary
t_end = time.time()
peak_mem_bytes = mx.metal.get_peak_memory()
peak_mem_mb = peak_mem_bytes / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_mem_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

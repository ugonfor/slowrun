#!/usr/bin/env python3
"""
Evaluate ensemble by averaging logits from multiple model checkpoints.
Can be used to evaluate any combination of checkpoints.

Usage:
    torchrun --standalone --nproc_per_node=1 evaluate_ensemble.py checkpoints/model_0.pt checkpoints/model_1.pt ...
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import math
import sys
import argparse
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tiktoken

# =============================================================================
# Model (must match training code exactly)
# =============================================================================

MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
EVAL_TOKENS = 10_000_000

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs, flush=True)

# Flash Attention
def _load_flash_attn():
    if not torch.cuda.is_available():
        return None, None
    major, _ = torch.cuda.get_device_capability()
    if major == 9:
        try:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            from kernels import get_kernel
            fa3 = get_kernel('varunneal/flash-attention-3').flash_attn_interface
            return fa3, "fa3"
        except Exception:
            pass
    try:
        import flash_attn as fa2_module
        return fa2_module, "fa2"
    except ImportError:
        pass
    return None, None

_fa_module, _fa_version = _load_flash_attn()

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    return _fa_module.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

def norm(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = 30
    n_head: int = 14
    n_kv_head: int = 14
    n_embd: int = 1792
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.0  # No dropout during eval

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.relu(self.c_fc(x)).square()))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            if loss_reduction == 'none':
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits

# =============================================================================
# DataLoader
# =============================================================================

class DataLoader:
    def __init__(self, filepath, B, T, device="cuda", seed=0):
        data = torch.load(filepath, weights_only=True)
        chunks = data['chunks']
        valid_counts = data['valid_counts']
        file_B = data['batch_size']
        sequence_size = data['sequence_size']
        assert sequence_size == T + 1

        all_seqs = []
        for chunk, vc in zip(chunks, valid_counts):
            rows = chunk.view(file_B, sequence_size)[:vc]
            all_seqs.append(rows)
        all_seqs = torch.cat(all_seqs, dim=0).long()

        _, rank, _, world_size = get_dist_info()
        seqs_per_step = B * world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps, world_size, B, -1)

        self.rank_data = all_seqs[:, rank].contiguous()
        self.num_steps = num_steps
        self.device = device
        self.pos = 0
        self.epoch = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Paths to model checkpoints")
    parser.add_argument("--data-dir", type=str, default="fineweb_data")
    parser.add_argument("--n_layer", type=int, default=30)
    parser.add_argument("--n_head", type=int, default=14)
    parser.add_argument("--n_embd", type=int, default=1792)
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and torch.cuda.is_available():
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(vocab_size):
        token_bytes_list.append(0 if i == eot_id else len(encoder.decode_single_token_bytes(i)))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    config = GPTConfig(vocab_size=vocab_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)

    # Evaluate each individual model and then the ensemble
    print0(f"\nEvaluating {len(args.checkpoints)} checkpoint(s)...")
    B_eval = 2
    val_path = os.path.join(args.data_dir, "fineweb_val.pt")
    eval_steps = EVAL_TOKENS // (B_eval * MAX_SEQ_LEN * ddp_world_size)

    # Individual evaluations
    for ckpt_path in args.checkpoints:
        print0(f"\n  Evaluating {ckpt_path}...")
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        val_loader = DataLoader(val_path, B_eval, MAX_SEQ_LEN, device=device)
        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        total_tokens = torch.tensor(0, dtype=torch.int64, device=device)
        total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

        batch_iter = iter(val_loader)
        for _ in range(eval_steps):
            x, y, _ = next(batch_iter)
            with autocast_ctx:
                loss2d = model(x, y, loss_reduction='none').view(-1)
            y_flat = y.view(-1)
            mask = y_flat != -1
            total_loss += loss2d[mask].sum().double()
            total_tokens += mask.sum()
            nb = token_bytes[y_flat]
            total_nats += (loss2d * (nb > 0)).sum().double()
            total_bytes += nb.sum()

        if dist.is_initialized():
            for t in [total_loss, total_tokens, total_nats, total_bytes]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        val_loss = total_loss.item() / total_tokens.item()
        val_bpb = total_nats.item() / (math.log(2) * total_bytes.item())
        print0(f"    Val Loss: {val_loss:.6f} | Val BPB: {val_bpb:.6f}")
        del model, state_dict
        torch.cuda.empty_cache()

    # Ensemble evaluation (if more than 1 checkpoint)
    if len(args.checkpoints) > 1:
        print0(f"\n  Evaluating ensemble ({len(args.checkpoints)} models, logit averaging)...")
        models = []
        for ckpt_path in args.checkpoints:
            with torch.device("meta"):
                model = GPT(config)
            model.to_empty(device=device)
            model.init_weights()
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
            del state_dict

        B_ens = 1  # Small batch for memory with multiple models
        val_loader = DataLoader(val_path, B_ens, MAX_SEQ_LEN, device=device)
        ens_eval_steps = EVAL_TOKENS // (B_ens * MAX_SEQ_LEN * ddp_world_size)

        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        total_tokens = torch.tensor(0, dtype=torch.int64, device=device)
        total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

        batch_iter = iter(val_loader)
        for _ in range(ens_eval_steps):
            x, y, _ = next(batch_iter)
            logits_sum = None
            for model in models:
                with autocast_ctx:
                    logits = model(x).float()
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum.add_(logits)
                del logits
            avg_logits = logits_sum / len(models)

            flat_logits = avg_logits.view(-1, avg_logits.size(-1))
            y_flat = y.view(-1)
            loss2d = F.cross_entropy(flat_logits, y_flat, ignore_index=-1, reduction='none')
            mask = y_flat != -1
            total_loss += loss2d[mask].sum().double()
            total_tokens += mask.sum()
            nb = token_bytes[y_flat]
            total_nats += (loss2d * (nb > 0)).sum().double()
            total_bytes += nb.sum()
            del logits_sum, avg_logits

        if dist.is_initialized():
            for t in [total_loss, total_tokens, total_nats, total_bytes]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        ens_loss = total_loss.item() / total_tokens.item()
        ens_bpb = total_nats.item() / (math.log(2) * total_bytes.item())
        print0(f"\n  >>> Ensemble Val Loss: {ens_loss:.6f} | Val BPB: {ens_bpb:.6f}")

        del models
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

import math
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)
    return mask


def per_query_entropy_regularizer(attn: torch.Tensor, lam: float, eps: float = 1e-9) -> torch.Tensor:
    ent = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(-1)  # (B,H,T)
    return -lam * ent.mean()


def segment_coverage_regularizer(attn: torch.Tensor, idx: torch.Tensor, T: int, num_segments: int, lam: float,
                                 alpha_late: float = 0.0, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, _, k = attn.shape
    S = int(num_segments)
    seg = torch.clamp((idx * S) // max(T, 1), 0, S - 1)  # (B,H,T,k)

    m = attn.new_zeros((B, H, S))
    m.scatter_add_(dim=-1, index=seg.reshape(B, H, -1), src=attn.reshape(B, H, -1))
    m = m / (T + eps)
    m = m / (m.sum(-1, keepdim=True) + eps)

    s = torch.arange(S, device=attn.device, dtype=attn.dtype)
    if float(alpha_late) != 0.0:
        w = 1.0 + float(alpha_late) * (s / max(S - 1, 1))
        pi = w / w.sum()
    else:
        pi = torch.full((S,), 1.0 / S, device=attn.device, dtype=attn.dtype)

    kl = (m.clamp_min(eps) * (m.clamp_min(eps).log() - pi.clamp_min(eps).log())).sum(-1)  # (B,H)
    loss = float(lam) * kl.mean()

    last_q = int(math.ceil(0.75 * S))
    late_mass = m[:, :, last_q:].sum(-1).mean()

    return loss, kl.mean(), late_mass


class SparseTopKSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, k: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.k = int(k)
        self.dropout = float(dropout)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.out = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,T,D)
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)

        Q = split(q)
        K = split(k)
        V = split(v)

        # scores full (prototype)
        scores = torch.einsum("bhtd,bhsd->bhts", Q, K) / math.sqrt(self.head_dim)  # (B,H,T,T)

        if attn_mask is not None:
            # attn_mask: (B,T) where 1 is keep
            # mask out keys where attention_mask==0
            key_mask = (attn_mask == 0).view(B, 1, 1, T)
            scores = scores.masked_fill(key_mask, float("-inf"))

        ksel = min(self.k, T)
        idx = scores.topk(ksel, dim=-1).indices  # (B,H,T,k)

        Ksel = K.gather(2, idx[..., None].expand(B, self.num_heads, T, ksel, self.head_dim))
        Vsel = V.gather(2, idx[..., None].expand(B, self.num_heads, T, ksel, self.head_dim))

        logits = (Q.unsqueeze(-2) * Ksel).sum(-1) / math.sqrt(self.head_dim)  # (B,H,T,k)
        attn = F.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = (attn.unsqueeze(-1) * Vsel).sum(-2)  # (B,H,T,hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return out, attn, idx


class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.dropout = float(dropout)
        self.q = nn.Linear(self.d_model, self.d_model)
        self.kv = nn.Linear(self.d_model, 2 * self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: Optional[torch.Tensor]):
        B, T, D = x.shape
        S = memory.shape[1]
        Q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        kv = self.kv(memory)
        K, V = kv.chunk(2, dim=-1)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhtd,bhsd->bhts", Q, K) / math.sqrt(self.head_dim)
        if memory_mask is not None:
            key_mask = (memory_mask == 0).view(B, 1, 1, S)
            scores = scores.masked_fill(key_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("bhts,bhsd->bhtd", attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = float(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, k: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SparseTopKSelfAttention(d_model=d_model, num_heads=num_heads, k=k, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, attention_mask):
        h = self.ln1(x)
        aout, attn, idx = self.attn(h, attn_mask=attention_mask)
        x = x + F.dropout(aout, p=self.dropout, training=self.training)
        h2 = self.ln2(x)
        fout = self.ff(h2)
        x = x + F.dropout(fout, p=self.dropout, training=self.training)
        return x, attn, idx


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, memory, memory_mask):
        B, T, _ = x.shape
        h = self.ln1(x)
        causal = _make_causal_mask(T, x.device)
        aout, _ = self.self_attn(h, h, h, attn_mask=causal)
        x = x + F.dropout(aout, p=self.dropout, training=self.training)

        h2 = self.ln2(x)
        cout = self.cross_attn(h2, memory=memory, memory_mask=memory_mask)
        x = x + F.dropout(cout, p=self.dropout, training=self.training)

        h3 = self.ln3(x)
        fout = self.ff(h3)
        x = x + F.dropout(fout, p=self.dropout, training=self.training)
        return x


class TransformerSummarizer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        k: int,
        topk_mode: str,
        scr_cfg,
        acr_cfg,
        max_encoder_len: int,
        max_decoder_len: int,
        label_smoothing: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.d_model = int(d_model)
        self.max_encoder_len = int(max_encoder_len)
        self.max_decoder_len = int(max_decoder_len)
        self.label_smoothing = float(label_smoothing)

        num_heads = 8
        assert self.d_model % num_heads == 0

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.pos_emb_enc = nn.Embedding(self.max_encoder_len, self.d_model)
        self.pos_emb_dec = nn.Embedding(self.max_decoder_len, self.d_model)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model=self.d_model, num_heads=num_heads, k=int(k), dropout=float(dropout))
            for _ in range(int(num_encoder_layers))
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model=self.d_model, num_heads=num_heads, dropout=float(dropout))
            for _ in range(int(num_decoder_layers))
        ])

        self.ln_enc = nn.LayerNorm(self.d_model)
        self.ln_dec = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.scr_enabled = bool(getattr(scr_cfg, "enabled", False))
        self.scr_lambda = float(getattr(scr_cfg, "lambda", 0.0) or 0.0)
        self.num_segments = int(getattr(scr_cfg, "num_segments", 32) or 32)
        self.alpha_late = float(getattr(scr_cfg, "alpha_late", 0.0) or 0.0)

        self.acr_enabled = bool(getattr(acr_cfg, "enabled", False))
        self.acr_lambda = float(getattr(acr_cfg, "lambda", 0.0) or 0.0)

        assert topk_mode == "exact", "Only exact top-k prototype supported"

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_attn_metrics: bool = False):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb_enc(pos)[None, :, :]

        total_scr = x.new_tensor(0.0)
        total_acr = x.new_tensor(0.0)
        cov_kl_vals = []
        late_vals = []

        for layer in self.enc_layers:
            x, attn, idx = layer(x, attention_mask=attention_mask)
            if return_attn_metrics and (self.scr_enabled or self.acr_enabled):
                if self.scr_enabled and self.scr_lambda > 0:
                    scr_loss, cov_kl, late_mass = segment_coverage_regularizer(
                        attn=attn,
                        idx=idx,
                        T=T,
                        num_segments=self.num_segments,
                        lam=self.scr_lambda,
                        alpha_late=self.alpha_late,
                    )
                    total_scr = total_scr + scr_loss
                    cov_kl_vals.append(cov_kl.detach())
                    late_vals.append(late_mass.detach())
                if self.acr_enabled and self.acr_lambda != 0.0:
                    total_acr = total_acr + per_query_entropy_regularizer(attn, lam=self.acr_lambda)

        x = self.ln_enc(x)

        attn_metrics = None
        if return_attn_metrics and (self.scr_enabled or self.acr_enabled):
            cov = torch.stack(cov_kl_vals).mean() if len(cov_kl_vals) else x.new_tensor(float("nan"))
            late = torch.stack(late_vals).mean() if len(late_vals) else x.new_tensor(float("nan"))
            attn_metrics = {
                "segment_coverage_kl": float(cov.item()),
                "late_segment_mass": float(late.item()),
                "scr_loss": float(total_scr.item()) if self.scr_enabled else 0.0,
                "acr_loss": float(total_acr.item()) if self.acr_enabled else 0.0,
            }

        return x, total_scr, total_acr, attn_metrics

    def decode(self, decoder_input_ids: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor):
        B, T = decoder_input_ids.shape
        pos = torch.arange(T, device=decoder_input_ids.device)
        x = self.token_emb(decoder_input_ids) + self.pos_emb_dec(pos)[None, :, :]
        for layer in self.dec_layers:
            x = layer(x, memory=memory, memory_mask=memory_mask)
        x = self.ln_dec(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                return_attn_metrics: bool = False) -> Dict[str, Any]:
        # labels are ONLY used for loss; decoder inputs use shifted labels (teacher forcing)
        memory, scr_loss, acr_loss, attn_metrics = self.encode(input_ids, attention_mask, return_attn_metrics=return_attn_metrics)

        decoder_input_ids = torch.where(labels[:, :-1] == -100, self.pad_token_id, labels[:, :-1]).contiguous()
        logits = self.decode(decoder_input_ids, memory=memory, memory_mask=attention_mask)

        target = labels[:, 1:].contiguous()
        vocab = logits.size(-1)

        loss = F.cross_entropy(
            logits.view(-1, vocab),
            target.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        loss = loss + scr_loss + acr_loss

        out = {"loss": loss}
        if return_attn_metrics:
            out["attn_metrics"] = attn_metrics
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, num_beams: int = 1):
        # greedy decoding for simplicity (num_beams ignored)
        memory, _, _, _ = self.encode(input_ids, attention_mask, return_attn_metrics=False)
        B = input_ids.size(0)
        device = input_ids.device

        ys = torch.full((B, 1), self.eos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros((B,), dtype=torch.bool, device=device)
        for _ in range(int(max_new_tokens)):
            logits = self.decode(ys, memory=memory, memory_mask=attention_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            ys = torch.cat([ys, next_token[:, None]], dim=1)
            finished = finished | (next_token == self.eos_token_id)
            if bool(finished.all()):
                break
        return ys


def _lcs_len(a: List[str], b: List[str]) -> int:
    # DP O(n*m) on tokens; used for ROUGE-L
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = dp[i - 1, j] if dp[i - 1, j] >= dp[i, j - 1] else dp[i, j - 1]
    return int(dp[n, m])


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    d = {}
    if len(tokens) < n:
        return d
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        d[ng] = d.get(ng, 0) + 1
    return d


def _rouge_n(pred: str, ref: str, n: int) -> float:
    pt = pred.split()
    rt = ref.split()
    pc = _ngram_counts(pt, n)
    rc = _ngram_counts(rt, n)
    overlap = 0
    ref_total = sum(rc.values())
    pred_total = sum(pc.values())
    for k, v in pc.items():
        overlap += min(v, rc.get(k, 0))
    if ref_total == 0 or pred_total == 0:
        return 0.0
    prec = overlap / pred_total
    rec = overlap / ref_total
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _rouge_l(pred: str, ref: str) -> float:
    pt = pred.split()
    rt = ref.split()
    lcs = _lcs_len(pt, rt)
    if len(pt) == 0 or len(rt) == 0:
        return 0.0
    prec = lcs / len(pt)
    rec = lcs / len(rt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_rouge_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    r1 = []
    r2 = []
    rl = []
    for p, r in zip(preds, refs):
        p = (p or "").strip()
        r = (r or "").strip()
        r1.append(_rouge_n(p, r, 1))
        r2.append(_rouge_n(p, r, 2))
        rl.append(_rouge_l(p, r))
    return {"rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)), "rougeL": float(np.mean(rl))}

import os
import json
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
import optuna

from transformers import AutoTokenizer

from .preprocess import build_datasets, SummarizationCollator
from .model import TransformerSummarizer, compute_rouge_scores


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _is_main_process() -> bool:
    return True


def get_precision_context(precision: str, device_type: str):
    precision = (precision or "fp32").lower()
    if precision in ("bf16", "bfloat16"):
        if device_type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if precision in ("fp16", "float16"):
        if device_type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return torch.autocast(device_type=device_type, enabled=False)


def build_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    lr = float(cfg.training.learning_rate)
    wd = float(cfg.training.weight_decay)

    no_decay = {"bias", "LayerNorm.weight"}
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    opt_name = str(cfg.training.optimizer).lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


class LinearWarmupDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if self.total_steps <= 0:
            return [base_lr for base_lr in self.base_lrs]
        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = float(step) / float(max(1, self.warmup_steps))
        else:
            denom = max(1, self.total_steps - self.warmup_steps)
            scale = max(0.0, float(self.total_steps - step) / float(denom))
        return [base_lr * scale for base_lr in self.base_lrs]


@dataclass
class EvalResult:
    loss: float
    rouge1: float
    rouge2: float
    rougeL: float
    segment_coverage_kl: float
    late_segment_mass: float


@torch.no_grad()
def run_eval(model: TransformerSummarizer, dl: DataLoader, device: torch.device, tokenizer, precision: str,
             max_new_tokens: int, num_beams: int) -> EvalResult:
    model.eval()
    losses = []
    all_preds = []
    all_refs = []
    cov_kls = []
    late_masses = []

    autocast_ctx = get_precision_context(precision, device.type)

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast_ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_attn_metrics=True,
            )
            loss = out["loss"].float().item()
        losses.append(loss)

        attn_metrics = out.get("attn_metrics", None)
        if attn_metrics is not None:
            cov_kls.append(float(attn_metrics["segment_coverage_kl"]))
            late_masses.append(float(attn_metrics["late_segment_mass"]))

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
        refs = tokenizer.batch_decode(torch.where(labels == -100, tokenizer.pad_token_id, labels), skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)

    rouge = compute_rouge_scores(all_preds, all_refs)
    cov_kl = float(np.mean(cov_kls)) if len(cov_kls) else float("nan")
    late_mass = float(np.mean(late_masses)) if len(late_masses) else float("nan")
    return EvalResult(
        loss=float(np.mean(losses)) if losses else float("nan"),
        rouge1=float(rouge["rouge1"]),
        rouge2=float(rouge["rouge2"]),
        rougeL=float(rouge["rougeL"]),
        segment_coverage_kl=cov_kl,
        late_segment_mass=late_mass,
    )


def assert_gradients_ok(model: nn.Module) -> None:
    total_norm = 0.0
    count = 0
    nonzero = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, "Gradient is None for a parameter (custom logic may have broken backprop)"
        g = p.grad.detach()
        gn = float(g.abs().sum().item())
        total_norm += gn
        count += 1
        if gn > 0:
            nonzero += 1
    assert count > 0
    assert nonzero > 0 and total_norm > 0.0, "All gradients are zero before optimizer.step()"


def suggest_from_space(trial: optuna.Trial, space: DictConfig):
    name = str(space.param_name)
    dist = str(space.distribution_type).lower()
    if dist == "loguniform":
        return trial.suggest_float(name, float(space.low), float(space.high), log=True)
    if dist == "uniform":
        return trial.suggest_float(name, float(space.low), float(space.high), log=False)
    if dist == "categorical":
        return trial.suggest_categorical(name, list(space.choices))
    raise ValueError(f"Unsupported distribution_type: {space.distribution_type}")


def apply_suggested_params(cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    for k, v in params.items():
        if k == "learning_rate":
            cfg.training.learning_rate = float(v)
        elif k == "dropout":
            cfg.training.dropout = float(v)
        elif k == "scr_lambda":
            cfg.model.regularizers.scr["lambda"] = float(v)
        elif k == "acr_lambda":
            cfg.model.regularizers.acr["lambda"] = float(v)
        elif k == "num_segments":
            cfg.model.regularizers.scr.num_segments = int(v)
        elif k == "alpha_late":
            cfg.model.regularizers.scr.alpha_late = float(v)
        else:
            # allow future extensions
            pass

    return cfg


def objective_for_optuna(base_cfg: DictConfig, seed: int, device: torch.device) -> float:
    def _objective(trial: optuna.Trial) -> float:
        params = {}
        for space in base_cfg.optuna.search_spaces:
            params[space.param_name] = suggest_from_space(trial, space)

        cfg = apply_suggested_params(base_cfg, params)
        cfg.training.epochs = 1
        cfg.training.max_steps = 50
        cfg.training.log_every_n_steps = 0
        cfg.training.eval_every_n_steps = 50
        cfg.wandb.mode = "disabled"

        val_metric = train_single_seed(cfg, seed=seed, device=device, enable_wandb=False, return_best_metric=True)
        return float(val_metric)

    return _objective


def train_single_seed(cfg: DictConfig, seed: int, device: torch.device, enable_wandb: bool,
                      return_best_metric: bool = False) -> float:
    set_seed(seed)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_name = getattr(cfg.dataset.preprocessing, "tokenizer", "default")
    if tokenizer_name == "default":
        tokenizer_name = "google/long-t5-local-base"  # sentencepiece, supports long inputs; used as tokenizer only

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=".cache/")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert tokenizer.pad_token_id is not None

    max_encoder_len = int(cfg.dataset.preprocessing.max_encoder_len)
    max_decoder_len = int(cfg.dataset.preprocessing.max_decoder_len)

    ds = build_datasets(cfg, tokenizer=tokenizer)

    collator = SummarizationCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    dl_train = DataLoader(
        ds["train"],
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds["validation"],
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model = TransformerSummarizer(
        vocab_size=int(tokenizer.vocab_size) if getattr(tokenizer, "vocab_size", None) is not None else int(len(tokenizer)),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else int(tokenizer.sep_token_id) if tokenizer.sep_token_id is not None else int(tokenizer.pad_token_id),
        d_model=int(cfg.model.architecture.d_model),
        num_encoder_layers=int(cfg.model.architecture.encoder_layers),
        num_decoder_layers=int(cfg.model.architecture.decoder_layers),
        dropout=float(cfg.training.dropout),
        k=int(cfg.model.architecture.attention.k),
        topk_mode=str(cfg.model.architecture.attention.topk_mode),
        scr_cfg=cfg.model.regularizers.scr,
        acr_cfg=cfg.model.regularizers.acr,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        label_smoothing=float(cfg.training.label_smoothing),
    ).to(device)

    # Post-init assertions
    assert model.pad_token_id == tokenizer.pad_token_id
    assert model.vocab_size > 1000
    test_out_dim = model.lm_head.out_features
    assert test_out_dim == model.vocab_size

    optimizer = build_optimizer(cfg, model)

    steps_per_epoch = max(1, len(dl_train) // int(cfg.training.grad_accum_steps))
    max_steps_cfg = cfg.training.max_steps
    if max_steps_cfg is None or str(max_steps_cfg).lower() == "null":
        total_steps = int(cfg.training.epochs) * steps_per_epoch
    else:
        total_steps = int(max_steps_cfg)

    scheduler = LinearWarmupDecay(optimizer, warmup_steps=int(cfg.training.warmup_steps), total_steps=total_steps)

    scaler = None
    use_amp = device.type == "cuda" and str(cfg.training.precision).lower() in ("bf16", "fp16", "float16", "bfloat16")
    if device.type == "cuda" and str(cfg.training.precision).lower() in ("fp16", "float16"):
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    global_step = 0
    best_val = -1e9

    autocast_ctx = get_precision_context(str(cfg.training.precision), device.type)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    tokens_seen = 0

    for epoch in range(int(cfg.training.epochs)):
        for it, batch in enumerate(dl_train):
            if cfg.training.max_steps is not None and str(cfg.training.max_steps).lower() != "null":
                if global_step >= int(cfg.training.max_steps):
                    break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if global_step == 0:
                assert input_ids.ndim == 2 and labels.ndim == 2
                assert input_ids.shape[0] == labels.shape[0]
                assert input_ids.shape[1] == max_encoder_len
                assert labels.shape[1] == max_decoder_len

            tokens_seen += int(attention_mask.sum().item())

            with autocast_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_attn_metrics=True,
                )
                loss = out["loss"] / int(cfg.training.grad_accum_steps)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % int(cfg.training.grad_accum_steps) == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                if float(cfg.training.gradient_clip) is not None and float(cfg.training.gradient_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.gradient_clip))

                # Pre-optimizer critical assertion
                assert_gradients_ok(model)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # metrics
                lr = float(scheduler.get_last_lr()[0])
                step_time = max(1e-6, time.time() - t0)
                tps = float(tokens_seen / step_time)

                log_payload = {
                    "step": global_step,
                    "epoch": epoch + (it / max(1, len(dl_train))),
                    "train_loss": float(out["loss"].float().item()),
                    "lr": lr,
                    "tokens_per_sec": tps,
                }

                if out.get("attn_metrics") is not None:
                    for k, v in out["attn_metrics"].items():
                        log_payload[f"train_{k}"] = float(v)

                if device.type == "cuda":
                    log_payload["peak_gpu_mem"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))

                if enable_wandb and cfg.training.log_every_n_steps and int(cfg.training.log_every_n_steps) > 0:
                    if global_step % int(cfg.training.log_every_n_steps) == 0:
                        wandb.log(log_payload, step=global_step)

                if cfg.training.eval_every_n_steps and int(cfg.training.eval_every_n_steps) > 0:
                    if global_step % int(cfg.training.eval_every_n_steps) == 0:
                        val = run_eval(
                            model,
                            dl_val,
                            device,
                            tokenizer,
                            precision=str(cfg.training.precision),
                            max_new_tokens=max_decoder_len,
                            num_beams=1,
                        )
                        val_payload = {
                            "step": global_step,
                            "val_loss": val.loss,
                            "val_rouge1": val.rouge1,
                            "val_rouge2": val.rouge2,
                            "val_rougeL": val.rougeL,
                            "val_segment_coverage_kl": val.segment_coverage_kl,
                            "val_late_segment_mass": val.late_segment_mass,
                        }
                        if enable_wandb:
                            wandb.log(val_payload, step=global_step)

                        if val.rougeL > best_val:
                            best_val = float(val.rougeL)

        if cfg.training.max_steps is not None and str(cfg.training.max_steps).lower() != "null":
            if global_step >= int(cfg.training.max_steps):
                break

    if math.isinf(best_val) or math.isnan(best_val) or best_val < -1e8:
        # if eval never ran
        val = run_eval(
            model,
            dl_val,
            device,
            tokenizer,
            precision=str(cfg.training.precision),
            max_new_tokens=max_decoder_len,
            num_beams=1,
        )
        best_val = float(val.rougeL)

    return best_val if return_best_metric else best_val


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # cfg.run is selected by hydra defaults in config.yaml
    # mode adjustments already applied in main.py; but keep defensive here
    if str(cfg.mode) == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.max_steps = 2
        cfg.training.eval_every_n_steps = 1
        cfg.training.log_every_n_steps = 1

    run_id = str(cfg.run.run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optuna (no wandb intermediate)
    best_params = None
    if bool(cfg.optuna.enabled) and int(cfg.optuna.n_trials) > 0 and str(cfg.wandb.mode) != "disabled":
        study = optuna.create_study(direction=str(cfg.optuna.direction))
        obj = objective_for_optuna(cfg, seed=int(cfg.training.seed), device=device)
        study.optimize(obj, n_trials=int(cfg.optuna.n_trials))
        best_params = dict(study.best_params)

    final_cfg = cfg
    if best_params is not None:
        final_cfg = apply_suggested_params(cfg, best_params)

    enable_wandb = str(final_cfg.wandb.mode) != "disabled"
    if enable_wandb:
        wandb.init(
            entity=str(final_cfg.wandb.entity),
            project=str(final_cfg.wandb.project),
            id=run_id,
            config=OmegaConf.to_container(final_cfg, resolve=True),
            resume="allow",
        )

    seeds = list(final_cfg.training.seeds) if getattr(final_cfg.training, "seeds", None) is not None else [int(final_cfg.training.seed)]
    per_seed = {}
    for s in seeds:
        best_val = train_single_seed(final_cfg, seed=int(s), device=device, enable_wandb=enable_wandb, return_best_metric=False)
        per_seed[str(s)] = {"best_val_rougeL": float(best_val)}
        if enable_wandb:
            wandb.log({"seed": int(s), "best_val_rougeL_seed": float(best_val)})

    mean_best = float(np.mean([v["best_val_rougeL"] for v in per_seed.values()]))
    std_best = float(np.std([v["best_val_rougeL"] for v in per_seed.values()]))

    if enable_wandb:
        wandb.summary["best_val_rougeL_mean"] = mean_best
        wandb.summary["best_val_rougeL_std"] = std_best
        if best_params is not None:
            wandb.summary["optuna_best_params"] = json.dumps(best_params)
        print(wandb.run.get_url())
        wandb.finish()
    else:
        print(f"trial_mode complete for {run_id}: best_val_rougeL_mean={mean_best:.4f} std={std_best:.4f}")


if __name__ == "__main__":
    main()

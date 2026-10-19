#!/usr/bin/env python3
"""M2-v3 unified 17-organ trainer using the canonical M2-7 contract.

Smoke:  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
          run_m2_v3_trainer.py --mode smoke20 ...
"""
from __future__ import annotations

import argparse, csv, gzip, json, math, os, random, sys, time, logging, hashlib
from datetime import timedelta
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ── Import M2 model from same directory ────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from m2_v3_model import M2Pool32TypedEvidenceRawLog1pV3

# ── Default paths ──────────────────────────────────────────────
_PIPE = (__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + '/new_pipeline_800gb')
_RUN = f"{_PIPE}/full_run_firstv1_rawlog1p_20260703_110639"
_CORRECTED = (__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + '/deepseek_outcome/m2_v3_17organ_gene_identity_fix_20260909_225119/result')
_P256 = f"{_CORRECTED}/program256_canonical_identity_m16/gene_to_program_m16_overlay.pt"
_HVG384 = f"{_CORRECTED}/corrected_gene_assets/section_hvg384_ids.pt"


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="M2-v3 pool32/homo12/hetero8 trainer (shared raw-log1p cache)")
    p.add_argument("--mode", choices=["smoke20", "short_train", "full", "eval"], default="smoke20")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=96)
    # data
    p.add_argument("--rowmap_root", default=f"{_RUN}/step3_rowmap")
    p.add_argument("--feature_store", default=f"{_RUN}/step1_features/master_feature_store.pt")
    p.add_argument("--gene_chunk_manifest", default=f"{_CORRECTED}/corrected_gene_assets/gene_label_chunks_manifest.tsv")
    p.add_argument("--neighbor_overlay", default=f"{_RUN}/step4_overlays/m2_pool32_homo_hetero_overlay.pt")
    p.add_argument("--v3_split_root", default=f"{_PIPE}/step10_split_v3_section_size_tech_balanced_20260704_073545",
                   help="Directory with v3 train/val/test_row_indices.pt (preferred over rowmap split)")
    p.add_argument("--program_overlay_path", default=_P256,
                   help="Program-256 raw-log1p overlay dict with gene_to_program_ids/mask; default is M16 P256")
    p.add_argument("--program_ids_path", default="", help="Legacy two-file Program ids path; disabled by default")
    p.add_argument("--program_mask_path", default="", help="Legacy two-file Program mask path; disabled by default")
    # sampling
    p.add_argument("--target_k", type=int, default=1024)
    p.add_argument("--target_chunk_size", type=int, default=2048,
                   help="Forward/loss chunk size along target-gene dimension for full all_gene_20 batches")
    p.add_argument("--prompt_k", type=int, default=0, help="Deprecated fixed prompt_k; dynamic 8%% prompt is computed from target_k")
    p.add_argument("--prompt_ratio", type=float, default=0.08)
    p.add_argument("--prompt_k_cap", type=int, default=64)
    p.add_argument("--prompt_start_step", type=int, default=5000)
    p.add_argument("--hvg_ratio", type=float, default=0.7)
    p.add_argument("--sampler_schedule", choices=["hvg384_70_allgene_20_longtail_10"], default="hvg384_70_allgene_20_longtail_10")
    p.add_argument("--high_priority_source", choices=["section_HVG384"], default="section_HVG384")
    p.add_argument("--section_hvg_path", default=_HVG384)
    p.add_argument("--prompt_target_disjoint", action="store_true", default=True)
    # model
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--feature_dim", type=int, default=1536)
    p.add_argument("--alpha_init", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gene_bias", action="store_true", default=True)
    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=42)
    # eval / ckpt
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--eval_micro_batch_rows_per_rank", type=int, default=128,
                   help="Eval micro-batch rows per rank; does not change eval sample universe")
    p.add_argument("--eval_rows_per_rank", type=int, default=None,
                   help="Deprecated alias for --eval_micro_batch_rows_per_rank; does not truncate eval universe")
    p.add_argument("--eval_max_rows", type=int, default=None,
                   help="Optional smoke/debug-only eval universe cap per rank; None means no truncation")
    p.add_argument("--debug_eval_rows", type=int, default=None,
                   help="Alias for --eval_max_rows for debug/smoke only; forbidden in full mode")
    p.add_argument("--eval_sampler_mode", choices=["all_gene_20", "high_priority_70", "long_tail_10", "train_schedule"],
                   default="all_gene_20")
    p.add_argument("--eval_ablation_mode", choices=["full", "image_zero", "image_shuffle", "disable_neighbor", "neighbor_zero", "neighbor_shuffle", "disable_candidate", "disable_homo", "disable_hetero", "organ_tech_zero", "organ_zero", "tech_zero", "disable_program_prior", "program_prior_off", "disable_prompt", "disable_gene_bias", "gene_id_shuffle"], default="full")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--output_root", default="")
    p.add_argument("--save_checkpoint", action="store_true", default=False)
    p.add_argument("--save_at_steps", default="", help="Comma-separated local/global steps for rank0 model checkpoint saves, e.g. 5000")
    p.add_argument("--checkpoint_interval", type=int, default=0, help="Optional periodic rank0 model checkpoint interval")
    p.add_argument("--save_full_state", action="store_true", default=False,
                   help="Atomically rotate a fault-tolerant model/AdamW/RNG/iterator checkpoint")
    p.add_argument("--resume_full_state", action="store_true", default=False,
                   help="Strictly restore AdamW, early-stop, RNG, and per-rank iterator state")
    p.add_argument("--full_state_filename", default="last_full_state_checkpoint.pt")
    p.add_argument("--early_stopping", action="store_true", default=False)
    p.add_argument("--early_stop_start_step", type=int, default=10000)
    p.add_argument("--early_stop_patience_evals", type=int, default=20)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0005)
    p.add_argument("--write_best_model", action="store_true", default=False)
    p.add_argument("--resume_checkpoint", default="",
                   help="Model checkpoint, or complete training checkpoint with --resume_full_state")
    p.add_argument("--unified_17organ_init", action="store_true", default=False,
                   help="Reproduce canonical M2-7 initialization from M1, then expand only organ rows 7..16")
    p.add_argument("--canonical_n_organs", type=int, default=7)
    p.add_argument("--expected_n_organs", type=int, default=17)
    p.add_argument("--expected_n_techs", type=int, default=4)
    p.add_argument("--resume_global_step", type=int, default=0,
                   help="Absolute global step for resumed runs; controls prompt/gene_bias/checkpoint/eval schedule")
    p.add_argument("--direct_17organ_model_only_resume", action="store_true", default=False)
    p.add_argument("--scratch17", action="store_true", default=False,
                   help="Initialize the 17-organ M2 model from seed without reading any checkpoint")
    p.add_argument("--model_init_seed", type=int, default=42,
                   help="Private model initialization seed shared by all ranks")
    p.add_argument("--prior_best_metric", type=float, default=float("-inf"))
    p.add_argument("--prior_best_step", type=int, default=0)
    p.add_argument("--prior_bad_evals", type=int, default=0)
    p.add_argument("--resume_epoch", type=int, default=0)
    p.add_argument("--allow_existing_output", action="store_true", default=False,
                   help="Allow writing into an existing non-empty output_root; default refuses to avoid overwriting runs")
    p.add_argument("--sampling_audit_interval", type=int, default=1000,
                   help="Reporting-only interval; never changes sampling probabilities")
    # DDP (auto from torchrun)
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--ddp_backend", default="nccl")
    p.add_argument("--ddp_timeout_minutes", type=int, default=60,
                   help="Recovery-only NCCL watchdog timeout; does not affect model math.")
    # target
    p.add_argument("--target_scale", default="raw_counts_log1p")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════
_log = logging.getLogger("m2v3")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)


def log(msg: str):
    r = int(os.environ.get("RANK", 0))
    _log.info(f"R{r} {msg}")


# ═══════════════════════════════════════════════════════════════
#  Deterministic helpers
# ═══════════════════════════════════════════════════════════════
def det_perm(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed & 0x7FFFFFFF)
    return torch.randperm(n, generator=g)


# ═══════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════
def load_codebooks(rowmap_root: str) -> dict:
    with open(Path(rowmap_root) / "codebooks.json") as f:
        cb = json.load(f)
    return {
        "n_organs": len(cb["organ_id"]),
        "n_techs": len(cb["tech_id"]),
        "n_datasets": len(cb["dataset_id"]),
        "n_ngs": len(cb["normalization_group_id"]),
        "raw": cb,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint without exposing a partially serialized destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _capture_rank_runtime_state(rank: int, device: torch.device, epoch: int,
                                batches_consumed_in_epoch: int) -> dict:
    state = {
        "rank": int(rank),
        "epoch": int(epoch),
        "batches_consumed_in_epoch": int(batches_consumed_in_epoch),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda_rng_state"] = torch.cuda.get_rng_state(device).cpu()
    else:
        state["torch_cuda_rng_state"] = None
    return state


def _restore_rank_runtime_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    cuda_state = state.get("torch_cuda_rng_state")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state.cpu(), device)


def _state_sha256(state: dict[str, torch.Tensor], keys=None) -> str:
    h = hashlib.sha256()
    for key in sorted(state if keys is None else keys):
        h.update(key.encode())
        h.update(_tensor_sha256(state[key]).encode())
    return h.hexdigest()


def _partial_load_canonical_m1(model: nn.Module, checkpoint_path: str, device) -> dict:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume_checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("candidate_branch.", "homo_branch.", "hetero_branch.")
    allowed_unexpected_prefixes = ("neighbor_attn.", "decoder.")
    bad_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    bad_unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected_prefixes)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"canonical M1 partial-load mismatch missing={bad_missing[:20]} "
            f"unexpected={bad_unexpected[:20]}"
        )
    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "initialized_m2_key_count": len(missing),
        "discarded_m1_neighbor_key_count": sum(key.startswith("neighbor_attn.") for key in unexpected),
        "discarded_decoder_key_count": sum(key.startswith("decoder.") for key in unexpected),
        "optimizer_restored": False,
        "early_stop_state_restored": False,
    }


def build_unified_17organ_model(args, cb, n_genes, prog_ids, prog_mask, n_prog, device, r0):
    """Reproduce the M2-7 start state, then expand only the organ vocabulary.

    Model initialization is performed under a private seed-42 RNG scope on every
    rank. DDP therefore receives the exact rank-0 canonical start state without
    perturbing subsequent rank-specific sampler RNG streams.
    """
    if not args.unified_17organ_init:
        raise RuntimeError("canonical M2-17 requires --unified_17organ_init")
    if not args.resume_checkpoint:
        raise RuntimeError("canonical M2-17 requires the frozen M1 step68000 checkpoint")
    if cb["n_organs"] != int(args.expected_n_organs):
        raise RuntimeError(f"expected {args.expected_n_organs} organs, found {cb['n_organs']}")
    if cb["n_techs"] != int(args.expected_n_techs):
        raise RuntimeError(f"expected {args.expected_n_techs} technologies, found {cb['n_techs']}")
    if int(args.canonical_n_organs) != 7:
        raise RuntimeError("canonical_n_organs must remain 7")

    model_kwargs = dict(
        n_genes=n_genes,
        n_techs=cb["n_techs"],
        gene_to_program_ids=prog_ids,
        gene_to_program_mask=prog_mask,
        n_programs=n_prog,
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        alpha_init=args.alpha_init,
        gene_bias=args.gene_bias,
        dropout=args.dropout,
    )
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(args.base_seed))
        canonical = M2Pool32TypedEvidenceRawLog1pV3(
            n_organs=int(args.canonical_n_organs), **model_kwargs
        ).to(device)
        load_audit = _partial_load_canonical_m1(canonical, args.resume_checkpoint, device)
        canonical_state = canonical.state_dict()
        canonical_hash = _state_sha256(canonical_state)

        torch.manual_seed(int(args.base_seed))
        expanded = M2Pool32TypedEvidenceRawLog1pV3(
            n_organs=cb["n_organs"], **model_kwargs
        ).to(device)
        new_rows_before = expanded.organ_emb.weight[int(args.canonical_n_organs):].detach().clone()
        with torch.no_grad():
            expanded_state = expanded.state_dict()
            for key, value in canonical_state.items():
                if key == "organ_emb.weight":
                    expanded.organ_emb.weight[: int(args.canonical_n_organs)].copy_(value)
                elif expanded_state[key].shape == value.shape:
                    expanded_state[key].copy_(value)
                else:
                    raise RuntimeError(
                        f"unexpected shared initialization shape mismatch {key}: "
                        f"canonical={tuple(value.shape)} expanded={tuple(expanded_state[key].shape)}"
                    )

        expanded_state = expanded.state_dict()
        shared_keys = [key for key in canonical_state if key != "organ_emb.weight"]
        mismatches = [
            key for key in shared_keys
            if not torch.equal(expanded_state[key], canonical_state[key])
        ]
        old_rows_equal = torch.equal(
            expanded.organ_emb.weight[: int(args.canonical_n_organs)],
            canonical.organ_emb.weight,
        )
        tech_equal = torch.equal(expanded.tech_emb.weight, canonical.tech_emb.weight)
        new_rows_preserved = torch.equal(
            expanded.organ_emb.weight[int(args.canonical_n_organs):], new_rows_before
        )
        if mismatches or not old_rows_equal or not tech_equal or not new_rows_preserved:
            raise RuntimeError(
                f"unified initialization failed mismatches={mismatches[:20]} "
                f"old_rows_equal={old_rows_equal} tech_equal={tech_equal} "
                f"new_rows_preserved={new_rows_preserved}"
            )
        audit = {
            "status": "PASS",
            "strategy": "canonical_m2_7_seed42_then_expand_organ_rows_only",
            "base_seed": int(args.base_seed),
            "canonical_n_organs": int(args.canonical_n_organs),
            "expanded_n_organs": int(cb["n_organs"]),
            "n_techs": int(cb["n_techs"]),
            "shared_tensor_count": len(shared_keys),
            "shared_tensor_mismatch_count": len(mismatches),
            "old_organ_rows_equal": old_rows_equal,
            "tech_rows_equal": tech_equal,
            "new_organ_rows_preserved_from_seed42_expanded_instance": new_rows_preserved,
            "canonical_start_state_sha256": canonical_hash,
            "new_organ_rows_sha256": _tensor_sha256(new_rows_before),
            "old_organ_rows_sha256": _tensor_sha256(canonical.organ_emb.weight),
            "tech_rows_sha256": _tensor_sha256(canonical.tech_emb.weight),
            "partial_load": load_audit,
        }
        del canonical
    if r0:
        path = Path(args.output_root) / "m2_17organ_initialization_audit.json"
        with open(path, "w") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
    log(
        f"Unified init PASS shared={audit['shared_tensor_count']} "
        f"new_organ_rows_sha256={audit['new_organ_rows_sha256'][:12]}"
    )
    return expanded, audit


def build_scratch_17organ_model(args, cb, n_genes, prog_ids, prog_mask, n_prog, device, r0):
    """Initialize M2-17 identically on every rank without disturbing runtime RNG."""
    if not args.scratch17:
        raise RuntimeError("scratch initialization requires --scratch17")
    if args.resume_checkpoint or args.resume_full_state or args.unified_17organ_init:
        raise RuntimeError("scratch initialization forbids checkpoint and unified-init inputs")
    if int(args.resume_global_step) != 0:
        raise RuntimeError("scratch initialization requires resume_global_step=0")
    if cb["n_organs"] != int(args.expected_n_organs) or cb["n_techs"] != int(args.expected_n_techs):
        raise RuntimeError(
            f"scratch requires {args.expected_n_organs} organs/{args.expected_n_techs} technologies, "
            f"got {cb['n_organs']}/{cb['n_techs']}"
        )
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(args.model_init_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.model_init_seed))
        model = M2Pool32TypedEvidenceRawLog1pV3(
            n_genes=n_genes,
            n_organs=cb["n_organs"],
            n_techs=cb["n_techs"],
            gene_to_program_ids=prog_ids,
            gene_to_program_mask=prog_mask,
            n_programs=n_prog,
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            alpha_init=args.alpha_init,
            gene_bias=args.gene_bias,
            dropout=args.dropout,
        ).to(device)
    audit = {
        "status": "PASS_SCRATCH17_INITIALIZED",
        "strategy": "private_shared_model_seed_then_restore_rank_runtime_rng",
        "model_init_seed": int(args.model_init_seed),
        "runtime_seed": int(args.seed + args.rank),
        "n_organs": int(cb["n_organs"]),
        "n_techs": int(cb["n_techs"]),
        "model_state_sha256": _state_sha256(model.state_dict()),
        "checkpoint_read": False,
        "optimizer_restored": False,
        "early_stop_state_restored": False,
    }
    if r0:
        with open(Path(args.output_root) / "scratch17_initialization_audit.json", "w") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
    return model, audit


def build_direct_17organ_model_only_resume(args, cb, n_genes, prog_ids, prog_mask, n_prog, device, r0):
    """Load a 17-organ model; train_loop optionally restores the full runtime state."""
    if not args.resume_checkpoint:
        raise RuntimeError("direct 17-organ resume requires --resume_checkpoint")
    if cb["n_organs"] != 17 or cb["n_techs"] != 4:
        raise RuntimeError(f"direct resume requires 17 organs/4 technologies, got {cb['n_organs']}/{cb['n_techs']}")
    model = M2Pool32TypedEvidenceRawLog1pV3(
        n_genes=n_genes, n_organs=17, n_techs=4,
        gene_to_program_ids=prog_ids, gene_to_program_mask=prog_mask,
        n_programs=n_prog, feature_dim=args.feature_dim, d_model=args.d_model,
        alpha_init=args.alpha_init, gene_bias=args.gene_bias, dropout=args.dropout,
    ).to(device)
    checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
    checkpoint_step = int(checkpoint.get("step", -1))
    if checkpoint_step != int(args.resume_global_step):
        raise RuntimeError(f"checkpoint step {checkpoint_step} != resume_global_step {args.resume_global_step}")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    audit = {
        "status": "PASS_MODEL_ONLY_RESUME",
        "checkpoint": str(args.resume_checkpoint),
        "checkpoint_step": checkpoint_step,
        "model_state_sha256": _state_sha256(model.state_dict()),
        "strict_load": True,
        "optimizer_restored": bool(args.resume_full_state),
        "early_stop_state_restored": bool(args.resume_full_state),
        "data_iterator_position_restored": bool(args.resume_full_state),
        "data_iterator_position_restored": False,
        "lossless_resume": False,
    }
    if r0:
        with open(Path(args.output_root) / "model_only_resume_audit.json", "w") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
    return model, audit


def load_program_ontology(overlay_path: str, ids_path: str, mask_path: str, n_genes: int, device):
    if overlay_path:
        ov = torch.load(overlay_path, map_location="cpu", weights_only=False)
        prog_ids = torch.as_tensor(ov["gene_to_program_ids"]).long()
        prog_mask = torch.as_tensor(ov["gene_to_program_mask"]).bool()
        n_programs = int(ov.get("runtime_program_count", 256))
        membership_padding = int(ov.get("membership_padding", prog_ids.shape[1]))
        program_dim = int(ov.get("program_dim", 128))
        source = "program256_overlay"
    else:
        if not ids_path or not mask_path:
            raise RuntimeError("Either --program_overlay_path or both legacy --program_ids_path/--program_mask_path are required")
        prog_ids = torch.load(ids_path, map_location="cpu").long()
        prog_mask = torch.load(mask_path, map_location="cpu").bool()
        valid = prog_ids[prog_mask]
        n_programs = int(valid.max().item()) + 1 if valid.numel() > 0 else 1
        membership_padding = int(prog_ids.shape[1])
        program_dim = None
        source = "legacy_two_file"
    assert prog_ids.shape == prog_mask.shape
    if prog_ids.shape[0] != n_genes:
        raise RuntimeError(f"Program overlay rows {prog_ids.shape[0]} != current n_genes {n_genes}; refusing silent truncate/pad")
    valid = prog_ids[prog_mask]
    if valid.numel() and (int(valid.min().item()) < 0 or int(valid.max().item()) >= n_programs):
        raise RuntimeError("Program ids outside runtime program range")
    deg0 = int((prog_mask.sum(dim=1) == 0).sum().item())
    max_degree = int(prog_mask.sum(dim=1).max().item()) if prog_mask.numel() else 0
    log(f"Program ontology: source={source} {prog_ids.shape[0]} genes x {prog_ids.shape[1]} slots, "
        f"{n_programs} programs, max_degree={max_degree}, {deg0} degree-0, program_dim={program_dim}")
    return prog_ids.to(device), prog_mask.to(device), n_programs

def load_rowmap_tensors(rowmap_root: str):
    root = Path(rowmap_root)
    rm = {}
    for name in ["row_to_feature_index", "row_to_gene_chunk_id", "row_to_gene_row_index",
                  "row_to_normalization_group_id", "row_to_dataset_id",
                  "row_to_organ_id", "row_to_tech_id", "row_to_sid",
                  "train_row_indices", "val_row_indices", "test_row_indices"]:
        p = root / f"{name}.pt"
        if p.exists():
            rm[name] = torch.load(p, map_location="cpu")
        else:
            log(f"WARNING: {p.name} missing")
    # gene_chunk_table from label_centered_rowmap
    lcr_path = root / "label_centered_rowmap.pt"
    if lcr_path.exists():
        lcr = torch.load(lcr_path, map_location="cpu", weights_only=False)
        rm["gene_chunk_table"] = lcr.get("gene_chunk_table", [])
    n_rows = rm["row_to_feature_index"].shape[0]
    log(f"Rowmap: {n_rows} rows")
    return rm


def validate_context_alignment(args, rm: dict, cb: dict, device, ddp: bool, r0: bool) -> dict:
    """Validate row-level organ/tech tensors against section metadata on rank0."""
    audit = {"status": "NOT_RUN", "rank0_only": True}
    passed = True
    if r0:
        try:
            required = ("row_to_feature_index", "row_to_sid", "row_to_organ_id", "row_to_tech_id")
            missing = [name for name in required if name not in rm]
            if missing:
                raise RuntimeError(f"missing context tensors: {missing}")
            n_rows = int(rm["row_to_feature_index"].numel())
            lengths = {name: int(rm[name].numel()) for name in required}
            if any(length != n_rows for length in lengths.values()):
                raise RuntimeError(f"context tensor length mismatch: {lengths}")
            sid = rm["row_to_sid"].long()
            organ = rm["row_to_organ_id"].long()
            tech = rm["row_to_tech_id"].long()
            raw = cb["raw"]
            n_sids, n_organs, n_techs = len(raw["sid"]), len(raw["organ_id"]), len(raw["tech_id"])
            ranges = {
                "sid": [int(sid.min()), int(sid.max())],
                "organ": [int(organ.min()), int(organ.max())],
                "tech": [int(tech.min()), int(tech.max())],
            }
            if ranges["sid"][0] < 0 or ranges["sid"][1] >= n_sids:
                raise RuntimeError(f"row_to_sid OOB: {ranges['sid']} vs {n_sids}")
            if ranges["organ"][0] < 0 or ranges["organ"][1] >= n_organs:
                raise RuntimeError(f"row_to_organ_id OOB: {ranges['organ']} vs {n_organs}")
            if ranges["tech"][0] < 0 or ranges["tech"][1] >= n_techs:
                raise RuntimeError(f"row_to_tech_id OOB: {ranges['tech']} vs {n_techs}")
            expected_organ = torch.full((n_sids,), -1, dtype=torch.long)
            expected_tech = torch.full((n_sids,), -1, dtype=torch.long)
            manifest_conflicts = 0
            with open(args.gene_chunk_manifest) as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    sid_code = raw["sid"].get(str(row["sid"]))
                    if sid_code is None:
                        continue
                    organ_code = raw["organ_id"].get(str(row["organ_id"]))
                    tech_code = raw["tech_id"].get(str(row["tech_id"]))
                    if organ_code is None or tech_code is None:
                        raise RuntimeError(f"unmapped context for sid={row['sid']}: organ={row['organ_id']} tech={row['tech_id']}")
                    if expected_organ[sid_code] not in (-1, organ_code):
                        manifest_conflicts += 1
                    if expected_tech[sid_code] not in (-1, tech_code):
                        manifest_conflicts += 1
                    expected_organ[sid_code] = int(organ_code)
                    expected_tech[sid_code] = int(tech_code)
            missing_sid_metadata = int(((expected_organ < 0) | (expected_tech < 0)).sum().item())
            organ_mismatch = int((organ != expected_organ[sid]).sum().item())
            tech_mismatch = int((tech != expected_tech[sid]).sum().item())
            combo = sid * (n_organs * n_techs) + organ * n_techs + tech
            unique_sid = torch.div(torch.unique(combo), n_organs * n_techs, rounding_mode="floor")
            section_context_conflicts = int((torch.bincount(unique_sid, minlength=n_sids) > 1).sum().item())
            passed = manifest_conflicts == missing_sid_metadata == organ_mismatch == tech_mismatch == section_context_conflicts == 0
            audit = {
                "status": "PASS" if passed else "FAIL", "rank0_only": True, "n_rows": n_rows,
                "tensor_lengths": lengths, "ranges": ranges, "n_sids": n_sids,
                "n_organs": n_organs, "n_techs": n_techs,
                "manifest_conflicts": manifest_conflicts, "missing_sid_metadata": missing_sid_metadata,
                "organ_mismatch_rows": organ_mismatch, "tech_mismatch_rows": tech_mismatch,
                "section_context_conflicts": section_context_conflicts,
                "gene_chunk_manifest": str(args.gene_chunk_manifest),
            }
        except Exception as exc:
            passed = False
            audit = {"status": "FAIL", "rank0_only": True, "error": repr(exc)}
        with open(Path(args.output_root) / "context_alignment_audit.json", "w") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
    pass_flag = torch.tensor([1 if passed else 0], dtype=torch.int32, device=device)
    if ddp:
        dist.broadcast(pass_flag, src=0)
    if int(pass_flag.item()) != 1:
        raise RuntimeError("context alignment preflight failed; see context_alignment_audit.json")
    return audit


def load_feature_store(path: str, device, half: bool = True):
    fs = torch.load(path, map_location="cpu", weights_only=False)
    feat = fs["features"]
    if half:
        feat = feat.half()
    feat = feat.to(device, non_blocking=True)
    log(f"Feature store: {feat.shape} {feat.dtype}")
    return feat


def load_gene_chunk_table(manifest_path: str) -> dict:
    table = {}
    with open(manifest_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cid = int(r["gene_chunk_id"])
            table[cid] = {"path": r["chunk_path"], "sid": r["sid"]}
    log(f"Gene chunk table: {len(table)} chunks")
    return table


def load_m2_neighbor_overlay(path: str, n_rows: int, device):
    ov = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "candidate_row_ids", "candidate_feature_indices", "candidate_valid_mask",
        "candidate_spatial_distance", "candidate_feature_cosine_sim",
        "homo_row_ids", "homo_feature_indices", "homo_valid_mask",
        "hetero_row_ids", "hetero_feature_indices", "hetero_valid_mask",
    }
    missing = sorted(required.difference(ov))
    if missing:
        raise RuntimeError(f"M2 overlay missing keys: {missing}")
    nbr = {key: torch.as_tensor(ov[key]).to(device) for key in required}
    expected = {
        "candidate_feature_indices": 32,
        "candidate_valid_mask": 32,
        "candidate_row_ids": 32,
        "homo_feature_indices": 12,
        "homo_valid_mask": 12,
        "homo_row_ids": 12,
        "hetero_feature_indices": 8,
        "hetero_valid_mask": 8,
        "hetero_row_ids": 8,
    }
    for key, width in expected.items():
        if nbr[key].shape != (n_rows, width):
            raise RuntimeError(f"M2 overlay {key} shape={tuple(nbr[key].shape)} expected={(n_rows, width)}")
    log(f"M2 index overlay: {n_rows} rows, candidate=32 homo=12 hetero=8")
    return nbr


def gather_m2_neighbors(nbr: dict, row_indices: torch.Tensor, features_gpu: torch.Tensor):
    candidate_rows = nbr["candidate_row_ids"][row_indices]
    candidate_fi = nbr["candidate_feature_indices"][row_indices].long()
    candidate_mask = nbr["candidate_valid_mask"][row_indices].bool()
    candidate_dist = nbr["candidate_spatial_distance"][row_indices].float()
    candidate_cos = nbr["candidate_feature_cosine_sim"][row_indices].float()

    def selected(prefix: str):
        selected_rows = nbr[f"{prefix}_row_ids"][row_indices]
        selected_fi = nbr[f"{prefix}_feature_indices"][row_indices].long()
        selected_mask = nbr[f"{prefix}_valid_mask"][row_indices].bool()
        matches = selected_rows[:, :, None].eq(candidate_rows[:, None, :])
        found = matches.any(dim=-1) & selected_rows.ge(0)
        rank = matches.to(torch.int64).argmax(dim=-1)
        candidate_valid = candidate_mask.gather(1, rank)
        valid = selected_mask & found & candidate_valid & selected_fi.ge(0)
        distance = candidate_dist.gather(1, rank)
        cosine = candidate_cos.gather(1, rank)
        feature = features_gpu[selected_fi.clamp(min=0)].float()
        feature = feature * valid.unsqueeze(-1).to(feature.dtype)
        return feature, valid, distance, cosine

    candidate_img = features_gpu[candidate_fi.clamp(min=0)].float()
    candidate_img = candidate_img * candidate_mask.unsqueeze(-1).to(candidate_img.dtype)
    homo_img, homo_mask, homo_dist, homo_cos = selected("homo")
    hetero_img, hetero_mask, hetero_dist, hetero_cos = selected("hetero")
    return {
        "candidate_img": candidate_img,
        "candidate_mask": candidate_mask,
        "candidate_dist": candidate_dist,
        "candidate_cos": candidate_cos,
        "homo_img": homo_img,
        "homo_mask": homo_mask,
        "homo_dist": homo_dist,
        "homo_cos": homo_cos,
        "hetero_img": hetero_img,
        "hetero_mask": hetero_mask,
        "hetero_dist": hetero_dist,
        "hetero_cos": hetero_cos,
    }


def load_split_indices(rowmap_root: str, v3_split_root: str, device):
    """Load pre-built train/val/test row index tensors.
    Prefer v3 split (step10) if available; fall back to rowmap split (step3).
    """
    for label, root in [("v3", v3_split_root), ("rowmap", rowmap_root)]:
        if not root:
            continue
        r = Path(root)
        tp = r / "train_row_indices.pt"
        vp = r / "val_row_indices.pt"
        sp = r / "test_row_indices.pt"
        if tp.exists() and vp.exists():
            train = torch.load(tp, map_location="cpu").long().to(device)
            val = torch.load(vp, map_location="cpu").long().to(device)
            test = torch.load(sp, map_location="cpu").long().to(device) if sp.exists() else torch.empty(0, dtype=torch.long, device=device)
            log(f"Split ({label}): train={len(train)} val={len(val)} test={len(test)}")
            return train, val, test
    raise RuntimeError(f"No split .pt files found in v3={v3_split_root} or rowmap={rowmap_root}")


# ═══════════════════════════════════════════════════════════════
#  GPU chunk loader
# ═══════════════════════════════════════════════════════════════
class GPUChunkLoader:
    def __init__(self, chunk_table: dict, device):
        self.chunk_table = chunk_table
        self.device = device
        self.cache: dict[int, dict] = {}
        self.current_cid = -1
        self.gidx_gpu = None
        self.raw_cpu = None
        self.obs_mask_gpu = None

    def load(self, cid: int):
        cid = int(cid)
        if cid == self.current_cid:
            return
        self.current_cid = cid
        if cid not in self.cache:
            entry = self.chunk_table[cid]
            self.cache[cid] = torch.load(entry["path"], map_location="cpu", weights_only=False)
        ch = self.cache[cid]
        gidx = np.asarray(ch["local_to_global_gene_idx"], dtype=np.int64)
        self.gidx_gpu = torch.from_numpy(gidx).long().to(self.device)
        self.raw_cpu = ch["target_genes_raw_local_csr"]
        obs = ch.get("observed_gene_mask_local", torch.ones(len(gidx), dtype=torch.bool))
        if torch.is_tensor(obs):
            self.obs_mask_gpu = obs.bool().to(self.device)
        else:
            self.obs_mask_gpu = torch.from_numpy(np.asarray(obs, dtype=bool)).bool().to(self.device)
        self.observed_cols_gpu = torch.where(self.obs_mask_gpu)[0]
        nnz_by_gene = np.asarray(self.raw_cpu.getnnz(axis=0), dtype=np.float32)
        denom = max(1, int(self.raw_cpu.shape[0]))
        self.prevalence_gpu = torch.from_numpy(nnz_by_gene / float(denom)).float().to(self.device)

    def get_row(self, grow: int):
        row = self.raw_cpu.getrow(int(grow))
        return row.indices.astype(np.int64, copy=False), row.data.astype(np.float32, copy=False)



def sampler_mode_for_step(global_step: int, sampler_mode_override: str | None = None):
    """Exact 70/20/10 deterministic cycle shared by all DDP ranks.

    70: section_HVG384 primary over nonzero genes, 20: all measured section genes
    including zero, 10: long-tail measured genes, not Program residual.
    """
    if sampler_mode_override and sampler_mode_override != "train_schedule":
        if sampler_mode_override == "long_tail_10":
            return "long_tail_10", 0.5
        return sampler_mode_override, 1.0
    m = int(global_step) % 10
    if m <= 6:
        return "high_priority_70", 1.0
    if m <= 8:
        return "all_gene_20", 1.0
    return "long_tail_10", 0.5


def load_section_hvg_map(path: str, device):
    if not path:
        raise RuntimeError("FAIL_MISSING_SECTION_HVG384: --section_hvg_path is required")
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"FAIL_MISSING_SECTION_HVG384: {path}")
    raw = torch.load(p, map_location="cpu", weights_only=False)
    out = {}
    for k, v in raw.items():
        # Latest section_HVG384 is keyed by sid string from current gene chunks.
        out[str(k)] = torch.as_tensor(v, dtype=torch.long, device=device)
    if not out:
        raise RuntimeError(f"FAIL_MISSING_SECTION_HVG384: empty mapping {path}")
    return out


def _shuffle_tensor(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if x.numel() <= 1:
        return x
    return x[torch.randperm(x.numel(), device=x.device, generator=generator)]


def select_target_local_cols_by_mode(nonzero_cols, measured_cols, gidx, prevalence, mode, mode_target_k,
                                     row_idx, global_step, seed, gene_has_program_gpu,
                                     sid_value=None, section_hvg_map=None,
                                     high_priority_source="section_HVG384"):
    """Select local gene columns for current M1_v2 sampler contract.

    70% uses nonzero ∩ section_HVG384, then nonzero Program backfill, then nonzero all-gene backfill.
    20% uses all measured section genes, including measured zero.
    10% uses long-tail measured genes: non-HVG genes ordered by low section prevalence.
    """
    metrics = {
        "sampler_source": "empty",
        "high_priority_source": "section_HVG384",
        "p256_used_as_high_priority_70": False,
        "p256_used_for_long_tail_or_backfill": True,
        "hvg384_target_count": 0,
        "program_backfill_count": 0,
        "all_observed_backfill_count": 0,
        "long_tail_target_count": 0,
        "residual_backfill_count": 0,
        "all_gene_target_count": 0,
        "long_tail_target_count": 0,
        "long_tail_backfill_count": 0,
        "zero_target_count": 0,
        "nonzero_target_count": 0,
    }
    if measured_cols.numel() == 0:
        return measured_cols[:0], metrics
    if section_hvg_map is None:
        raise RuntimeError("FAIL_MISSING_SECTION_HVG384: section_hvg_map is None")
    sid_key = str(sid_value)
    if sid_key not in section_hvg_map:
        raise RuntimeError(f"FAIL_MISSING_SECTION_HVG384: sid={sid_key}")
    hvg_ids = section_hvg_map[sid_key]
    g = torch.Generator(device=measured_cols.device)
    g.manual_seed(int(seed + global_step * 7919 + row_idx * 13 + 314159))
    nz_gids = gidx[nonzero_cols] if nonzero_cols.numel() else gidx[nonzero_cols]
    nz_hvg = torch.isin(nz_gids, hvg_ids) if nonzero_cols.numel() else torch.zeros(0, dtype=torch.bool, device=measured_cols.device)
    nz_prog = gene_has_program_gpu[nz_gids].bool() if nonzero_cols.numel() else torch.zeros(0, dtype=torch.bool, device=measured_cols.device)

    if mode == "high_priority_70":
        hvg_pool = _shuffle_tensor(nonzero_cols[nz_hvg], g)
        prog_pool = _shuffle_tensor(nonzero_cols[(~nz_hvg) & nz_prog], g)
        all_pool = _shuffle_tensor(nonzero_cols[(~nz_hvg) & (~nz_prog)], g)
        need = int(mode_target_k); parts=[]
        take=min(need, int(hvg_pool.numel())); parts.append(hvg_pool[:take]); metrics["hvg384_target_count"]=take; need-=take
        if need>0:
            take=min(need, int(prog_pool.numel())); parts.append(prog_pool[:take]); metrics["program_backfill_count"]=take; need-=take
        if need>0:
            take=min(need, int(all_pool.numel())); parts.append(all_pool[:take]); metrics["all_observed_backfill_count"]=take; need-=take
        pool=torch.cat(parts) if parts else measured_cols[:0]
        metrics["sampler_source"]="section_HVG384"
    elif mode == "all_gene_20":
        # True all-gene branch: use every measured gene in the section/chunk, including zeros.
        # Do not apply target_k/mode_target_k here; otherwise this becomes sampled_gene_20.
        pool=_shuffle_tensor(measured_cols, g)
        metrics["sampler_source"]="all_measured_genes_full"
        metrics["all_gene_target_count"]=int(pool.numel())
    elif mode == "long_tail_10":
        measured_gids = gidx[measured_cols]
        non_hvg_mask = ~torch.isin(measured_gids, hvg_ids)
        long_cols = measured_cols[non_hvg_mask]
        # Low prevalence first; deterministic tie by global gene id.
        if long_cols.numel() > 0:
            scores = prevalence[long_cols]
            gids_l = gidx[long_cols].float()
            order = torch.argsort(scores * 1000000.0 + gids_l / 1000000.0)
            long_cols = long_cols[order]
        need=int(mode_target_k)
        parts=[]
        take=min(need, int(long_cols.numel())); parts.append(long_cols[:take]); metrics["long_tail_target_count"]=take; need-=take
        if need>0:
            backfill=_shuffle_tensor(measured_cols[~torch.isin(measured_cols, long_cols[:take])], g)
            take=min(need, int(backfill.numel())); parts.append(backfill[:take]); metrics["long_tail_backfill_count"]=take; need-=take
        pool=torch.cat(parts) if parts else measured_cols[:0]
        metrics["sampler_source"]="long_tail_measured"
    else:
        pool=_shuffle_tensor(measured_cols, g)[:int(mode_target_k)]
        metrics["sampler_source"]="all_measured_genes"
        metrics["all_gene_target_count"]=int(pool.numel())
    if pool.numel() > 0:
        nz_hit = torch.isin(pool, nonzero_cols)
        metrics["nonzero_target_count"] = int(nz_hit.sum().item())
        metrics["zero_target_count"] = int(pool.numel() - nz_hit.sum().item())
    return pool, metrics


# ═══════════════════════════════════════════════════════════════
#  Section-Chunk Target Pool Cache (P0)
# ═══════════════════════════════════════════════════════════════
def verify_chunk_section_mapping(chunk_table: dict) -> bool:
    """Verify each chunk_id maps to exactly one section_id.
    Returns True if mapping is 1:1, False otherwise.
    """
    sid_per_chunk = {}
    for cid, entry in chunk_table.items():
        sid = entry.get("sid", "")
        if cid in sid_per_chunk and sid_per_chunk[cid] != sid:
            return False
        sid_per_chunk[cid] = sid
    return True


class SectionChunkTargetCache:
    """Per-batch cache for section-chunk level target pool data.

    Keyed by (section_id, chunk_id). Stores precomputed gene sets that are
    identical for all rows within the same chunk.
    """
    def __init__(self):
        self._cache: dict[tuple[str, int], dict] = {}
        self.rebuild_count = 0
        self.unique_chunk_count = 0
        self.rows_processed = 0

    def get_or_build(self, sid: str, cid: int, measured_cols: torch.Tensor,
                     gidx: torch.Tensor, prevalence: torch.Tensor,
                     gene_has_program_gpu: torch.Tensor,
                     section_hvg_map: dict, sampler_mode: str) -> dict:
        """Return cached target pool for (sid, cid), building if needed."""
        key = (sid, cid)
        if key in self._cache:
            self.rows_processed += 1
            return self._cache[key]

        self.rebuild_count += 1
        self.unique_chunk_count += 1
        self.rows_processed += 1

        measured_gids = gidx[measured_cols]
        hvg_ids = section_hvg_map.get(str(sid), torch.empty(0, dtype=torch.long, device=measured_cols.device))

        entry = {
            "measured_local_cols": measured_cols,
            "measured_global_gene_ids": measured_gids,
        }

        # HVG384 intersection (for high_priority_70)
        if hvg_ids.numel() > 0:
            hvg_mask = torch.isin(measured_gids, hvg_ids)
            entry["hvg384_local_cols"] = measured_cols[hvg_mask]
            entry["non_hvg_local_cols"] = measured_cols[~hvg_mask]
        else:
            entry["hvg384_local_cols"] = torch.empty(0, dtype=torch.long, device=measured_cols.device)
            entry["non_hvg_local_cols"] = measured_cols

        # Program member intersection (for high_priority_70 backfill)
        prog_mask = gene_has_program_gpu[measured_gids].bool()
        entry["program_member_local_cols"] = measured_cols[prog_mask]

        # Long tail: non-HVG measured genes sorted by prevalence ascending
        if hvg_ids.numel() > 0:
            non_hvg_mask = ~torch.isin(measured_gids, hvg_ids)
            long_cols = measured_cols[non_hvg_mask]
        else:
            long_cols = measured_cols
        if long_cols.numel() > 0:
            scores = prevalence[long_cols]
            gids_l = gidx[long_cols].float()
            order = torch.argsort(scores * 1000000.0 + gids_l / 1000000.0)
            entry["long_tail_local_cols"] = long_cols[order]
        else:
            entry["long_tail_local_cols"] = long_cols

        self._cache[key] = entry
        return entry


# ═══════════════════════════════════════════════════════════════
#  Batch builder
# ═══════════════════════════════════════════════════════════════
def build_batch(row_indices, target_k, prompt_k, hvg_ratio,
                features_gpu, rf_gpu, rgrow_gpu, rcid_gpu, rorg_gpu, rtech_gpu, rsid_gpu,
                nbr, chunk_loader: GPUChunkLoader, device,
                global_step=0, rank=0, stream_id=0, seed=42,
                prompt_start_step=5000, prompt_ratio=0.08, prompt_k_cap=64,
                gene_has_program_gpu=None, section_hvg_map=None, high_priority_source="program_member",
                sampler_mode_override=None):
    B = int(len(row_indices))
    if B == 0:
        return None

    # M2 overlays store only row/feature indices and side information. Features
    # are gathered from the same shared feature cache used by M0 and M1.
    m2_neighbors = gather_m2_neighbors(nbr, row_indices, features_gpu)

    T = max(1, target_k)
    sampler_mode, target_fraction = sampler_mode_for_step(global_step, sampler_mode_override=sampler_mode_override)
    # 70/10 are budgeted branches. all_gene_20 is full measured-gene supervision and must not be capped by target_k.
    mode_target_k = -1 if sampler_mode == "all_gene_20" else max(1, int(math.floor(T * target_fraction)))
    prompt_budget_base = T if mode_target_k < 0 else mode_target_k
    P = 0 if int(global_step) < int(prompt_start_step) else min(int(math.floor(prompt_budget_base * float(prompt_ratio))), int(prompt_k_cap))
    all_tg, all_tz, all_pg, all_pz = [], [], [], []
    t_lens, p_lens = [], []
    overlap_counts = []
    mode_target_counts = []
    sampler_sources = []
    hvg384_counts = []
    program_backfill_counts = []
    all_observed_backfill_counts = []
    long_tail_counts = []
    residual_backfill_counts = []
    all_gene_counts = []
    long_tail_counts = []
    long_tail_backfill_counts = []
    zero_target_counts = []
    nonzero_target_counts = []
    valid_pos = []

    # ── Chunk-grouped iteration with SectionChunkTargetCache (P0) ──
    pool_cache = SectionChunkTargetCache()
    cid_to_positions = defaultdict(list)
    row_cids = rcid_gpu[row_indices].cpu().numpy()
    for i in range(B):
        cid_to_positions[int(row_cids[i])].append(i)

    for cid, batch_positions in cid_to_positions.items():
        chunk_loader.load(cid)
        gidx = chunk_loader.gidx_gpu
        obs = chunk_loader.obs_mask_gpu
        measured_cols = chunk_loader.observed_cols_gpu
        if measured_cols.numel() == 0:
            continue

        sid_value = chunk_loader.chunk_table[cid].get("sid", "")
        cached = pool_cache.get_or_build(
            sid=sid_value, cid=cid, measured_cols=measured_cols,
            gidx=gidx, prevalence=chunk_loader.prevalence_gpu,
            gene_has_program_gpu=gene_has_program_gpu,
            section_hvg_map=section_hvg_map, sampler_mode=sampler_mode)

        for pos_in_batch in batch_positions:
            row_idx = int(row_indices[pos_in_batch].item())
            grow = int(rgrow_gpu[row_idx].item())

            # ── Per-row: CSR nonzero extraction ──
            cols_np, vals_np = chunk_loader.get_row(grow)
            cols_t = torch.as_tensor(cols_np, dtype=torch.long, device=device) if len(cols_np) else torch.empty(0, dtype=torch.long, device=device)
            vals_t = torch.as_tensor(vals_np, dtype=torch.float32, device=device) if len(vals_np) else torch.empty(0, dtype=torch.float32, device=device)
            if cols_t.numel() > 0:
                nz_valid = (gidx[cols_t] >= 0) & obs[cols_t]
                nonzero_cols = cols_t[nz_valid]
                nonzero_vals = vals_t[nz_valid]
            else:
                nonzero_cols = torch.empty(0, dtype=torch.long, device=device)
                nonzero_vals = torch.empty(0, dtype=torch.float32, device=device)

            # ── Target selection ──
            if sampler_mode == "all_gene_20":
                # NO per-row shuffle — chunk-level shared pool AND order
                t_cols = cached["measured_local_cols"]
                sampler_source = "all_measured_genes_full"
            elif sampler_mode == "long_tail_10":
                g = torch.Generator(device=device)
                g.manual_seed(int(seed + global_step * 7919 + rank * 104729 + row_idx * 13 + stream_id * 997 + 314159))
                long_cols = cached["long_tail_local_cols"]
                need = int(mode_target_k)
                parts = []
                take = min(need, int(long_cols.numel()))
                if take > 0:
                    parts.append(long_cols[:take])
                    need -= take
                if need > 0 and take > 0:
                    used = long_cols[:take]
                    remaining = cached["measured_local_cols"][~torch.isin(cached["measured_local_cols"], used)]
                    backfill = _shuffle_tensor(remaining, g)
                    take2 = min(need, int(backfill.numel()))
                    if take2 > 0:
                        parts.append(backfill[:take2])
                t_cols = torch.cat(parts) if parts else cached["measured_local_cols"][:0]
                sampler_source = "long_tail_measured"
            elif sampler_mode == "high_priority_70":
                g = torch.Generator(device=device)
                g.manual_seed(int(seed + global_step * 7919 + rank * 104729 + row_idx * 13 + stream_id * 997 + 314159))
                hvg_cols = cached["hvg384_local_cols"]
                non_hvg_cols = cached["non_hvg_local_cols"]
                prog_cols_all = cached["program_member_local_cols"]
                nz_set = nonzero_cols
                hvg_nz = hvg_cols[torch.isin(hvg_cols, nz_set)] if hvg_cols.numel() and nz_set.numel() else hvg_cols[:0]
                non_hvg_nz = non_hvg_cols[torch.isin(non_hvg_cols, nz_set)] if non_hvg_cols.numel() and nz_set.numel() else non_hvg_cols[:0]
                prog_nz = non_hvg_nz[torch.isin(non_hvg_nz, prog_cols_all)] if non_hvg_nz.numel() and prog_cols_all.numel() else non_hvg_nz[:0]
                rest_nz = non_hvg_nz[~torch.isin(non_hvg_nz, prog_cols_all)] if non_hvg_nz.numel() else non_hvg_nz[:0]
                hvg_pool = _shuffle_tensor(hvg_nz, g)
                prog_pool = _shuffle_tensor(prog_nz, g)
                all_pool = _shuffle_tensor(rest_nz, g)
                need = int(mode_target_k)
                parts = []
                take = min(need, int(hvg_pool.numel())); parts.append(hvg_pool[:take]); need -= take
                if need > 0:
                    take = min(need, int(prog_pool.numel())); parts.append(prog_pool[:take]); need -= take
                if need > 0:
                    take = min(need, int(all_pool.numel())); parts.append(all_pool[:take]); need -= take
                t_cols = torch.cat(parts) if parts else cached["measured_local_cols"][:0]
                sampler_source = "section_HVG384"
            else:
                g = torch.Generator(device=device)
                g.manual_seed(int(seed + global_step * 7919 + rank * 104729 + row_idx * 13 + stream_id * 997 + 314159))
                t_cols = _shuffle_tensor(cached["measured_local_cols"], g)[:int(mode_target_k)]
                sampler_source = "all_measured_genes"

            if t_cols.numel() == 0:
                continue

            # ── Per-row: value extraction ──
            eff_T = int(t_cols.numel())
            t_gg = gidx[t_cols]
            t_v = torch.zeros(eff_T, dtype=torch.float32, device=device)
            if nonzero_cols.numel() > 0:
                order = torch.argsort(nonzero_cols)
                sorted_cols = nonzero_cols[order]
                pos = torch.searchsorted(sorted_cols, t_cols)
                in_range = pos < sorted_cols.numel()
                hit = in_range & (sorted_cols[pos.clamp(max=max(0, sorted_cols.numel()-1))] == t_cols)
                if hit.any():
                    t_v[hit] = nonzero_vals[order[pos[hit]]]

            # ── Per-row: prompt selection ──
            prompt_pool = nonzero_cols[~torch.isin(nonzero_cols, t_cols)]
            if prompt_pool.numel() > 0:
                pperm = det_perm(prompt_pool.numel(), seed + global_step * 7919 + rank * 104729 + row_idx * 17 + stream_id * 997 + 42)
                prompt_pool = prompt_pool[pperm]
            eff_P = min(P, int(prompt_pool.numel())) if P > 0 else 0
            if eff_P > 0:
                p_sel = prompt_pool[:eff_P]
                p_gg = gidx[p_sel]
                order2 = torch.argsort(nonzero_cols)
                sorted_cols2 = nonzero_cols[order2]
                pos2 = torch.searchsorted(sorted_cols2, p_sel)
                p_v = nonzero_vals[order2[pos2]]
            else:
                p_gg = torch.empty(0, dtype=torch.long, device=device)
                p_v = torch.empty(0, dtype=torch.float32, device=device)

            pgf = torch.zeros(P, dtype=torch.long, device=device) if P > 0 else torch.empty(0, dtype=torch.long, device=device)
            pzf = torch.zeros(P, dtype=torch.float32, device=device) if P > 0 else torch.empty(0, dtype=torch.float32, device=device)
            if eff_P > 0:
                pgf[:eff_P] = p_gg
                pzf[:eff_P] = p_v

            all_tg.append(t_gg); all_tz.append(t_v)
            all_pg.append(pgf); all_pz.append(pzf)
            t_lens.append(eff_T); p_lens.append(eff_P)
            overlap_counts.append(int(torch.isin(p_gg, t_gg).sum().item()) if eff_P > 0 else 0)
            mode_target_counts.append(eff_T)
            sampler_sources.append(sampler_source)
            # simplified metric counts (no per-row select_target metrics)
            nz_hit = int(torch.isin(t_cols, nonzero_cols).sum().item()) if nonzero_cols.numel() else 0
            hvg384_counts.append(eff_T if sampler_mode == "high_priority_70" else 0)
            program_backfill_counts.append(0)
            all_observed_backfill_counts.append(0)
            long_tail_counts.append(eff_T if sampler_mode == "long_tail_10" else 0)
            residual_backfill_counts.append(0)
            all_gene_counts.append(eff_T if sampler_mode == "all_gene_20" else 0)
            long_tail_backfill_counts.append(0)
            zero_target_counts.append(int((t_v == 0).sum().item()))
            nonzero_target_counts.append(nz_hit)
            valid_pos.append(pos_in_batch)

    if not valid_pos:
        return None

    vp = torch.as_tensor(valid_pos, dtype=torch.long, device=device)
    Bv = len(valid_pos)

    feat_idx = rf_gpu[row_indices[vp]]
    center_img = features_gpu[feat_idx].float()

    # per-row masks. Target padding is dynamic because all_gene_20 supervises the full measured panel.
    tl = torch.tensor(t_lens, dtype=torch.long, device=device)
    pl = torch.tensor(p_lens, dtype=torch.long, device=device)
    T_pad = int(tl.max().item())
    target_gene_ids = torch.zeros(Bv, T_pad, dtype=torch.long, device=device)
    target_log1p = torch.zeros(Bv, T_pad, dtype=torch.float32, device=device)
    for j, (tg_row, tz_row) in enumerate(zip(all_tg, all_tz)):
        n = int(tg_row.numel())
        target_gene_ids[j, :n] = tg_row
        target_log1p[j, :n] = tz_row
    t_pos = torch.arange(T_pad, device=device).unsqueeze(0)
    target_mask = t_pos < tl.unsqueeze(1)
    if P > 0:
        p_pos = torch.arange(P, device=device).unsqueeze(0)
        prompt_mask = p_pos < pl.unsqueeze(1)
    else:
        prompt_mask = torch.zeros(Bv, 0, dtype=torch.bool, device=device)

    return {
        "center_img": center_img,
        "target_gene_ids": target_gene_ids,
        "target_log1p": target_log1p,
        "target_mask": target_mask,
        "prompt_gene_ids": torch.stack(all_pg) if P > 0 else torch.zeros(Bv, 0, dtype=torch.long, device=device),
        "prompt_vals": torch.stack(all_pz) if P > 0 else torch.zeros(Bv, 0, dtype=torch.float32, device=device),
        "prompt_mask": prompt_mask,
        "sampler_mode": sampler_mode,
        "sampler_source": ",".join(sorted(set(sampler_sources))),
        "mode_target_budget": (T_pad if mode_target_k < 0 else mode_target_k),
        "mode_target_budget_policy": ("full_measured_genes" if mode_target_k < 0 else "fixed_budget"),
        "mode_target_count": int(sum(mode_target_counts)),
        "high_priority_source": "section_HVG384",
        "p256_used_as_high_priority_70": False,
        "p256_used_for_long_tail_or_backfill": True,
        "hvg384_target_count": int(sum(hvg384_counts)),
        "program_backfill_count": int(sum(program_backfill_counts)),
        "all_observed_backfill_count": int(sum(all_observed_backfill_counts)),
        "long_tail_target_count": int(sum(long_tail_counts)),
        "residual_backfill_count": int(sum(residual_backfill_counts)),
        "all_gene_target_count": int(sum(all_gene_counts)),
        "long_tail_target_count": int(sum(long_tail_counts)),
        "long_tail_backfill_count": int(sum(long_tail_backfill_counts)),
        "zero_target_count": int(sum(zero_target_counts)),
        "nonzero_target_count": int(sum(nonzero_target_counts)),
        "hvg384_fraction": float(sum(hvg384_counts)) / max(1, int(sum(mode_target_counts))),
        "program_backfill_fraction": float(sum(program_backfill_counts)) / max(1, int(sum(mode_target_counts))),
        "all_measured_backfill_fraction": float(sum(all_observed_backfill_counts)) / max(1, int(sum(mode_target_counts))),
        "prompt_budget": P,
        "target_prompt_overlap_count": int(sum(overlap_counts)),
        **{key: value[vp] for key, value in m2_neighbors.items()},
        "organ_id": rorg_gpu[row_indices[vp]].long(),
        "tech_id": rtech_gpu[row_indices[vp]].long(),
        "n_valid": Bv,
        "n_requested": B,
        # ── Section-chunk cache audit (P0) ──
        "chunk_section_unique": True,
        "all_gene_pool_rebuild_count": pool_cache.rebuild_count if sampler_mode == "all_gene_20" else 0,
        "all_gene_unique_chunk_count": pool_cache.unique_chunk_count if sampler_mode == "all_gene_20" else 0,
        "all_gene_rows_processed": pool_cache.rows_processed if sampler_mode == "all_gene_20" else 0,
        "all_gene_pool_rebuild_per_row": False,
        "all_gene_shuffle_per_row": sampler_mode != "all_gene_20",
        "all_gene_target_pool_shared": sampler_mode == "all_gene_20",
        "all_gene_target_order_shared": sampler_mode == "all_gene_20",
        "long_tail_pool_rebuild_count": pool_cache.rebuild_count if sampler_mode == "long_tail_10" else 0,
        "long_tail_unique_chunk_count": pool_cache.unique_chunk_count if sampler_mode == "long_tail_10" else 0,
        "long_tail_pool_rebuild_per_row": False,
        "high_priority_base_pool_cached": sampler_mode == "high_priority_70",
        "row_value_extraction_per_row": True,
    }


# ═══════════════════════════════════════════════════════════════
#  Chunk-local batch iterator
# ═══════════════════════════════════════════════════════════════
def chunk_batch_iter(indices, batch_size, rcid_gpu, device,
                     training=True, seed=42, epoch=0, rank=0, stream_id=0,
                     drop_last=None):
    if drop_last is None:
        drop_last = training
    arr = indices.detach().cpu().numpy().astype(np.int64, copy=True)
    if training:
        arr = arr[det_perm(len(arr), seed + epoch * 1000003 + rank * 9176 + stream_id * 104729)]
    bc = defaultdict(list)
    rcids = rcid_gpu[torch.as_tensor(arr, device=device)].cpu().numpy()
    for i, c in zip(arr, rcids):
        bc[int(c)].append(int(i))
    cl = sorted(bc.keys())
    if training:
        p = det_perm(len(cl), seed + epoch * 1000003 + rank * 9176 + 17 + stream_id * 104729)
        cl = [cl[int(x)] for x in p]
    for cid in cl:
        rows = bc[cid]
        if training:
            p = det_perm(len(rows), seed + epoch * 1000003 + rank * 9176 + cid * 13 + stream_id * 104729)
            rows = [rows[int(x)] for x in p]
        for s in range(0, len(rows), batch_size):
            rr = rows[s:s + batch_size]
            if len(rr) == batch_size or (not drop_last and len(rr) > 0):
                yield cid, torch.tensor(rr, device=device, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
#  DDP helpers
# ═══════════════════════════════════════════════════════════════
def setup_ddp(args):
    ddp = args.world_size > 1
    if ddp:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.ddp_backend,
            timeout=timedelta(minutes=int(args.ddp_timeout_minutes)),
        )
    device = torch.device(f"cuda:{args.local_rank}" if ddp else "cuda:0")
    r0 = (not ddp) or args.rank == 0
    log(f"DDP enabled={ddp} rank={args.rank}/{args.world_size} device={device}")
    return ddp, device, r0


def amp_ctx(args):
    if args.amp_dtype == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if args.amp_dtype == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def ddp_sum(values, device, ddp):
    if not ddp:
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def parse_step_set(spec: str) -> set[int]:
    out: set[int] = set()
    if not spec:
        return out
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def ddp_min_int(val, ddp):
    if not ddp:
        return val
    t = torch.tensor([val], dtype=torch.long, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


# ═══════════════════════════════════════════════════════════════
#  Loss & metrics
# ═══════════════════════════════════════════════════════════════
def masked_smooth_l1_sum_count(pred, target, mask):
    p = pred.float()[mask]
    t = target.float()[mask]
    if p.numel() == 0:
        return pred.sum() * 0.0, torch.tensor(0.0, device=pred.device)
    return F.smooth_l1_loss(p, t, reduction="sum"), torch.tensor(float(p.numel()), device=pred.device)

def global_pair_mean_loss(pred, target, mask, world_size: int):
    local_sum, local_count = masked_smooth_l1_sum_count(pred, target, mask)
    global_count = local_count.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    # DDP averages gradients across ranks; multiply by world_size so the effective
    # gradient equals d(sum_rank loss_sum / global_pair_count).
    scaled_loss = local_sum * float(world_size) / global_count.clamp(min=1.0)
    return scaled_loss, local_sum.detach(), local_count.detach(), global_count.detach()


def metric_stats(pred, target, mask):
    x = pred.float()[mask]; y = target.float()[mask]
    if x.numel() == 0:
        return [0.0] * 9
    loss = F.smooth_l1_loss(x, y, reduction="sum").item()
    d = x - y
    return [loss, (d * d).sum().item(), d.abs().sum().item(),
            x.sum().item(), y.sum().item(), (x * x).sum().item(),
            (y * y).sum().item(), (x * y).sum().item(), float(x.numel())]


M1V3_AUX_METRIC_NAMES = [
    "gene_base_mean", "gene_base_std", "organ_gene_delta_mean", "organ_gene_delta_std",
    "image_residual_mean", "image_residual_std", "alpha_nbr_value", "alpha_prog_value",
    "gamma_c_mean", "gamma_c_std", "beta_c_mean", "beta_c_std",
    "gamma_n_mean", "gamma_n_std", "beta_n_mean", "beta_n_std",
    "center_mod_norm", "nbr_mod_norm", "ctx_norm", "alpha_nbr_ctx_norm",
    "alpha_nbr_ctx_to_center_ratio", "organ_ctx_norm", "tech_ctx_norm",
]


def _scalar_float(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "detach"):
        t = x.detach().float()
        return float(t.mean().item()) if t.numel() else default
    return default


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def per_gene_pcc_values(per_gene_stats):
    n_g, sx_g, sy_g, sx2_g, sy2_g, sxy_g = per_gene_stats
    den = torch.sqrt(
        torch.clamp(n_g * sx2_g - sx_g * sx_g, min=0)
        * torch.clamp(n_g * sy2_g - sy_g * sy_g, min=0)
    )
    pcc = (n_g * sxy_g - sx_g * sy_g) / den.clamp(min=1e-12)
    valid = (n_g >= 2) & torch.isfinite(pcc) & (den > 0)
    values = pcc[valid].detach().float().cpu()
    return values, int(valid.sum().item())


def per_spot_pcc_values(per_spot_stats):
    n_s, sx_s, sy_s, sx2_s, sy2_s, sxy_s = per_spot_stats
    den = torch.sqrt(
        torch.clamp(n_s * sx2_s - sx_s * sx_s, min=0)
        * torch.clamp(n_s * sy2_s - sy_s * sy_s, min=0)
    )
    pcc = (n_s * sxy_s - sx_s * sy_s) / den.clamp(min=1e-12)
    valid = (n_s >= 2) & torch.isfinite(pcc) & (den > 0)
    values = pcc[valid].detach().float().cpu()
    return values, int(valid.sum().item())


def compute_metrics(stats, per_gene_stats=None):
    ls, se, ae, sx, sy, sx2, sy2, sxy, n = stats
    if n <= 0:
        out = {"val_loss": float("nan"), "val_MSE": float("nan"), "val_PCC": float("nan"), "n": 0}
    else:
        dx = n * sx2 - sx * sx
        dy = n * sy2 - sy * sy
        den = math.sqrt(max(dx, 0) * max(dy, 0))
        pcc = (n * sxy - sx * sy) / den if den > 0 else 0.0
        out = {"val_loss": ls / n, "val_MSE": se / n, "val_PCC": pcc, "n": int(n)}
    if per_gene_stats is not None:
        vals, valid_genes = per_gene_pcc_values(per_gene_stats)
        out["per_gene_PCC_valid_genes"] = valid_genes
        out["per_gene_PCC_mean"] = float(vals.mean().item()) if vals.numel() else float("nan")
        out["per_gene_PCC_median"] = float(vals.median().item()) if vals.numel() else float("nan")
    return out


# ═══════════════════════════════════════════════════════════════
#  Eval
# ═══════════════════════════════════════════════════════════════
def run_eval(model, val_idx, args, feat, rf, rgrow, rcid, rorg, rtech,
             nbr, cl, device, ddp, step=0, return_group_metrics=False):
    model.eval()
    torch.cuda.empty_cache()
    ablation_mode = getattr(args, "eval_ablation_mode", "full")
    eval_ablation = {}
    eval_gene_id_shuffle = False
    if ablation_mode == "full":
        pass
    elif ablation_mode == "gene_id_shuffle":
        eval_gene_id_shuffle = True
    elif ablation_mode == "program_prior_off":
        eval_ablation["disable_program_prior"] = True
    else:
        eval_ablation[ablation_mode] = True
    # eval_micro_batch_rows_per_rank controls only evaluation micro-batching.
    # It must not define or change the eval sample universe. eval_max_rows and
    # debug_eval_rows are smoke/debug-only caps; they are forbidden in full mode.
    eval_bs = int(getattr(args, "eval_micro_batch_rows_per_rank", 128))
    if getattr(args, "eval_rows_per_rank", None) is not None:
        eval_bs = int(args.eval_rows_per_rank)
    eval_bs = max(1, eval_bs)
    eval_cap_raw = getattr(args, "eval_max_rows", None)
    debug_cap_raw = getattr(args, "debug_eval_rows", None)
    eval_cap = int(eval_cap_raw) if eval_cap_raw is not None else 0
    debug_cap = int(debug_cap_raw) if debug_cap_raw is not None else 0
    if debug_cap > 0:
        eval_cap = debug_cap
    if args.mode == "full" and eval_cap > 0:
        raise ValueError("full mode requires eval_max_rows/debug_eval_rows unset so eval_truncated=false")
    rows_full = val_idx[args.rank::args.world_size] if ddp else val_idx
    eval_rows_available_local = int(len(rows_full))
    rows = rows_full
    eval_truncated_local = False
    if eval_cap > 0 and len(rows) > eval_cap:
        p = det_perm(len(rows), args.seed + args.rank * 1009)
        rows = rows[p[:eval_cap]]
        eval_truncated_local = True
    eval_rows_local = int(len(rows))
    h = hashlib.sha256()
    h.update(rows.detach().cpu().numpy().astype(np.int64, copy=False).tobytes())
    eval_sample_universe_hash_local = h.hexdigest()
    eval_num_micro_batches_local = 0
    stats = [0.0] * 9
    n_rows = 0
    n_genes = getattr(args, "n_genes", 115055)
    per_gene = torch.zeros((6, int(n_genes)), dtype=torch.float64, device=device)
    spot_enabled = True
    per_spot = torch.zeros((6, len(val_idx)), dtype=torch.float64, device=device)
    row_to_spot = torch.full((int(rf.shape[0]),), -1, dtype=torch.long, device=device)
    row_to_spot[val_idx] = torch.arange(len(val_idx), dtype=torch.long, device=device)
    n_organs = int(getattr(args, "n_organs", 0))
    n_techs = int(getattr(args, "n_techs", 0))
    group_count = n_organs + n_techs
    group_stats = None
    per_gene_group = None
    if return_group_metrics:
        if n_organs <= 0 or n_techs <= 0:
            raise RuntimeError("group eval requires n_organs and n_techs")
        group_stats = torch.zeros((group_count, 9), dtype=torch.float64, device=device)
        per_gene_group = torch.zeros(
            (group_count, 6, int(n_genes)), dtype=torch.float64, device=device
        )
    aux_sums = {k: 0.0 for k in M1V3_AUX_METRIC_NAMES}
    aux_count = 0
    with torch.no_grad():
        for cid, rr in chunk_batch_iter(rows, eval_bs, rcid, device,
                                         training=False, seed=args.seed, stream_id=777,
                                         drop_last=False):
            b = build_batch(rr, args.target_k, args.prompt_k, args.hvg_ratio,
                            feat, rf, rgrow, rcid, rorg, rtech, args.rsid_gpu, nbr, cl, device,
                            global_step=step, rank=args.rank, stream_id=777, seed=args.seed,
                            prompt_start_step=args.prompt_start_step, prompt_ratio=args.prompt_ratio, prompt_k_cap=args.prompt_k_cap,
                            gene_has_program_gpu=args.gene_has_program_gpu, section_hvg_map=args.section_hvg_map, high_priority_source=args.high_priority_source,
                            sampler_mode_override=args.eval_sampler_mode)
            if b is None:
                continue
            eval_num_micro_batches_local += 1
            model_target_gene_ids = b["target_gene_ids"]
            model_center_img = b["center_img"]
            model_neighbor_inputs = {
                key: b[key]
                for key in (
                    "candidate_img", "candidate_mask", "candidate_dist", "candidate_cos",
                    "homo_img", "homo_mask", "homo_dist", "homo_cos",
                    "hetero_img", "hetero_mask", "hetero_dist", "hetero_cos",
                )
            }
            if ablation_mode == "image_shuffle" and model_center_img.shape[0] > 1:
                gen = torch.Generator(device=model_center_img.device)
                gen.manual_seed(int(getattr(args, "seed", 0)) + int(step) * 7919 + 606060)
                perm = torch.randperm(model_center_img.shape[0], device=model_center_img.device, generator=gen)
                model_center_img = model_center_img[perm]
            if ablation_mode == "neighbor_shuffle" and model_neighbor_inputs["candidate_img"].shape[0] > 1:
                gen = torch.Generator(device=model_neighbor_inputs["candidate_img"].device)
                gen.manual_seed(int(getattr(args, "seed", 0)) + int(step) * 7919 + 707070)
                perm = torch.randperm(model_neighbor_inputs["candidate_img"].shape[0], device=model_neighbor_inputs["candidate_img"].device, generator=gen)
                model_neighbor_inputs = {key: value[perm] for key, value in model_neighbor_inputs.items()}
            if eval_gene_id_shuffle:
                flat = model_target_gene_ids.reshape(-1)
                if flat.numel() > 1:
                    gen = torch.Generator(device=flat.device)
                    gen.manual_seed(int(getattr(args, "seed", 0)) + int(step) * 7919 + 909090)
                    perm = torch.randperm(flat.numel(), device=flat.device, generator=gen)
                    model_target_gene_ids = flat[perm].reshape_as(model_target_gene_ids)
            with amp_ctx(args):
                out = model(center_img=model_center_img, target_gene_ids=model_target_gene_ids,
                            **model_neighbor_inputs, organ_id=b["organ_id"], tech_id=b["tech_id"],
                            p_gids=b["prompt_gene_ids"], p_log1p=b["prompt_vals"], p_mask=b["prompt_mask"], global_step=step,
                            ablation=eval_ablation)
                pred = out["pred_log1p"]
            aux_count += 1
            for _k in M1V3_AUX_METRIC_NAMES:
                aux_sums[_k] += _scalar_float(out.get(_k), 0.0)
            mask = b["target_mask"].bool()
            s = metric_stats(pred, b["target_log1p"], mask)
            stats = [a + c for a, c in zip(stats, s)]
            gids = b["target_gene_ids"][mask].long()
            x = pred.float()[mask].double(); y = b["target_log1p"].float()[mask].double()
            per_gene[0].scatter_add_(0, gids, torch.ones_like(x))
            per_gene[1].scatter_add_(0, gids, x)
            per_gene[2].scatter_add_(0, gids, y)
            per_gene[3].scatter_add_(0, gids, x*x)
            per_gene[4].scatter_add_(0, gids, y*y)
            per_gene[5].scatter_add_(0, gids, x*y)
            if spot_enabled:
                spot_idx = row_to_spot[rr].unsqueeze(1).expand_as(mask)
                sel = mask & (spot_idx >= 0)
                spot_idx = spot_idx[sel]
                sx = pred.float()[sel].double()
                sy = b["target_log1p"].float()[sel].double()
                per_spot[0].scatter_add_(0, spot_idx, torch.ones_like(sx))
                per_spot[1].scatter_add_(0, spot_idx, sx)
                per_spot[2].scatter_add_(0, spot_idx, sy)
                per_spot[3].scatter_add_(0, spot_idx, sx*sx)
                per_spot[4].scatter_add_(0, spot_idx, sy*sy)
                per_spot[5].scatter_add_(0, spot_idx, sx*sy)
            if return_group_metrics:
                for group_offset, row_group in (
                    (0, b["organ_id"].long()),
                    (n_organs, b["tech_id"].long()),
                ):
                    for local_group in torch.unique(row_group).tolist():
                        selected = mask & (row_group == int(local_group)).unsqueeze(1)
                        if not selected.any():
                            continue
                        gx = pred.float()[selected].double()
                        gy = b["target_log1p"].float()[selected].double()
                        gd = gx - gy
                        group_id = group_offset + int(local_group)
                        group_stats[group_id] += torch.stack([
                            F.smooth_l1_loss(gx, gy, reduction="sum"),
                            (gd * gd).sum(), gd.abs().sum(), gx.sum(), gy.sum(),
                            (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum(),
                            torch.as_tensor(float(gx.numel()), dtype=torch.float64, device=device),
                        ])
                        ggids = b["target_gene_ids"][selected].long()
                        moments = (torch.ones_like(gx), gx, gy, gx * gx, gy * gy, gx * gy)
                        for stat_index, values in enumerate(moments):
                            per_gene_group[group_id, stat_index].scatter_add_(0, ggids, values)
            n_rows += b["n_valid"]
    reduced = ddp_sum(stats + [float(n_rows), float(eval_rows_available_local), float(eval_rows_local), float(eval_num_micro_batches_local), float(1 if eval_truncated_local else 0)], device, ddp)
    if ddp:
        dist.all_reduce(per_gene, op=dist.ReduceOp.SUM)
        if spot_enabled:
            dist.all_reduce(per_spot, op=dist.ReduceOp.SUM)
        if return_group_metrics:
            dist.all_reduce(group_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(per_gene_group, op=dist.ReduceOp.SUM)
        hash_gather = [None for _ in range(args.world_size)]
        dist.all_gather_object(hash_gather, eval_sample_universe_hash_local)
        eval_sample_universe_hash = hashlib.sha256("|".join(str(x) for x in hash_gather).encode()).hexdigest()
    else:
        eval_sample_universe_hash = eval_sample_universe_hash_local
    m = compute_metrics(reduced[:9], tuple(per_gene))
    if spot_enabled:
        spot_vals, n_spots = per_spot_pcc_values(per_spot)
        m["spot_PCC_mean"] = float(spot_vals.mean().item()) if n_spots else float("nan")
        m["spot_PCC_median"] = float(spot_vals.median().item()) if n_spots else float("nan")
        m["spot_PCC_valid_spots"] = int(n_spots)
    m["n_eval_rows"] = int(reduced[9])
    m["eval_total_rows_available"] = int(reduced[10])
    m["eval_rows_used"] = int(reduced[11])
    m["eval_micro_batch_rows_per_rank"] = int(eval_bs)
    m["eval_max_rows"] = int(eval_cap) if eval_cap > 0 else None
    m["eval_truncated"] = bool(int(reduced[13]) > 0)
    m["eval_num_micro_batches"] = int(reduced[12])
    m["formal_full_eval_valid"] = bool(
        m["eval_rows_used"] == m["eval_total_rows_available"]
        and not m["eval_truncated"]
    )
    m["eval_sample_universe_hash"] = eval_sample_universe_hash
    m["eval_global_pair_count"] = int(m.get("n", 0))
    m["global_step"] = step
    m["eval_sampler_mode"] = args.eval_sampler_mode
    m["high_priority_source"] = args.high_priority_source
    m["hvg384_path"] = args.section_hvg_path
    if aux_count > 0:
        for _k in M1V3_AUX_METRIC_NAMES:
            m[_k] = aux_sums[_k] / float(aux_count)
    m["eval_ablation_mode"] = ablation_mode
    m["eval_gene_id_shuffle"] = bool(eval_gene_id_shuffle)
    model.train()
    if not return_group_metrics:
        return m

    grouped = {"organ": {}, "technology": {}, "aggregate": {}}
    for kind, offset, count in (
        ("organ", 0, n_organs),
        ("technology", n_organs, n_techs),
    ):
        for index in range(count):
            grouped[kind][index] = compute_metrics(
                group_stats[offset + index].tolist(), tuple(per_gene_group[offset + index])
            )
    for name, start, stop in (
        ("old7", 0, min(7, n_organs)),
        ("added10", min(7, n_organs), n_organs),
    ):
        if start < stop:
            grouped["aggregate"][name] = compute_metrics(
                group_stats[start:stop].sum(dim=0).tolist(),
                tuple(per_gene_group[start:stop].sum(dim=0)),
            )
    return m, grouped


# ═══════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════
def count_batches(indices, bs, rcid, device):
    arr = indices.cpu().numpy()
    rcids = rcid[torch.as_tensor(arr, device=device)].cpu().numpy()
    bc = defaultdict(int)
    for c in rcids:
        bc[int(c)] += 1
    return sum(v // bs for v in bc.values())


def write_group_eval_rows(path: Path, step: int, grouped: dict, args, append: bool):
    fields = [
        "global_step", "group_type", "group_id", "group_name", "report_only",
        "val_loss", "val_MSE", "val_PCC", "per_gene_PCC_mean",
        "per_gene_PCC_median", "per_gene_PCC_valid_genes", "n",
    ]
    organ_names = getattr(args, "organ_id_to_name", {})
    tech_names = getattr(args, "tech_id_to_name", {})
    rows = []
    for kind in ("aggregate", "organ", "technology"):
        for group_id, metrics in grouped.get(kind, {}).items():
            if kind == "organ":
                name = organ_names.get(int(group_id), str(group_id))
            elif kind == "technology":
                name = tech_names.get(int(group_id), str(group_id))
            else:
                name = str(group_id)
            rows.append({
                "global_step": int(step), "group_type": kind,
                "group_id": group_id, "group_name": name, "report_only": True,
                **{key: metrics.get(key) for key in fields[5:]},
            })
    with open(path, "a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def train_loop(raw_model, args, feat, rf, rgrow, rcid, rorg, rtech,
               nbr, cl, train_idx, val_idx, device, ddp, r0, out_root):
    if ddp:
        # M1-v3 has delayed/conditional branches (prompt, gene bias, neighbor/program paths).
        # Some parameters can be legitimately unused for a given sampler mode/step, so DDP
        # must not require every parameter to receive gradients every iteration.
        model = DDP(raw_model, device_ids=[args.local_rank],
                     output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = raw_model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if len(opt.state) != 0:
        raise RuntimeError("new AdamW optimizer must have empty initial state")
    if r0 and not args.resume_full_state:
        with open(Path(out_root) / "optimizer_initial_state_audit.json", "w") as handle:
            json.dump({
                "status": "PASS_EMPTY_OPTIMIZER_STATE",
                "state_entries": 0,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "scheduler": "none",
            }, handle, indent=2, sort_keys=True)
    full_resume_payload = None
    full_resume_rank_state = None
    if args.resume_full_state:
        if not args.direct_17organ_model_only_resume:
            raise RuntimeError("--resume_full_state requires direct checkpoint initialization")
        full_resume_payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        required = {
            "format_version", "step", "model", "optimizer", "early_stopping",
            "rank_runtime_states", "training_progress", "last_eval_metrics",
        }
        missing = sorted(required - set(full_resume_payload))
        if missing:
            raise RuntimeError(f"full-state checkpoint missing fields: {missing}")
        if full_resume_payload.get("format_version") != "m2_17organ_full_state_v1":
            raise RuntimeError(f"unsupported full-state format: {full_resume_payload.get('format_version')}")
        if int(full_resume_payload["step"]) != int(args.resume_global_step):
            raise RuntimeError("full-state step does not match --resume_global_step")
        rank_states = full_resume_payload["rank_runtime_states"]
        if len(rank_states) != int(args.world_size):
            raise RuntimeError(
                f"checkpoint world_size={len(rank_states)} but runtime world_size={args.world_size}"
            )
        by_rank = {int(item["rank"]): item for item in rank_states}
        if set(by_rank) != set(range(int(args.world_size))):
            raise RuntimeError(f"invalid rank runtime states: {sorted(by_rank)}")
        full_resume_rank_state = by_rank[int(args.rank)]
        opt.load_state_dict(full_resume_payload["optimizer"])
        for group in opt.param_groups:
            if not math.isclose(float(group["lr"]), float(args.lr), rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(f"restored optimizer LR {group['lr']} != contract LR {args.lr}")
            if not math.isclose(float(group["weight_decay"]), float(args.weight_decay), rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(
                    f"restored optimizer weight_decay {group['weight_decay']} != {args.weight_decay}"
                )
        log(
            f"Full state loaded: step={full_resume_payload['step']} "
            f"optimizer_states={len(opt.state)} rank_cursor="
            f"{full_resume_rank_state['epoch']}:{full_resume_rank_state['batches_consumed_in_epoch']}"
        )

    # section-rank exclusive split
    src = train_idx
    if ddp and args.world_size > 1:
        arr = train_idx.cpu().numpy()
        rcids = rcid[torch.as_tensor(arr, device=device)].cpu().numpy()
        c2s = {cid: e["sid"] for cid, e in cl.chunk_table.items()}
        sids = [c2s.get(int(c), f"__{c}") for c in rcids]
        uniq = sorted(set(sids))
        owned = set(uniq[args.rank::args.world_size])
        keep = [int(r) for r, s in zip(arr, sids) if s in owned]
        src = torch.tensor(keep, dtype=torch.long, device=device)
        log(f"DDP section split: rank {args.rank} owns {len(owned)} sections, {len(keep)} rows")

    bpe = count_batches(src, args.batch_size, rcid, device)
    spe = ddp_min_int(bpe, ddp)
    log(f"Steps/epoch: {spe} (local batches: {bpe})")

    max_steps = args.max_steps
    if max_steps == 0:
        if args.mode == "smoke20":
            max_steps = 20
        elif args.mode == "short_train":
            max_steps = min(2000, spe * 2)
        elif args.max_epochs > 0:
            max_steps = args.max_epochs * spe
        else:
            max_steps = min(50000, spe * 50)

    eval_int = args.eval_interval if args.eval_interval > 0 else max(1, spe)
    final_global_step = int(getattr(args, "resume_global_step", 0) or 0) + int(max_steps)
    log(
        f"Training: optimizer_steps={max_steps} final_global_step={final_global_step} "
        f"eval_interval={eval_int} bs={args.batch_size}"
    )

    # A strict full-state resume must not consume runtime RNG or repeat a completed eval.
    baseline_step = int(getattr(args, "resume_global_step", 0) or 0)
    if full_resume_payload is not None:
        m0 = dict(full_resume_payload["last_eval_metrics"])
        groups0 = full_resume_payload.get("last_group_metrics", {})
    else:
        m0, groups0 = run_eval(
            model, val_idx, args, feat, rf, rgrow, rcid, rorg, rtech,
            nbr, cl, device, ddp, step=baseline_step, return_group_metrics=True,
        )
    if r0:
        prefix = "Restored last eval" if full_resume_payload is not None else "Baseline"
        log(f"{prefix}: loss={m0['val_loss']:.6f} MSE={m0['val_MSE']:.6f} PCC={m0['val_PCC']:.4f} n={m0['n']}")
        eval_path = Path(out_root) / "eval_metrics.tsv"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", newline="") as f:
            w = csv.DictWriter(f, delimiter="\t", fieldnames=list(m0.keys()))
            w.writeheader()
            w.writerow(m0)
        group_eval_path = Path(out_root) / "eval_metrics_by_group.tsv"
        write_group_eval_rows(group_eval_path, baseline_step, groups0, args, append=False)
        step_profile_path = Path(out_root) / "train_step_profile.tsv"
        step_profile_fields = [
            "step", "epoch", "loss_mean", "valid_rows", "requested_rows",
            "global_pair_count", "prompt_budget", "prompt_actual_count",
            "target_prompt_overlap_count", "sampler_mode", "sampler_source", "mode_target_budget", "mode_target_count",
            "high_priority_source", "hvg384_target_count", "program_backfill_count", "all_observed_backfill_count",
            "long_tail_target_count", "residual_backfill_count", "all_gene_target_count", "long_tail_target_count", "long_tail_backfill_count", "zero_target_count", "nonzero_target_count", "hvg384_fraction", "program_backfill_fraction",
            "all_measured_backfill_fraction", "p256_used_as_high_priority_70", "image_encoder_mode", "fusion_mode",
            "gate_fusion_enabled", "decoder_input_dim", "neighbor_q_mode", "program_runtime_count", "membership_padding",
            "program_emb_dim", "center_t_norm", "gene_ctx_norm", "meta_t_norm", "program_128_norm", "p_ctx_norm",
            "organ_ctx_norm", "tech_ctx_norm", "organ_zero_active", "tech_zero_active",
            "grad_norm_before_clip", "grad_finite",
            "organ_emb_grad_norm", "tech_emb_grad_norm", "organ_tech_proj_grad_norm",
            "nbr_ctx_norm", "decoder_input_norm", "gene_base_mean", "gene_base_std", "organ_gene_delta_mean", "organ_gene_delta_std",
            "image_residual_mean", "image_residual_std", "alpha_nbr_value", "alpha_prog_value",
            "gamma_c_mean", "gamma_c_std", "beta_c_mean", "beta_c_std", "gamma_n_mean", "gamma_n_std", "beta_n_mean", "beta_n_std",
            "center_mod_norm", "nbr_mod_norm", "ctx_norm", "alpha_nbr_ctx_norm", "alpha_nbr_ctx_to_center_ratio",
            "neighbor_attention_entropy_mean", "neighbor_attention_max_mean",
            "neighbor_valid_count_mean", "target_width", "target_chunk_size",
            "chunk_section_unique", "all_gene_pool_rebuild_count", "all_gene_unique_chunk_count",
            "all_gene_rows_processed", "all_gene_pool_rebuild_per_row", "all_gene_shuffle_per_row",
            "all_gene_target_pool_shared", "all_gene_target_order_shared",
            "long_tail_pool_rebuild_count", "long_tail_unique_chunk_count", "long_tail_pool_rebuild_per_row",
            "high_priority_base_pool_cached", "row_value_extraction_per_row",
        ]
        with open(step_profile_path, "w", newline="") as f:
            w = csv.DictWriter(f, delimiter="	", fieldnames=step_profile_fields)
            w.writeheader()

    restored_es = full_resume_payload.get("early_stopping", {}) if full_resume_payload is not None else {}
    best_metric = float(restored_es.get("best_metric", getattr(args, "prior_best_metric", -float("inf"))))
    best_step = int(restored_es.get("best_step", getattr(args, "prior_best_step", 0)))
    bad_evals = int(restored_es.get("bad_evals", getattr(args, "prior_bad_evals", 0)))
    should_stop = False
    es_path = Path(out_root) / "early_stop_summary.json"
    save_at_steps = parse_step_set(args.save_at_steps)
    saved_checkpoint_steps: set[int] = set()
    last_eval_metrics = dict(m0)
    last_group_metrics = groups0
    last_eval_step = int(
        full_resume_payload.get("training_progress", {}).get("last_eval_step", 0)
        if full_resume_payload is not None else 0
    )

    def save_model_checkpoint(step: int, metrics: dict | None, filename: str | None = None, force: bool = False):
        if not r0:
            return
        if not (force or args.save_checkpoint):
            return
        if step in saved_checkpoint_steps and filename is None:
            return
        rm = raw_model if not ddp else model.module
        if filename is None:
            filename = f"checkpoint_step{step}.pt"
        payload = {
            "step": int(step),
            "epoch": int(epoch),
            "model": rm.state_dict(),
            "metrics": metrics or {},
            "args": vars(args),
        }
        torch.save(payload, Path(out_root) / filename)
        saved_checkpoint_steps.add(int(step))
        log(f"Checkpoint: {Path(out_root) / filename}")

    def save_full_state_checkpoint(step: int, metrics: dict | None, phase: str):
        if not args.save_full_state:
            return
        local_runtime = _capture_rank_runtime_state(
            args.rank, device, epoch, batches_consumed_in_epoch
        )
        if ddp:
            gathered = [None for _ in range(args.world_size)]
            dist.all_gather_object(gathered, local_runtime)
        else:
            gathered = [local_runtime]
        if r0:
            rm = raw_model if not ddp else model.module
            payload = {
                "format_version": "m2_17organ_full_state_v1",
                "step": int(step),
                "epoch": int(epoch),
                "phase": str(phase),
                "model": rm.state_dict(),
                "optimizer": opt.state_dict(),
                "early_stopping": {
                    "best_metric": float(best_metric),
                    "best_step": int(best_step),
                    "bad_evals": int(bad_evals),
                    "should_stop": bool(should_stop),
                },
                "rank_runtime_states": gathered,
                "training_progress": {
                    "global_step": int(step),
                    "last_eval_step": int(last_eval_step),
                    "run_loss": float(run_loss),
                    "local_step_in_segment": int(local_step),
                    "exposure_rows": exposure_rows.detach().cpu(),
                    "exposure_batches": exposure_batches.detach().cpu(),
                },
                "last_eval_metrics": dict(metrics or {}),
                "last_group_metrics": last_group_metrics,
                "args": vars(args),
                "resume_contract": {
                    "optimizer_restored_on_next_resume": True,
                    "rank_rng_restored_on_next_resume": True,
                    "iterator_cursor_restored_on_next_resume": True,
                    "required_world_size": int(args.world_size),
                    "scratch_lineage_from_step0": bool(args.scratch17),
                    "source_lineage_has_optimizer_reset_at_step40000": False,
                },
            }
            target = Path(out_root) / args.full_state_filename
            _atomic_torch_save(payload, target)
            with open(Path(out_root) / "full_state_checkpoint_status.json", "w") as handle:
                json.dump({
                    "status": "PASS_FULL_STATE_CHECKPOINT_WRITTEN",
                    "path": str(target),
                    "step": int(step),
                    "phase": str(phase),
                    "format_version": payload["format_version"],
                    "optimizer_state_entries": len(opt.state),
                    "rank_runtime_state_count": len(gathered),
                }, handle, indent=2, sort_keys=True)
            log(f"Full-state checkpoint ({phase}): {target}")
        if ddp:
            dist.barrier()

    def write_es_state(step: int, metric: float, stopped: bool, reason: str = ""):
        if not r0:
            return
        state = {
            "local_step": int(step),
            "global_step": int(step),
            "metric": "per_gene_PCC_mean",
            "mode": "max",
            "best_metric": float(best_metric),
            "best_step": int(best_step),
            "bad_evals": int(bad_evals),
            "patience_evals": int(args.early_stop_patience_evals),
            "min_delta": float(args.early_stop_min_delta),
            "early_stop_start_step": int(args.early_stop_start_step),
            "should_stop": bool(stopped),
            "last_metric": float(metric) if math.isfinite(float(metric)) else None,
            "reason": reason,
        }
        with open(es_path, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def maybe_update_best(step: int, metrics: dict):
        nonlocal best_metric, best_step, bad_evals, should_stop
        metric = float(metrics.get("per_gene_PCC_mean", float("nan")))
        patience_active = int(step) >= int(args.early_stop_start_step)
        if not math.isfinite(metric):
            if patience_active:
                bad_evals += 1
            else:
                bad_evals = 0
            write_es_state(
                step, metric, should_stop,
                "nonfinite_metric" if patience_active else "optimizer_reset_adaptation_window",
            )
            return
        improved = metric > best_metric + float(args.early_stop_min_delta)
        if improved:
            best_metric = metric
            best_step = int(step)
            bad_evals = 0
            if r0 and (args.write_best_model or args.save_checkpoint or args.early_stopping):
                rm = raw_model if not ddp else model.module
                torch.save({"step": int(step), "model": rm.state_dict(), "metrics": metrics, "args": vars(args)},
                           Path(out_root) / "best_per_gene_pcc_checkpoint.pt")
        else:
            if patience_active:
                bad_evals += 1
            else:
                bad_evals = 0
        if args.early_stopping and patience_active and bad_evals >= int(args.early_stop_patience_evals):
            should_stop = True
        if should_stop:
            reason = "early_stop_patience"
        elif not patience_active:
            reason = "optimizer_reset_adaptation_window"
        else:
            reason = ""
        write_es_state(step, metric, should_stop, reason)

    epoch = int(
        full_resume_rank_state["epoch"]
        if full_resume_rank_state is not None else getattr(args, "resume_epoch", 0)
    )
    batches_consumed_in_epoch = int(
        full_resume_rank_state["batches_consumed_in_epoch"]
        if full_resume_rank_state is not None else 0
    )
    it = chunk_batch_iter(src, args.batch_size, rcid, device, training=True,
                          seed=args.base_seed + args.rank, epoch=epoch, rank=args.rank, stream_id=1)
    if batches_consumed_in_epoch:
        for skipped in range(batches_consumed_in_epoch):
            try:
                next(it)
            except StopIteration as exc:
                raise RuntimeError(
                    f"rank {args.rank} iterator cursor {batches_consumed_in_epoch} "
                    f"exceeds epoch {epoch} batch count at skip {skipped}"
                ) from exc
    local_step = 0
    global_step = int(getattr(args, "resume_global_step", 0) or 0)
    t0 = time.time()
    restored_progress = full_resume_payload.get("training_progress", {}) if full_resume_payload is not None else {}
    run_loss = float(restored_progress.get("run_loss", 0.0))
    exposure_rows = restored_progress.get(
        "exposure_rows", torch.zeros(int(args.n_organs), dtype=torch.long)
    ).to(device=device, dtype=torch.long)
    exposure_batches = restored_progress.get(
        "exposure_batches", torch.zeros(int(args.n_organs), dtype=torch.long)
    ).to(device=device, dtype=torch.long)
    exposure_path = Path(out_root) / "sampling_exposure_by_organ.tsv"
    first_forward_audited = False
    if r0:
        with open(exposure_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=[
                "global_step", "organ_id", "organ_name", "sampled_rows",
                "sampled_rank_batches", "row_fraction", "effective_repeats",
                "report_only_no_balancing",
            ])
            writer.writeheader()

    if full_resume_rank_state is not None:
        _restore_rank_runtime_rng(full_resume_rank_state, device)
        if r0:
            with open(Path(out_root) / "full_state_resume_audit.json", "w") as handle:
                json.dump({
                    "status": "PASS_EXACT_FULL_STATE_RESTORE",
                    "checkpoint": str(args.resume_checkpoint),
                    "global_step": int(global_step),
                    "optimizer_state_entries": len(opt.state),
                    "world_size": int(args.world_size),
                    "rank_rng_restored": True,
                    "iterator_cursor_restored": True,
                    "early_stop_state_restored": True,
                }, handle, indent=2, sort_keys=True)
    if args.save_full_state:
        phase = "restored_before_first_forward" if full_resume_payload is not None else (
            "initial_scratch_state" if args.scratch17 else "initial_optimizer_reset"
        )
        save_full_state_checkpoint(global_step, last_eval_metrics, phase)
    model.train()
    while local_step < max_steps:
        try:
            cid, rr = next(it)
            batches_consumed_in_epoch += 1
        except StopIteration:
            epoch += 1
            batches_consumed_in_epoch = 0
            it = chunk_batch_iter(src, args.batch_size, rcid, device, training=True,
                                  seed=args.base_seed + args.rank, epoch=epoch, rank=args.rank, stream_id=1)
            try:
                cid, rr = next(it)
                batches_consumed_in_epoch += 1
            except StopIteration:
                log(f"Data exhausted at epoch {epoch}")
                break

        cl.load(int(cid))
        b = build_batch(rr, args.target_k, args.prompt_k, args.hvg_ratio,
                        feat, rf, rgrow, rcid, rorg, rtech, args.rsid_gpu, nbr, cl, device,
                        global_step=global_step, rank=args.rank, stream_id=1, seed=args.base_seed,
                        prompt_start_step=args.prompt_start_step, prompt_ratio=args.prompt_ratio, prompt_k_cap=args.prompt_k_cap,
                        gene_has_program_gpu=args.gene_has_program_gpu, section_hvg_map=args.section_hvg_map, high_priority_source=args.high_priority_source)
        if b is None:
            continue

        batch_organ_counts = torch.bincount(
            b["organ_id"].long(), minlength=int(args.n_organs)
        ).to(dtype=torch.long)
        batch_organ_present = (batch_organ_counts > 0).to(dtype=torch.long)
        if ddp:
            dist.all_reduce(batch_organ_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_organ_present, op=dist.ReduceOp.SUM)
        exposure_rows += batch_organ_counts
        exposure_batches += batch_organ_present

        model.train()
        opt.zero_grad(set_to_none=True)
        target_width = int(b["target_gene_ids"].shape[1])
        chunk_size = max(1, int(getattr(args, "target_chunk_size", 2048)))
        local_total_count = b["target_mask"].sum().detach().to(dtype=torch.float32)
        global_pair_count = local_total_count.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_pair_count, op=dist.ReduceOp.SUM)
        global_pair_count = global_pair_count.clamp(min=1.0)
        # ── Sync target_width across DDP ranks to ensure same chunk count (prevents deadlock) ──
        if ddp:
            tw_t = torch.tensor([target_width], dtype=torch.long, device=device)
            dist.all_reduce(tw_t, op=dist.ReduceOp.MAX)
            target_width = int(tw_t.item())
        local_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_pair_count = local_total_count.detach()
        out = None
        any_chunk = False
        for t0_chunk in range(0, target_width, chunk_size):
            t1_chunk = min(target_width, t0_chunk + chunk_size)
            mask_chunk = b["target_mask"][:, t0_chunk:t1_chunk]
            any_chunk = True
            chunk_ids = b["target_gene_ids"][:, t0_chunk:t1_chunk]
            # ── Build gene_prior_cache per target chunk (not batch-wide) ──
            gene_prior_cache = None
            if target_width > chunk_size:
                gids_chunk = chunk_ids[mask_chunk].unique() if mask_chunk.any() else chunk_ids.unique()
                if gids_chunk.numel() > 0:
                    rm = raw_model if ddp else model.module if hasattr(model, 'module') else model
                    with amp_ctx(args):
                        gene_prior_cache = rm.build_gene_prior_cache(gids_chunk)
            with amp_ctx(args):
                out = model(center_img=b["center_img"], target_gene_ids=chunk_ids,
                            candidate_img=b["candidate_img"], candidate_mask=b["candidate_mask"],
                            candidate_dist=b["candidate_dist"], candidate_cos=b["candidate_cos"],
                            homo_img=b["homo_img"], homo_mask=b["homo_mask"],
                            homo_dist=b["homo_dist"], homo_cos=b["homo_cos"],
                            hetero_img=b["hetero_img"], hetero_mask=b["hetero_mask"],
                            hetero_dist=b["hetero_dist"], hetero_cos=b["hetero_cos"],
                            organ_id=b["organ_id"], tech_id=b["tech_id"],
                            p_gids=b["prompt_gene_ids"], p_log1p=b["prompt_vals"], p_mask=b["prompt_mask"],
                            global_step=global_step, gene_prior_cache=gene_prior_cache)
                chunk_sum, chunk_count = masked_smooth_l1_sum_count(out["pred_log1p"], b["target_log1p"][:, t0_chunk:t1_chunk], mask_chunk)
                loss = chunk_sum * float(args.world_size) / global_pair_count
            if not torch.isfinite(loss):
                log(f"FATAL non-finite loss step={global_step} target_chunk={t0_chunk}:{t1_chunk}")
                if ddp:
                    dist.barrier()
                os._exit(3)
            loss.backward()
            del gene_prior_cache  # free autograd graph after backward
            local_loss_sum = local_loss_sum + chunk_sum.detach()
        if not any_chunk or out is None:
            log(f"FATAL empty target chunks step={global_step}")
            if ddp:
                dist.barrier()
            os._exit(3)

        prompt_schedule_active = int(global_step) >= int(args.prompt_start_step)
        prompt_actual = int(b["prompt_mask"].sum().detach().item()) if b["prompt_mask"].numel() else 0
        expected_prompt_runtime = bool(prompt_schedule_active and prompt_actual > 0)
        if bool(out.get("gene_bias_enabled_runtime", False)) != bool(prompt_schedule_active):
            raise RuntimeError(
                f"gene_bias schedule mismatch at global_step={global_step}: "
                f"expected={prompt_schedule_active} actual={out.get('gene_bias_enabled_runtime')}"
            )
        if bool(out.get("prompt_enabled_runtime", False)) != expected_prompt_runtime:
            raise RuntimeError(
                f"prompt schedule mismatch at global_step={global_step}: "
                f"expected={expected_prompt_runtime} actual={out.get('prompt_enabled_runtime')} "
                f"prompt_actual={prompt_actual}"
            )
        if not first_forward_audited:
            if int(b["target_prompt_overlap_count"]) != 0:
                raise RuntimeError("target/prompt overlap must be zero")
            if r0:
                with open(Path(out_root) / "first_forward_contract.json", "w") as handle:
                    json.dump({
                        "status": "PASS", "global_step": int(global_step),
                        "prompt_start_step": int(args.prompt_start_step),
                        "prompt_schedule_active": prompt_schedule_active,
                        "prompt_actual_count": prompt_actual,
                        "prompt_enabled_runtime": bool(out.get("prompt_enabled_runtime", False)),
                        "gene_bias_enabled_runtime": bool(out.get("gene_bias_enabled_runtime", False)),
                        "target_prompt_overlap_count": 0,
                        "sampler_mode": b["sampler_mode"],
                        "expected_sampler_mode": sampler_mode_for_step(global_step),
                        "all_gene_batches_may_have_zero_prompt_due_to_strict_disjointness": True,
                    }, handle, indent=2, sort_keys=True)
            first_forward_audited = True

        profile_model = model.module if hasattr(model, "module") else model
        organ_emb_grad_norm = _module_grad_norm(profile_model.organ_emb)
        tech_emb_grad_norm = _module_grad_norm(profile_model.tech_emb)
        organ_tech_proj_grad_norm = _module_grad_norm(profile_model.organ_tech_proj)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(torch.as_tensor(total_grad_norm)).item():
            raise RuntimeError(f"non-finite gradient norm at global_step={global_step}")
        grad_norm_before_clip = float(torch.as_tensor(total_grad_norm).detach().float().item())
        opt.step()
        if ddp:
            torch.cuda.synchronize()

        local_step += 1
        global_step += 1
        run_loss += float((local_loss_sum / local_pair_count.clamp(min=1.0)).detach().item())

        audit_interval = max(1, int(args.sampling_audit_interval))
        if global_step % audit_interval == 0 or local_step >= max_steps:
            if r0:
                total_rows = int(exposure_rows.sum().item())
                train_counts = getattr(args, "train_rows_by_organ", {})
                with open(exposure_path, "a", newline="") as handle:
                    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=[
                        "global_step", "organ_id", "organ_name", "sampled_rows",
                        "sampled_rank_batches", "row_fraction", "effective_repeats",
                        "report_only_no_balancing",
                    ])
                    for organ_id in range(int(args.n_organs)):
                        sampled = int(exposure_rows[organ_id].item())
                        denominator = int(train_counts.get(organ_id, 0))
                        writer.writerow({
                            "global_step": int(global_step), "organ_id": organ_id,
                            "organ_name": args.organ_id_to_name.get(organ_id, str(organ_id)),
                            "sampled_rows": sampled,
                            "sampled_rank_batches": int(exposure_batches[organ_id].item()),
                            "row_fraction": sampled / max(1, total_rows),
                            "effective_repeats": sampled / max(1, denominator),
                            "report_only_no_balancing": True,
                        })

        checkpoint_due = False
        if args.save_checkpoint:
            if global_step in save_at_steps:
                checkpoint_due = True
                if global_step % eval_int != 0:
                    if global_step == int(args.prompt_start_step):
                        save_model_checkpoint(global_step, last_eval_metrics, f"checkpoint_step{global_step}_before_prompt_or_at_prompt_transition.pt")
                    else:
                        save_model_checkpoint(global_step, last_eval_metrics)
            elif args.checkpoint_interval > 0 and global_step % int(args.checkpoint_interval) == 0:
                checkpoint_due = True
                if global_step % eval_int != 0:
                    save_model_checkpoint(global_step, last_eval_metrics)
        if args.save_full_state and (
            checkpoint_due
            or (args.checkpoint_interval > 0 and global_step % int(args.checkpoint_interval) == 0)
        ):
            save_full_state_checkpoint(global_step, last_eval_metrics, "pre_eval")

        if r0 and global_step % args.log_interval == 0:
            avg = run_loss / args.log_interval
            log(f"step={global_step}/{final_global_step} local_step={local_step}/{max_steps} ep={epoch} loss={avg:.6f} "
                f"lr={args.lr:.2e} {time.time() - t0:.1f}s valid={b['n_valid']}/{b['n_requested']} "
                f"prompt_budget={b['prompt_budget']} prompt_actual={prompt_actual} overlap={b['target_prompt_overlap_count']} sampler_mode={b['sampler_mode']} source={b['sampler_source']} high_priority_source={b['high_priority_source']} hvg384={b['hvg384_target_count']} prog_backfill={b['program_backfill_count']} residual={b['long_tail_target_count']} decoder_dim={out.get('decoder_input_dim')} fusion={out.get('fusion_mode')} neighbor_q={out.get('neighbor_q_mode')}")
            with open(step_profile_path, "a", newline="") as f:
                csv.DictWriter(f, delimiter="	", fieldnames=step_profile_fields).writerow({
                    "step": global_step,
                    "epoch": epoch,
                    "loss_mean": avg,
                    "valid_rows": b["n_valid"],
                    "requested_rows": b["n_requested"],
                    "global_pair_count": float(global_pair_count.detach().item()),
                    "prompt_budget": int(b["prompt_budget"]),
                    "prompt_actual_count": prompt_actual,
                    "target_prompt_overlap_count": int(b["target_prompt_overlap_count"]),
                    "sampler_mode": b["sampler_mode"],
                    "sampler_source": b["sampler_source"],
                    "mode_target_budget": int(b["mode_target_budget"]),
                    "mode_target_count": int(b["mode_target_count"]),
                    "high_priority_source": b["high_priority_source"],
                    "hvg384_target_count": int(b["hvg384_target_count"]),
                    "program_backfill_count": int(b["program_backfill_count"]),
                    "all_observed_backfill_count": int(b["all_observed_backfill_count"]),
                    "long_tail_target_count": int(b["long_tail_target_count"]),
                    "residual_backfill_count": int(b["residual_backfill_count"]),
                    "all_gene_target_count": int(b["all_gene_target_count"]),
                    "long_tail_target_count": int(b["long_tail_target_count"]),
                    "long_tail_backfill_count": int(b["long_tail_backfill_count"]),
                    "zero_target_count": int(b["zero_target_count"]),
                    "nonzero_target_count": int(b["nonzero_target_count"]),
                    "hvg384_fraction": float(b["hvg384_fraction"]),
                    "program_backfill_fraction": float(b["program_backfill_fraction"]),
                    "all_measured_backfill_fraction": float(b["all_measured_backfill_fraction"]),
                    "p256_used_as_high_priority_70": bool(b["p256_used_as_high_priority_70"]),
                    "image_encoder_mode": out.get("image_encoder_mode"),
                    "fusion_mode": out.get("fusion_mode"),
                    "gate_fusion_enabled": bool(out.get("gate_fusion_enabled")),
                    "decoder_input_dim": int(out.get("decoder_input_dim")),
                    "neighbor_q_mode": out.get("neighbor_q_mode"),
                    "program_runtime_count": int(out.get("program_runtime_count")),
                    "membership_padding": int(out.get("membership_padding")),
                    "program_emb_dim": int(out.get("program_emb_dim")),
                    "center_t_norm": float(out.get("center_t_norm").detach().float().item()),
                    "gene_ctx_norm": float(out.get("gene_ctx_norm").detach().float().item()),
                    "meta_t_norm": float(out.get("meta_t_norm").detach().float().item()),
                    "program_128_norm": float(out.get("program_128_norm").detach().float().item()),
                    "organ_ctx_norm": _scalar_float(out.get("organ_ctx_norm")),
                    "tech_ctx_norm": _scalar_float(out.get("tech_ctx_norm")),
                    "organ_zero_active": bool(out.get("organ_zero_active", False)),
                    "tech_zero_active": bool(out.get("tech_zero_active", False)),
                    "grad_norm_before_clip": grad_norm_before_clip,
                    "grad_finite": True,
                    "organ_emb_grad_norm": organ_emb_grad_norm,
                    "tech_emb_grad_norm": tech_emb_grad_norm,
                    "organ_tech_proj_grad_norm": organ_tech_proj_grad_norm,
                    "p_ctx_norm": float(out.get("p_ctx_norm").detach().float().item()),
                    "nbr_ctx_norm": float(out.get("nbr_ctx_norm").detach().float().item()),
                    "decoder_input_norm": float(out.get("decoder_input_norm").detach().float().item()),
                    "gene_base_mean": _scalar_float(out.get("gene_base_mean")),
                    "gene_base_std": _scalar_float(out.get("gene_base_std")),
                    "organ_gene_delta_mean": _scalar_float(out.get("organ_gene_delta_mean")),
                    "organ_gene_delta_std": _scalar_float(out.get("organ_gene_delta_std")),
                    "image_residual_mean": _scalar_float(out.get("image_residual_mean")),
                    "image_residual_std": _scalar_float(out.get("image_residual_std")),
                    "alpha_nbr_value": _scalar_float(out.get("alpha_nbr_value")),
                    "alpha_prog_value": _scalar_float(out.get("alpha_prog_value")),
                    "gamma_c_mean": _scalar_float(out.get("gamma_c_mean")),
                    "gamma_c_std": _scalar_float(out.get("gamma_c_std")),
                    "beta_c_mean": _scalar_float(out.get("beta_c_mean")),
                    "beta_c_std": _scalar_float(out.get("beta_c_std")),
                    "gamma_n_mean": _scalar_float(out.get("gamma_n_mean")),
                    "gamma_n_std": _scalar_float(out.get("gamma_n_std")),
                    "beta_n_mean": _scalar_float(out.get("beta_n_mean")),
                    "beta_n_std": _scalar_float(out.get("beta_n_std")),
                    "center_mod_norm": _scalar_float(out.get("center_mod_norm")),
                    "nbr_mod_norm": _scalar_float(out.get("nbr_mod_norm")),
                    "ctx_norm": _scalar_float(out.get("ctx_norm")),
                    "alpha_nbr_ctx_norm": _scalar_float(out.get("alpha_nbr_ctx_norm")),
                    "alpha_nbr_ctx_to_center_ratio": _scalar_float(out.get("alpha_nbr_ctx_to_center_ratio")),
                    "neighbor_attention_entropy_mean": float(out.get("neighbor_attention_entropy").detach().float().mean().item()) if out.get("neighbor_attention_entropy") is not None else 0.0,
                    "neighbor_attention_max_mean": float(out.get("neighbor_attention_max").detach().float().mean().item()) if out.get("neighbor_attention_max") is not None else 0.0,
                    "neighbor_valid_count_mean": float(out.get("neighbor_valid_count").detach().float().mean().item()) if out.get("neighbor_valid_count") is not None else 0.0,
                    "target_width": int(b["target_gene_ids"].shape[1]),
                    "target_chunk_size": int(getattr(args, "target_chunk_size", 2048)),
                    # ── Section-chunk cache audit (P0) ──
                    "chunk_section_unique": bool(getattr(args, "chunk_section_unique", True)),
                    "all_gene_pool_rebuild_count": int(b.get("all_gene_pool_rebuild_count", 0)),
                    "all_gene_unique_chunk_count": int(b.get("all_gene_unique_chunk_count", 0)),
                    "all_gene_rows_processed": int(b.get("all_gene_rows_processed", 0)),
                    "all_gene_pool_rebuild_per_row": bool(b.get("all_gene_pool_rebuild_per_row", False)),
                    "all_gene_shuffle_per_row": bool(b.get("all_gene_shuffle_per_row", False)),
                    "all_gene_target_pool_shared": bool(b.get("all_gene_target_pool_shared", False)),
                    "all_gene_target_order_shared": bool(b.get("all_gene_target_order_shared", False)),
                    "long_tail_pool_rebuild_count": int(b.get("long_tail_pool_rebuild_count", 0)),
                    "long_tail_unique_chunk_count": int(b.get("long_tail_unique_chunk_count", 0)),
                    "long_tail_pool_rebuild_per_row": bool(b.get("long_tail_pool_rebuild_per_row", False)),
                    "high_priority_base_pool_cached": bool(b.get("high_priority_base_pool_cached", False)),
                    "row_value_extraction_per_row": bool(b.get("row_value_extraction_per_row", True)),
                })
            run_loss = 0.0

        if global_step % eval_int == 0 or local_step >= max_steps:
            m, grouped = run_eval(
                model, val_idx, args, feat, rf, rgrow, rcid, rorg, rtech,
                nbr, cl, device, ddp, step=global_step, return_group_metrics=True,
            )
            last_eval_metrics = dict(m)
            last_group_metrics = grouped
            last_eval_step = int(global_step)
            if r0:
                log(f"Eval step={global_step} loss={m['val_loss']:.6f} MSE={m['val_MSE']:.6f} "
                    f"PCC={m['val_PCC']:.4f} per_gene={m.get('per_gene_PCC_mean', float('nan')):.4f} n={m['n']}")
                with open(eval_path, "a", newline="") as f:
                    csv.DictWriter(f, delimiter="\t", fieldnames=list(m.keys())).writerow(m)
                write_group_eval_rows(group_eval_path, global_step, grouped, args, append=True)
            maybe_update_best(global_step, m)
            if args.save_checkpoint and (
                global_step in save_at_steps
                or (args.checkpoint_interval > 0 and global_step % int(args.checkpoint_interval) == 0)
            ):
                save_model_checkpoint(global_step, m, f"checkpoint_step{global_step}.pt", force=True)
            save_model_checkpoint(global_step, m, "last_checkpoint.pt", force=(args.save_checkpoint or args.early_stopping or args.write_best_model))
            save_full_state_checkpoint(global_step, m, "post_eval")
            if ddp:
                stop_t = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device=device)
                dist.broadcast(stop_t, src=0)
                should_stop = bool(int(stop_t.item()))
            if should_stop:
                log(f"Early stopping triggered at step={global_step} best_step={best_step} best_metric={best_metric:.6f}")
                break

    # final eval: reuse the most recent eval if it was just computed at this same step.
    if int(last_eval_step) == int(global_step):
        mf = dict(last_eval_metrics)
        if r0:
            log("Final: reused eval at step=%s loss=%.6f MSE=%.6f PCC=%.4f per_gene=%.4f" % (global_step, mf["val_loss"], mf["val_MSE"], mf["val_PCC"], mf.get("per_gene_PCC_mean", float("nan"))))
    else:
        mf, grouped = run_eval(
            model, val_idx, args, feat, rf, rgrow, rcid, rorg, rtech,
            nbr, cl, device, ddp, step=global_step, return_group_metrics=True,
        )
        last_eval_metrics = dict(mf)
        last_group_metrics = grouped
        last_eval_step = int(global_step)
        if r0:
            write_group_eval_rows(group_eval_path, global_step, grouped, args, append=True)
            log("Final: loss=%.6f MSE=%.6f PCC=%.4f per_gene=%.4f" % (mf["val_loss"], mf["val_MSE"], mf["val_PCC"], mf.get("per_gene_PCC_mean", float("nan"))))

    if args.save_checkpoint and r0:
        rm = raw_model if not ddp else model.module
        ckpt = Path(out_root) / f"checkpoint_step{global_step}.pt"
        payload = {"step": global_step, "epoch": epoch, "model": rm.state_dict(), "metrics": mf, "args": vars(args)}
        torch.save(payload, ckpt)
        torch.save(payload, Path(out_root) / "last_checkpoint.pt")
        log(f"Checkpoint: {ckpt}")
    elif r0 and (args.early_stopping or args.write_best_model):
        rm = raw_model if not ddp else model.module
        payload = {"step": global_step, "epoch": epoch, "model": rm.state_dict(), "metrics": mf, "args": vars(args)}
        torch.save(payload, Path(out_root) / "last_checkpoint.pt")
        log("Checkpoint: " + str(Path(out_root) / "last_checkpoint.pt"))
    if args.save_full_state:
        save_full_state_checkpoint(global_step, mf, "final")
    return mf


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.rank = int(os.environ.get("RANK", args.rank))
    args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if args.resume_full_state and not args.resume_checkpoint:
        raise RuntimeError("--resume_full_state requires --resume_checkpoint")
    if args.resume_full_state and not args.direct_17organ_model_only_resume:
        raise RuntimeError("--resume_full_state requires --direct_17organ_model_only_resume")
    if args.resume_full_state and not args.save_full_state:
        raise RuntimeError("full-state resume must keep --save_full_state enabled")
    if args.scratch17 and (args.resume_checkpoint or args.resume_full_state or args.unified_17organ_init):
        raise RuntimeError("--scratch17 forbids checkpoint/full-state/unified initialization inputs")
    if args.scratch17 and int(args.resume_global_step) != 0:
        raise RuntimeError("--scratch17 requires --resume_global_step 0")

    assert args.target_scale == "raw_counts_log1p", f"Expected raw_counts_log1p, got {args.target_scale}"
    if args.early_stopping:
        args.write_best_model = True
    if not args.output_root:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_root = f"{args.rowmap_root}/../m1_v2_smoke_{ts}"
    out_path = Path(args.output_root)
    if out_path.exists() and any(out_path.iterdir()) and not args.allow_existing_output:
        print(f"FATAL output_root exists and is non-empty: {out_path}. Use a new timestamped output_root or --allow_existing_output.", file=sys.stderr)
        sys.exit(2)

    ddp, device, r0 = setup_ddp(args)

    if r0:
        Path(args.output_root).mkdir(parents=True, exist_ok=True)
        log(f"Output: {args.output_root}")

    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)

    # ── data ──
    cb = load_codebooks(args.rowmap_root)
    rm = load_rowmap_tensors(args.rowmap_root)
    n_rows = rm["row_to_feature_index"].shape[0]
    validate_context_alignment(args, rm, cb, device, ddp, r0)

    tp_path = Path(args.rowmap_root).parent / "step2_gene_rawlog1p" / "target_policy.json"
    n_genes = 115055
    if tp_path.exists():
        with open(tp_path) as f:
            n_genes = int(json.load(f).get("n_global_genes", n_genes))
    log(f"n_genes={n_genes}")

    args.n_genes = n_genes
    prog_ids, prog_mask, n_prog = load_program_ontology(args.program_overlay_path, args.program_ids_path, args.program_mask_path, n_genes, device)
    args.gene_has_program_gpu = prog_mask.any(dim=1).bool()
    args.section_hvg_map = load_section_hvg_map(args.section_hvg_path, device)
    log(f"section_HVG384: {len(args.section_hvg_map)} sections from {args.section_hvg_path}")
    feat = load_feature_store(args.feature_store, device, half=True)
    ct = load_gene_chunk_table(args.gene_chunk_manifest)
    chunk_section_unique = verify_chunk_section_mapping(ct)
    log(f"chunk_section_unique = {chunk_section_unique}")
    if not chunk_section_unique:
        log("WARNING: chunk_id does NOT map 1:1 to section_id. Cache key must use (sid, cid).")
    args.chunk_section_unique = chunk_section_unique
    nbr = load_m2_neighbor_overlay(args.neighbor_overlay, n_rows, device)

    rf = rm["row_to_feature_index"].long().to(device)
    rcid = rm["row_to_gene_chunk_id"].long().to(device)
    rgrow = rm["row_to_gene_row_index"].long().to(device)
    rorg = rm["row_to_organ_id"].long().to(device)
    rtech = rm["row_to_tech_id"].long().to(device)
    rsid = rm.get("row_to_sid", torch.zeros(n_rows, dtype=torch.long)).long().to(device)
    args.rsid_gpu = rsid

    train_idx, val_idx, test_idx = load_split_indices(args.rowmap_root, args.v3_split_root, device)
    cl = GPUChunkLoader(ct, device)

    args.n_organs = int(cb["n_organs"])
    args.n_techs = int(cb["n_techs"])
    args.organ_id_to_name = {int(value): str(key) for key, value in cb["raw"]["organ_id"].items()}
    args.tech_id_to_name = {int(value): str(key) for key, value in cb["raw"]["tech_id"].items()}
    train_organ_counts = torch.bincount(
        rorg[train_idx].long(), minlength=args.n_organs
    ).detach().cpu().tolist()
    args.train_rows_by_organ = {index: int(value) for index, value in enumerate(train_organ_counts)}

    # ── model ──
    model_variant = "M2"
    if args.scratch17:
        model, initialization_audit = build_scratch_17organ_model(
            args, cb, n_genes, prog_ids, prog_mask, n_prog, device, r0
        )
    elif args.direct_17organ_model_only_resume:
        # Recovery lineage: model tensors are loaded exactly, while optimizer and
        # iterator state are intentionally reset because the source checkpoint
        # contains model state only. Any positive checkpoint step is permitted.
        if int(args.resume_global_step) <= 0 or args.unified_17organ_init:
            raise RuntimeError("direct model-only recovery requires step>0 and forbids unified initializer")
        model, initialization_audit = build_direct_17organ_model_only_resume(
            args, cb, n_genes, prog_ids, prog_mask, n_prog, device, r0
        )
    else:
        if int(args.resume_global_step) != 68000:
            raise RuntimeError(
                f"canonical unified retraining must start at global step 68000, got {args.resume_global_step}"
            )
        model, initialization_audit = build_unified_17organ_model(
            args, cb, n_genes, prog_ids, prog_mask, n_prog, device, r0
        )
    model.prompt_start_step = int(args.prompt_start_step)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {model.model_name} {n_params / 1e6:.1f}M params gene_bias={args.gene_bias}")

    local_hash = initialization_audit.get("model_state_sha256", _state_sha256(model.state_dict()))
    hashes = [local_hash]
    if ddp:
        hashes = [None for _ in range(args.world_size)]
        dist.all_gather_object(hashes, local_hash)
    if len(set(hashes)) != 1:
        raise RuntimeError(f"initial model state differs across ranks: {hashes}")
    if r0:
        with open(Path(args.output_root) / "initial_state_hash_by_rank.tsv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["rank", "model_state_sha256"])
            writer.writeheader()
            writer.writerows({"rank": rank, "model_state_sha256": value} for rank, value in enumerate(hashes))

    runtime_rng_fingerprint = _tensor_sha256(torch.get_rng_state())
    if device.type == "cuda":
        runtime_rng_fingerprint = hashlib.sha256(
            (runtime_rng_fingerprint + _tensor_sha256(torch.cuda.get_rng_state(device))).encode()
        ).hexdigest()
    rng_rows = [{
        "rank": int(args.rank),
        "runtime_seed": int(args.seed + args.rank),
        "sampler_seed_expression": f"{args.base_seed}+rank-aware iterator/sampler offsets",
        "runtime_rng_fingerprint": runtime_rng_fingerprint,
    }]
    if ddp:
        gathered_rng = [None for _ in range(args.world_size)]
        dist.all_gather_object(gathered_rng, rng_rows[0])
        rng_rows = gathered_rng
    if args.scratch17 and len({row["runtime_rng_fingerprint"] for row in rng_rows}) != int(args.world_size):
        raise RuntimeError("rank-aware runtime RNG fingerprints are not unique")
    if r0:
        with open(Path(args.output_root) / "rng_domain_audit.tsv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rng_rows[0]))
            writer.writeheader()
            writer.writerows(rng_rows)

    if args.scratch17:
        log(
            f"Scratch M2-17 initialized from seed={args.model_init_seed}; "
            f"resume_global_step=0 runtime_seed={args.seed}+rank checkpoint_read=false"
        )
    elif args.direct_17organ_model_only_resume and args.resume_full_state:
        log(
            f"Direct 17-organ full-state resume requested: checkpoint_step={args.resume_global_step}; "
            "model loads before train_loop and AdamW/RNG/iterator/early-stop restore inside train_loop"
        )
    elif args.direct_17organ_model_only_resume:
        log(
            f"Direct 17-organ model-only resume complete: checkpoint_step={args.resume_global_step} "
            f"prior_best={args.prior_best_metric}@{args.prior_best_step} prior_bad_evals={args.prior_bad_evals} "
            "optimizer_restored=false data_iterator_position_restored=false"
        )
    else:
        log(
            f"Canonical M1 partial-load complete: initialized_m2_keys="
            f"{initialization_audit['partial_load']['initialized_m2_key_count']} "
            f"resume_global_step={int(args.resume_global_step)} "
            "optimizer_restored=false early_stop_state_restored=false"
        )

    if args.mode == "eval":
        eval_step = int(getattr(args, "resume_global_step", 0) or 0)
        m = run_eval(model, val_idx, args, feat, rf, rgrow, rcid, rorg, rtech,
                     nbr, cl, device, ddp=ddp, step=eval_step)
        eval_path = Path(args.output_root) / "eval_metrics.tsv"
        if r0:
            with open(eval_path, "w", newline="") as f:
                w = csv.DictWriter(f, delimiter="	", fieldnames=list(m.keys()))
                w.writeheader()
                w.writerow(m)
            log(f"Eval step={eval_step} loss={m['val_loss']:.6f} MSE={m['val_MSE']:.6f} "
                f"PCC={m['val_PCC']:.4f} per_gene={m.get('per_gene_PCC_mean', float('nan')):.4f} n={m['n']}")
            log(f"Eval metrics written: {eval_path}")
        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    train_loop(model, args, feat, rf, rgrow, rcid, rorg, rtech,
               nbr, cl, train_idx, val_idx, device, ddp, r0, args.output_root)
    if ddp:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()

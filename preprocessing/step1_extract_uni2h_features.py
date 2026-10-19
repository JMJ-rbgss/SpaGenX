#!/usr/bin/env python3
"""First-version-compatible UNI2-h feature extraction for new_pipeline_800gb.

Authority: large_pretrain_v1 UNI2-h worker from 2026-06-01.
This script keeps first-version image/spot/white/OOM/AMP behavior while accepting
new_pipeline inventory TSVs. It writes per-sample shards; merge step creates the
master feature store and manifest.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # first-version contract: allow large histology images

try:
    import openslide
except Exception:
    openslide = None
try:
    import anndata as ad
except Exception:
    ad = None
try:
    import timm
except Exception:
    timm = None

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
X_CAND = ["x", "X", "xaxis", "pxl_col_in_fullres", "imagecol", "pixel_x", "coord_x", "array_col", "col"]
Y_CAND = ["y", "Y", "yaxis", "pxl_row_in_fullres", "imagerow", "pixel_y", "coord_y", "array_row", "row"]
ID_CAND = ["spot_id", "barcode", "barcodes", "spot", "cell_id", "Unnamed: 0", "index"]


def s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def pick_col(cols, candidates):
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def parse_first_path(x):
    parts = [p for p in s(x).split("|") if p]
    return parts[0] if parts else ""


def load_coords_csv(path):
    df = pd.read_csv(path)
    x = pick_col(df.columns, X_CAND)
    y = pick_col(df.columns, Y_CAND)
    sid = pick_col(df.columns, ID_CAND)
    if x is None or y is None:
        raise ValueError(f"coord CSV missing x/y columns: {path}")
    spot_ids = df[sid].astype(str).tolist() if sid else [str(i) for i in range(len(df))]
    return spot_ids, df[[x, y]].astype(float).to_numpy(), f"coord_csv:{Path(path).name}:{x},{y}"


def load_coords_h5ad(path):
    if ad is None:
        raise RuntimeError("anndata is required for h5ad coordinates")
    a = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" in a.obsm:
            coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
            return [str(x) for x in a.obs_names], coords[:, :2], "h5ad.obsm[spatial]"
        cols = list(a.obs.columns)
        x = pick_col(cols, X_CAND)
        y = pick_col(cols, Y_CAND)
        if x is None or y is None:
            raise ValueError(f"h5ad missing spatial/obs coordinates: {path}")
        coords = np.asarray(a.obs[[x, y]], dtype=np.float32)
        return [str(x) for x in a.obs_names], coords[:, :2], f"h5ad.obs:{x},{y}"
    finally:
        try:
            a.file.close()
        except Exception:
            pass


def white_ratio(arr, thresh):
    gray = arr.astype(np.float32).mean(axis=2)
    return float((gray >= float(thresh)).mean())


def crop_with_padding(img, x0, y0, x1, y1, pad_value=255):
    h, w = img.shape[:2]
    out = np.full((y1 - y0, x1 - x0, 3), pad_value, dtype=np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[(sy0 - y0):(sy1 - y0), (sx0 - x0):(sx1 - x0)] = img[sy0:sy1, sx0:sx1]
    return out, (sx0 != x0 or sy0 != y0 or sx1 != x1 or sy1 != y1)


def get_mpp(slide, default=0.5):
    for k in ("openslide.mpp-x", "aperio.MPP"):
        try:
            v = slide.properties.get(k)
            if v:
                return float(v)
        except Exception:
            pass
    return float(default)


def load_uni2h(checkpoint, device, feature_dim):
    if timm is None:
        raise RuntimeError("timm is required for UNI2-h")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    pe = state.get("patch_embed.proj.weight")
    if pe is None:
        raise RuntimeError("UNI2-h checkpoint missing patch_embed.proj.weight")
    embed_dim = int(pe.shape[0])
    patch = int(pe.shape[-1])
    depth = len({k.split(".")[1] for k in state if k.startswith("blocks.") and k.split(".")[1].isdigit()})
    qkv = state.get("blocks.0.attn.qkv.weight")
    num_heads = max(1, int(qkv.shape[0] // 3 // 64)) if qkv is not None else next((h for h in [48,32,24,16,12,8] if embed_dim % h == 0), 16)
    fc1 = state.get("blocks.0.mlp.fc1.weight")
    mlp_ratio = float(fc1.shape[0]) / float(embed_dim) if torch.is_tensor(fc1) else 4.0
    rt = state.get("reg_token")
    reg_tokens = int(rt.shape[1]) if torch.is_tensor(rt) and rt.ndim == 3 else 0
    kwargs = dict(img_size=224, patch_size=patch, in_chans=3, num_classes=0, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, init_values=1e-5, class_token=True, no_embed_class=True, dynamic_img_size=True, mlp_layer=timm.layers.mlp.SwiGLUPacked)
    try:
        model = timm.models.vision_transformer.VisionTransformer(**kwargs, reg_tokens=reg_tokens)
    except TypeError:
        model = timm.models.vision_transformer.VisionTransformer(**kwargs)
    ret = model.load_state_dict(state, strict=False)
    missing = list(getattr(ret, "missing_keys", []))
    unexpected = list(getattr(ret, "unexpected_keys", []))
    coverage = (len(model.state_dict()) - len(missing)) / max(1, len(model.state_dict()))
    if coverage < 0.97:
        raise RuntimeError(f"UNI2-h load coverage too low {coverage:.4f}; missing={missing[:8]} unexpected={unexpected[:8]}")
    if len(unexpected) > 16:
        raise RuntimeError(f"UNI2-h too many unexpected keys {len(unexpected)}")
    model.to(device).eval()
    with torch.inference_mode():
        y = model(torch.zeros(1, 3, 224, 224, device=device))
    if y.ndim != 2 or int(y.shape[1]) != int(feature_dim):
        raise RuntimeError(f"feature dim mismatch {tuple(y.shape)} expected {feature_dim}")
    print(f"UNI2-h loaded coverage={coverage:.4f} output={tuple(y.shape)} mlp_ratio={mlp_ratio:.6f}", flush=True)
    return model


def infer_batch(model, patches, device, amp=True):
    x = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).to(device=device, dtype=torch.float32) / 255.0
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.inference_mode():
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model(x)
        else:
            y = model(x)
    return y.detach().cpu().float()


def process_sample(row, model, device, args, out_dir):
    sample = s(row.get("sid")) or s(row.get("sample_id"))
    if not sample:
        raise ValueError("inventory row missing sid/sample_id")
    out_path = out_dir / f"{sample}.pt"
    if out_path.exists() and not args.overwrite:
        obj = torch.load(out_path, map_location="cpu", weights_only=False)
        n = int(obj.get("features", torch.empty(0)).shape[0])
        return {"sid": sample, "sample_id": sample, "shard_path": str(out_path), "status": "SKIP_EXISTS", "n_features": n}
    h5ad_path = parse_first_path(row.get("h5ad_path", ""))
    coord_path = parse_first_path(row.get("coord_path", "")) or parse_first_path(row.get("metadata_path_candidates", ""))
    if h5ad_path:
        spot_ids, coords, coord_src = load_coords_h5ad(h5ad_path)
    else:
        spot_ids, coords, coord_src = load_coords_csv(coord_path)
    if args.max_spots_per_sample > 0:
        spot_ids = spot_ids[:args.max_spots_per_sample]
        coords = coords[:args.max_spots_per_sample]
    img_path = parse_first_path(row.get("wsi_path", "")) or parse_first_path(row.get("image_path", "")) or parse_first_path(row.get("image_path_candidates", ""))
    if not img_path:
        raise ValueError(f"{sample}: missing image/wsi path")
    route = s(row.get("route")) or s(row.get("image_route"))
    lower = img_path.lower()
    is_wsi = route == "route_wsi_openslide_pyramid" or lower.endswith((".svs", ".tif", ".tiff", ".ndpi", ".mrxs"))
    patches, valid_meta, qc_patches = [], [], []
    n_fail = n_white = n_padded = 0
    read_backend = "openslide" if is_wsi else "PIL_load_once"
    read_size = args.patch_size
    best_level = ""
    mpp_used = ""
    target_mpp = args.target_mpp if is_wsi else ""
    mpp_policy = "target_mpp_pyramid" if is_wsi else "native_pixel_224"
    physical_scale_confidence = "from_audit_or_openslide" if is_wsi else "low"
    image_w = image_h = ""
    if is_wsi:
        if openslide is None:
            raise RuntimeError("openslide unavailable for WSI route")
        slide = openslide.OpenSlide(img_path)
        image_w, image_h = slide.dimensions
        mpp_used = get_mpp(slide, default=args.target_mpp)
        target_ds = float(args.target_mpp) / max(float(mpp_used), 1e-6)
        best_level = slide.get_best_level_for_downsample(target_ds)
        level_ds = float(slide.level_downsamples[best_level])
        read_size = max(1, int(round(args.patch_size * target_ds / level_ds)))
        offset_base = read_size * level_ds / 2.0
        for i, (cx, cy) in enumerate(coords):
            try:
                x0 = int(round(float(cx) - offset_base)); y0 = int(round(float(cy) - offset_base))
                patch = slide.read_region((x0, y0), best_level, (read_size, read_size)).convert("RGB")
                if read_size != args.patch_size:
                    patch = patch.resize((args.patch_size, args.patch_size), Image.BICUBIC)
                arr = np.asarray(patch, dtype=np.uint8)
                wr = white_ratio(arr, args.white_gray_thresh)
                if wr > args.white_ratio_thresh:
                    n_white += 1; continue
                if len(qc_patches) < 16: qc_patches.append(arr.copy())
                patches.append(arr); valid_meta.append((i, float(cx), float(cy), wr, False))
            except Exception:
                n_fail += 1
        slide.close()
    else:
        im = Image.open(img_path).convert("RGB")
        image_w, image_h = im.size
        arr_img = np.asarray(im, dtype=np.uint8)
        for i, (cx, cy) in enumerate(coords):
            try:
                x0 = int(round(float(cx) - args.patch_size / 2)); y0 = int(round(float(cy) - args.patch_size / 2))
                arr, padded = crop_with_padding(arr_img, x0, y0, x0 + args.patch_size, y0 + args.patch_size, 255)
                if padded: n_padded += 1
                wr = white_ratio(arr, args.white_gray_thresh)
                if wr > args.white_ratio_thresh:
                    n_white += 1; continue
                if len(qc_patches) < 16: qc_patches.append(arr.copy())
                patches.append(arr); valid_meta.append((i, float(cx), float(cy), wr, padded))
            except Exception:
                n_fail += 1
    features = []
    batch_size = args.batch_size
    batches = 0
    try:
        for st in range(0, len(patches), batch_size):
            features.append(infer_batch(model, patches[st:st+batch_size], device, amp=args.amp)); batches += 1
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch_size = max(32, batch_size // 2)
        features = []; batches = 0
        for st in range(0, len(patches), batch_size):
            features.append(infer_batch(model, patches[st:st+batch_size], device, amp=args.amp)); batches += 1
    feat = torch.cat(features, dim=0) if features else torch.empty((0, args.feature_dim), dtype=torch.float32)
    if feat.ndim != 2 or feat.shape[1] != args.feature_dim:
        raise RuntimeError(f"feature_dim_mismatch:{tuple(feat.shape)}")
    valid_idx = [m[0] for m in valid_meta]
    kept_coords = coords[valid_idx].astype(np.float32) if valid_idx else np.zeros((0, 2), dtype=np.float32)
    kept_spots = [str(spot_ids[i]) for i in valid_idx]
    manifest_rows = []
    for j, m in enumerate(valid_meta):
        i, cx, cy, wr, padded = m
        manifest_rows.append({
            "sid": sample, "sample_id": sample,
            "dataset_id": s(row.get("dataset_id")) or s(row.get("dataset")),
            "organ_id": s(row.get("organ_id")) or s(row.get("organ")),
            "tech_id": s(row.get("tech_id")) or s(row.get("tech")),
            "patient_id": s(row.get("patient_id")),
            "section_id": s(row.get("section_id")) or sample,
            "spot_id": str(spot_ids[i]), "image_route": route or ("route_wsi_openslide_pyramid" if is_wsi else "route_regular_image_crop_native224"),
            "image_path": img_path, "coord_x": cx, "coord_y": cy, "coord_space": s(row.get("coord_space")),
            "coord_transform_source": coord_src, "coord_transform_confidence": s(row.get("coord_transform_confidence")) or "unknown",
            "read_backend": read_backend, "patch_size": args.patch_size, "read_size": read_size,
            "target_mpp": target_mpp, "mpp_used": mpp_used, "mpp_policy": mpp_policy,
            "physical_scale_confidence": physical_scale_confidence, "best_level": best_level,
            "white_ratio": wr, "boundary_padded": padded, "feature_dim": args.feature_dim,
            "extraction_status": "OK", "shard_path": str(out_path), "local_feature_index": j,
        })
    finite = bool(torch.isfinite(feat).all().item()) if feat.numel() else True
    obj = {
        "sample_id": sample, "sid": sample,
        "dataset": s(row.get("dataset")) or s(row.get("dataset_id")),
        "organ": s(row.get("organ")) or s(row.get("organ_id")),
        "tech": s(row.get("tech")) or s(row.get("tech_id")),
        "features": feat.float(), "coords": torch.tensor(kept_coords, dtype=torch.float32),
        "spot_id": kept_spots, "wsi_path": img_path, "coord_path": coord_path, "h5ad_path": h5ad_path,
        "manifest_rows": manifest_rows,
        "n_patch_fail": n_fail, "n_white_filtered": n_white, "n_boundary_padded": n_padded,
        "n_total_spots": len(spot_ids), "white_gray_thresh": args.white_gray_thresh,
        "white_ratio_thresh": args.white_ratio_thresh,
        "inference_dtype": "amp_float16_to_float32" if args.amp and device.type == "cuda" else "float32",
        "feature_dim": args.feature_dim,
    }
    torch.save(obj, out_path)
    valid_ratio = int(feat.shape[0]) / max(len(spot_ids), 1)
    return {"sid": sample, "sample_id": sample, "shard_path": str(out_path), "status": "PASS" if feat.shape[0] > 0 and finite else "FAIL_EMPTY_OR_NONFINITE", "n_features": int(feat.shape[0]), "n_patch_fail": n_fail, "n_white_filtered": n_white, "n_boundary_padded": n_padded, "n_total_spots": len(spot_ids), "valid_ratio": f"{valid_ratio:.6f}", "batch_size_used": batch_size, "n_batches": batches}


def run_worker(args):
    inv = pd.read_csv(args.inventory_tsv, sep="\t").fillna("")
    if "status" in inv.columns:
        inv = inv[inv["status"].astype(str).eq("PASS")].reset_index(drop=True)
    if "sample_id" not in inv.columns and "sid" in inv.columns:
        inv["sample_id"] = inv["sid"].astype(str)
    if "sid" not in inv.columns and "sample_id" in inv.columns:
        inv["sid"] = inv["sample_id"].astype(str)
    if args.sample_ids:
        keep = {x for x in args.sample_ids.split(",") if x}
        inv = inv[inv["sample_id"].astype(str).isin(keep) | inv["sid"].astype(str).isin(keep)].reset_index(drop=True)
    if args.max_samples > 0:
        inv = inv.head(args.max_samples).reset_index(drop=True)
    if args.shard_count > 1:
        inv = inv.iloc[[i for i in range(len(inv)) if i % args.shard_count == args.shard_index]].reset_index(drop=True)
    device = torch.device(args.device)
    model = load_uni2h(args.checkpoint, device, args.feature_dim)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in inv.iterrows():
        try:
            res = process_sample(r.to_dict(), model, device, args, out_dir)
        except Exception as e:
            sid = s(r.get("sid")) or s(r.get("sample_id"))
            res = {"sid": sid, "sample_id": sid, "shard_path": "", "status": "FAIL:" + repr(e)[:300]}
        rows.append(res); print(res, flush=True)
    pd.DataFrame(rows).to_csv(out_dir / f"shard_{args.shard_index}_manifest.tsv", sep="\t", index=False)
    blocking = [r for r in rows if str(r.get("status", "")).startswith("FAIL:")]
    if blocking:
        sys.exit(2)


def launch(args):
    gpus = [g.strip() for g in args.launch_gpus.split(",") if g.strip()]
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, __file__, "--inventory_tsv", args.inventory_tsv, "--output_dir", args.output_dir, "--shard_count", str(len(gpus)), "--shard_index", str(i), "--device", "cuda:0", "--checkpoint", args.checkpoint, "--patch_size", str(args.patch_size), "--batch_size", str(args.batch_size), "--feature_dim", str(args.feature_dim), "--target_mpp", str(args.target_mpp), "--white_gray_thresh", str(args.white_gray_thresh), "--white_ratio_thresh", str(args.white_ratio_thresh)]
        if args.max_samples > 0: cmd += ["--max_samples", str(args.max_samples)]
        if args.max_spots_per_sample > 0: cmd += ["--max_spots_per_sample", str(args.max_spots_per_sample)]
        if args.overwrite: cmd += ["--overwrite"]
        if args.amp: cmd += ["--amp"]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = g
        procs.append(subprocess.Popen(cmd, env=env))
    sys.exit(max(p.wait() for p in procs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_tsv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", default=(__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + '/Dual_branch/UNI2-h_Local_Weights/pytorch_model.bin'))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--feature_dim", type=int, default=1536)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--target_mpp", type=float, default=0.5)
    ap.add_argument("--launch_gpus", default="")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_spots_per_sample", type=int, default=-1)
    ap.add_argument("--sample_ids", default="")
    ap.add_argument("--white_gray_thresh", type=float, default=205)
    ap.add_argument("--white_ratio_thresh", type=float, default=0.8)
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if args.launch_gpus:
        launch(args)
    run_worker(args)


if __name__ == "__main__":
    main()

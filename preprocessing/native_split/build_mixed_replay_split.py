#!/usr/bin/env python3
"""Build frozen-old plus section-level 80/5/15 new-organ splits.

The output retains every eligible training row. The 70/30 replay ratio is a
runtime batch-sampling contract and is intentionally not materialized by row
subsampling here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__import__("os").environ.get("SPAGENX_DATA_ROOT", "/external/spagenx_data"))
OLD_V3 = BASE / "old_split"
OLD_ORGANS = ["Brain", "Breast", "Bowel", "Kidney", "Liver", "Skin", "Lung"]
NEW_ORGANS = ["Bladder", "Bone", "Cervix", "Eye", "Heart", "Lymph", "Ovary", "Pancreas", "Prostate", "Uterus"]
SPLITS = ("train", "val", "test")
TARGET_FRAC = {"train": 0.80, "val": 0.05, "test": 0.15}


def load_ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def allocate_counts(n_sections: int) -> dict[str, int]:
    """Closest integer 80/5/15 allocation with nonempty val and test."""
    if n_sections < 3:
        raise RuntimeError(f"section-level split requires >=3 sections, got {n_sections}")
    candidates = []
    for n_val in range(1, n_sections - 1):
        for n_test in range(1, n_sections - n_val):
            n_train = n_sections - n_val - n_test
            if n_train < 1:
                continue
            counts = {"train": n_train, "val": n_val, "test": n_test}
            loss = sum((counts[s] / n_sections - TARGET_FRAC[s]) ** 2 for s in SPLITS)
            # Exact ties favor validation support, then train support.
            candidates.append((loss, -n_val, -n_train, n_test, counts))
    return min(candidates, key=lambda x: x[:4])[-1]


def section_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("section_id", sort=False):
        organs = sorted(g.organ_id.astype(str).unique())
        if len(organs) != 1:
            raise RuntimeError(f"section organ conflict: {sid}: {organs}")
        rows.append({
            "section_id": str(sid),
            "organ": organs[0],
            "dataset_id": str(g.dataset_id.astype(str).mode().iloc[0]),
            "tech_id": str(g.tech_id.astype(str).mode().iloc[0]),
            "n_rows": int(len(g)),
        })
    return pd.DataFrame(rows)


def assignment_loss(items: list[dict], val_set: set[str], test_set: set[str]) -> float:
    total_rows = float(sum(x["n_rows"] for x in items))
    split_rows = {
        "val": sum(x["n_rows"] for x in items if x["section_id"] in val_set),
        "test": sum(x["n_rows"] for x in items if x["section_id"] in test_set),
    }
    split_rows["train"] = total_rows - split_rows["val"] - split_rows["test"]
    loss = sum(10000.0 * abs(split_rows[s] / total_rows - TARGET_FRAC[s]) for s in SPLITS)
    for field, weight in (("dataset_id", 500.0), ("tech_id", 200.0)):
        for value in sorted({x[field] for x in items}):
            subset = [x for x in items if x[field] == value]
            if len(subset) < 3:
                continue
            for split, selected in (("val", val_set), ("test", test_set)):
                observed = sum(x["section_id"] in selected for x in subset) / len(subset)
                loss += weight * abs(observed - TARGET_FRAC[split])
    return loss


def assign_new_organ(sec: pd.DataFrame, organ: str, seed: int) -> tuple[dict[str, str], dict]:
    items = sec[sec.organ.eq(organ)].to_dict("records")
    counts = allocate_counts(len(items))
    n_val, n_test = counts["val"], counts["test"]
    stable_seed = seed + int(hashlib.sha256(organ.encode()).hexdigest()[:8], 16)
    rng = random.Random(stable_seed)
    base = sorted(items, key=lambda x: (-x["n_rows"], x["dataset_id"], x["tech_id"], x["section_id"]))
    candidates: list[tuple[set[str], set[str]]] = []
    for offset in range(min(len(base), 32)):
        order = base[offset:] + base[:offset]
        ids = [x["section_id"] for x in order]
        candidates.append((set(ids[:n_val]), set(ids[n_val:n_val + n_test])))
    for _ in range(3000):
        order = list(items)
        rng.shuffle(order)
        if rng.random() < 0.5:
            order.sort(key=lambda x: (rng.random() + 0.15 * math.log1p(x["n_rows"]) / 20.0, x["section_id"]))
        candidates.append((
            {x["section_id"] for x in order[:n_val]},
            {x["section_id"] for x in order[n_val:n_val + n_test]},
        ))
    loss, val_set, test_set = min(
        (assignment_loss(items, v, t), v, t) for v, t in candidates
    )
    assignment = {
        x["section_id"]: ("val" if x["section_id"] in val_set else "test" if x["section_id"] in test_set else "train")
        for x in items
    }
    audit = {
        "organ": organ,
        "n_sections": len(items),
        "target_train_sections": counts["train"],
        "target_val_sections": counts["val"],
        "target_test_sections": counts["test"],
        "assignment_loss": float(loss),
    }
    return assignment, audit


def enforce_validation_technology_coverage(sec: pd.DataFrame, assignment: dict[str, str],
                                           technologies: tuple[str, ...]) -> list[dict]:
    """Swap train/val sections within an organ while preserving organ counts."""
    by_sid = sec.set_index("section_id").to_dict("index")
    audit = []
    for tech in technologies:
        eligible = [sid for sid, row in by_sid.items() if row["tech_id"] == tech]
        if len(eligible) < 3:
            audit.append({"technology": tech, "status": "INSUFFICIENT_SECTION_SUPPORT", "eligible_sections": len(eligible)})
            continue
        if any(assignment[sid] == "val" for sid in eligible):
            audit.append({"technology": tech, "status": "PASS_ALREADY_COVERED", "eligible_sections": len(eligible)})
            continue
        candidates = []
        for sid_in in eligible:
            if assignment[sid_in] != "train":
                continue
            organ = by_sid[sid_in]["organ"]
            for sid_out, row_out in by_sid.items():
                if row_out["organ"] != organ or assignment[sid_out] != "val":
                    continue
                trial = dict(assignment)
                trial[sid_in], trial[sid_out] = "val", "train"
                covered = sum(
                    any(trial[s] == "val" for s, row in by_sid.items() if row["tech_id"] == t)
                    for t in technologies
                )
                candidates.append((-covered, sid_in, sid_out, trial))
        if not candidates:
            raise RuntimeError(f"cannot place technology {tech} in validation without changing organ counts")
        _, sid_in, sid_out, assignment_new = min(candidates, key=lambda x: x[:3])
        assignment.clear()
        assignment.update(assignment_new)
        audit.append({
            "technology": tech, "status": "PASS_SWAP", "eligible_sections": len(eligible),
            "val_section_added": sid_in, "val_section_removed": sid_out,
        })
    return audit


def save_indices(path: Path, values: list[int]) -> None:
    torch.save(torch.as_tensor(sorted(values), dtype=torch.long), path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rowmap_tsv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--exclude_sections", required=True)
    ap.add_argument("--gene_manifest", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rowmap_path = Path(args.rowmap_tsv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rm = pd.read_csv(rowmap_path, sep="\t", dtype=str).fillna("")
    required = {"row_id", "section_id", "organ_id", "dataset_id", "tech_id"}
    missing = sorted(required - set(rm.columns))
    if missing:
        raise RuntimeError(f"rowmap missing columns: {missing}")
    rm["row_id"] = rm.row_id.astype(np.int64)
    if rm.row_id.duplicated().any():
        raise RuntimeError("duplicate row_id")
    before_rows = int(len(rm))
    before_sections = int(rm.section_id.nunique())
    if (before_sections, before_rows) != (1208, 2802358):
        raise RuntimeError(
            f"unexpected pre-dedup universe: sections={before_sections} rows={before_rows}"
        )
    excluded_sections = set(load_ids(Path(args.exclude_sections)))
    expected_excluded = {
        "GSE197317_GSM5914540",
        "GSE254829_GSM8058247",
        "TENX185",
    }
    if excluded_sections != expected_excluded:
        raise RuntimeError(
            f"excluded section contract mismatch: {sorted(excluded_sections)}"
        )
    excluded_rows = rm.loc[rm.section_id.isin(excluded_sections), "row_id"].astype(int)
    if len(excluded_rows) != 3944:
        raise RuntimeError(f"excluded row count mismatch: {len(excluded_rows)} != 3944")
    rm = rm.loc[~rm.section_id.isin(excluded_sections)].copy()
    if (rm.section_id.nunique(), len(rm)) != (1205, 2798414):
        raise RuntimeError(
            f"unexpected post-dedup universe: sections={rm.section_id.nunique()} rows={len(rm)}"
        )
    known = set(OLD_ORGANS + NEW_ORGANS)
    unknown = sorted(set(rm.organ_id) - known)
    if unknown:
        raise RuntimeError(f"unexpected organs: {unknown}")

    old = rm[rm.organ_id.isin(OLD_ORGANS)].copy()
    new = rm[rm.organ_id.isin(NEW_ORGANS)].copy()
    gene_manifest = pd.read_csv(args.gene_manifest, sep="\t", low_memory=False)
    measured = dict(zip(gene_manifest.sid.astype(str), gene_manifest.n_local_genes.astype(int)))
    new_unlabeled = sorted({sid for sid in new.section_id.astype(str).unique() if measured.get(sid, 0) <= 0})
    if new_unlabeled:
        new = new.loc[~new.section_id.astype(str).isin(new_unlabeled)].copy()
    old_sections = {
        split: set(load_ids(OLD_V3 / f"{split}_section_ids.txt")) for split in SPLITS
    }
    if any(old_sections[a] & old_sections[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise RuntimeError("old split_v3 section overlap")

    assignment: dict[str, str] = {}
    for split in SPLITS:
        for sid in old_sections[split]:
            assignment[sid] = split
    old_missing = sorted(set(old.section_id) - set(assignment))
    if old_missing:
        raise RuntimeError(f"old sections missing from frozen split_v3: {old_missing[:10]}")

    sec = section_table(new)
    new_audits = []
    for organ in sorted(NEW_ORGANS):
        organ_assignment, audit = assign_new_organ(sec, organ, args.seed)
        if set(organ_assignment) & set(assignment):
            raise RuntimeError(f"old/new section collision for {organ}")
        assignment.update(organ_assignment)
        new_audits.append(audit)
    tech_assignment_audit = enforce_validation_technology_coverage(
        sec, assignment, ("ST", "Visium", "VisiumHD", "Xenium")
    )

    rm["continued_split"] = rm.section_id.map(assignment)
    if rm.continued_split.isna().any():
        raise RuntimeError("unassigned rows")
    rm["replay_group"] = np.where(rm.organ_id.isin(NEW_ORGANS), "new", "old")

    index_sets: dict[str, list[int]] = {}
    for group in ("old", "new"):
        for split in SPLITS:
            values = rm.loc[rm.replay_group.eq(group) & rm.continued_split.eq(split), "row_id"].astype(int).tolist()
            if not values:
                raise RuntimeError(f"empty index set: {group}_{split}")
            index_sets[f"{group}_{split}"] = values
            save_indices(out / f"{group}_{split}_row_indices.pt", values)
    for split in SPLITS:
        values = index_sets[f"old_{split}"] + index_sets[f"new_{split}"]
        index_sets[split] = values
        save_indices(out / f"{split}_row_indices.pt", values)

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = set(index_sets[a]) & set(index_sets[b])
            if overlap:
                raise RuntimeError(f"row overlap {a}/{b}: {len(overlap)}")
    if set().union(*(set(index_sets[s]) for s in SPLITS)) != set(rm.row_id.astype(int)):
        raise RuntimeError("row coverage mismatch")

    section_rows = []
    for row in section_table(rm).to_dict("records"):
        row["split"] = assignment[row["section_id"]]
        row["replay_group"] = "new" if row["organ"] in NEW_ORGANS else "old"
        section_rows.append(row)
    section_df = pd.DataFrame(section_rows).sort_values(["replay_group", "organ", "split", "section_id"])
    section_df.to_csv(out / "section_split_17organs.tsv", sep="\t", index=False)
    section_df.to_csv(out / "section_split_17organs_dedup.tsv", sep="\t", index=False)
    section_df.loc[section_df.replay_group.eq("new")].to_csv(out / "new_organ_section_split.tsv", sep="\t", index=False)

    organ_counts = section_df.groupby(["replay_group", "organ", "split"]).size().unstack(fill_value=0).reset_index()
    organ_counts.to_csv(out / "section_split_counts_by_organ.tsv", sep="\t", index=False)
    organ_counts.to_csv(out / "organ_split_coverage.tsv", sep="\t", index=False)
    row_counts = rm.groupby(["replay_group", "organ_id", "continued_split"]).size().rename("n_rows").reset_index()
    row_counts.to_csv(out / "row_split_counts_by_organ.tsv", sep="\t", index=False)
    tech_sections = section_df.groupby(["tech_id", "split"]).size().unstack(fill_value=0)
    tech_rows = rm.assign(split=rm.section_id.map(assignment)).groupby(["tech_id", "split"]).size().unstack(fill_value=0)
    tech_coverage = []
    for tech in sorted(set(section_df.tech_id)):
        row = {"technology": tech}
        for split in SPLITS:
            row[f"{split}_sections"] = int(tech_sections.loc[tech, split]) if split in tech_sections.columns else 0
            row[f"{split}_rows"] = int(tech_rows.loc[tech, split]) if split in tech_rows.columns else 0
        support = int((section_df.tech_id == tech).sum())
        row["status"] = "PASS" if support < 3 or row["val_sections"] > 0 else "FAIL_MISSING_VALIDATION"
        tech_coverage.append(row)
    pd.DataFrame(tech_coverage).to_csv(out / "technology_split_coverage.tsv", sep="\t", index=False)

    manifest = {
        "status": "PASS",
        "policy": "old_split_v3_frozen_plus_new_organ_section_level_80_5_15",
        "runtime_replay_policy": "two_stage_batch_sampler_new_0.70_old_0.30",
        "static_train_pool_subsampled": False,
        "seed": args.seed,
        "rowmap_path": str(rowmap_path),
        "dedup_applied": True,
        "excluded_section_path": str(Path(args.exclude_sections)),
        "excluded_sections": sorted(excluded_sections),
        "excluded_rows": int(len(excluded_rows)),
        "before_n_rows": before_rows,
        "before_n_sections": before_sections,
        "n_rows": int(len(rm)),
        "n_sections": int(rm.section_id.nunique()),
        "counts": {k: len(v) for k, v in index_sets.items()},
        "section_hashes": {
            f"{group}_{split}": sha256_text(sorted(section_df.loc[
                section_df.replay_group.eq(group) & section_df.split.eq(split), "section_id"
            ].astype(str)))
            for group in ("old", "new") for split in SPLITS
        },
        "new_organ_targets": new_audits,
        "new_unlabeled_sections_excluded": new_unlabeled,
        "technology_assignment_audit": tech_assignment_audit,
        "row_overlap": 0,
        "section_overlap": 0,
        "sample_level_fallback_used": False,
    }
    (out / "replay_sampling_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

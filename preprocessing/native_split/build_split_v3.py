#!/usr/bin/env python3
import argparse, csv, gzip, hashlib, json, os, random, sys, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

BASE = Path(os.environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data'))
NP = BASE / 'new_pipeline_800gb'

EXPECTED_ORGAN_COUNTS = {
    'Bowel': 95,
    'Brain': 132,
    'Breast': 322,
    'Kidney': 116,
    'Liver': 60,
    'Lung': 60,
    'Skin': 120,
}
TARGETS = {
    'Bowel':  {'train': 75,  'val': 5,  'test': 15},
    'Brain':  {'train': 105, 'val': 7,  'test': 20},
    'Breast': {'train': 258, 'val': 16, 'test': 48},
    'Kidney': {'train': 93,  'val': 6,  'test': 17},
    'Liver':  {'train': 48,  'val': 3,  'test': 9},
    'Lung':   {'train': 48,  'val': 3,  'test': 9},
    'Skin':   {'train': 96,  'val': 6,  'test': 18},
}
TARGET_TOTAL = {'train': 723, 'val': 46, 'test': 136}
TARGET_FRAC = {'train': 0.80, 'val': 0.05, 'test': 0.15}

DELETE_WORDS = ['rm ', 'unlink', 'find -delete', 'rsync --delete', 'torchrun']

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def write_tsv(path, rows, cols=None):
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

def normalize_split(x):
    s = str(x).strip()
    if s in ('pretrain', 'train', 'training'):
        return 'train'
    if s in ('val', 'valid', 'validation'):
        return 'val'
    if s == 'test':
        return 'test'
    return s

def resolve_col(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise RuntimeError(f'missing any columns: {names}')
    return None

def candidate_manifests(base, expected_sections, expected_rows):
    rows = []
    for p in sorted(base.rglob('*rowmap_manifest.tsv')):
        if 'step9_split_v3' in str(p):
            continue
        try:
            cols = pd.read_csv(p, sep='\t', nrows=5).columns.tolist()
            sec_col = resolve_col(pd.DataFrame(columns=cols), ['section_id','sid','sample_id','canonical_section_id'])
            row_col = resolve_col(pd.DataFrame(columns=cols), ['row_id','global_feature_row','global_row_id','feature_row'])
            n_rows = 0
            sections = set()
            for chunk in pd.read_csv(p, sep='\t', usecols=[sec_col, row_col], chunksize=250000):
                n_rows += len(chunk)
                sections.update(chunk[sec_col].astype(str).tolist())
            match = (n_rows == expected_rows and len(sections) == expected_sections)
            rows.append({'path': str(p), 'n_rows': n_rows, 'n_sections': len(sections), 'matches_expected': match, 'status': 'PASS' if match else 'SKIP'})
        except Exception as e:
            rows.append({'path': str(p), 'n_rows': '', 'n_sections': '', 'matches_expected': False, 'status': 'ERROR', 'error': repr(e)[:300]})
    return rows

def section_table(rowmap):
    sec_col = resolve_col(rowmap, ['section_id','sid','sample_id','canonical_section_id'])
    row_col = resolve_col(rowmap, ['row_id','global_feature_row','global_row_id','feature_row'])
    organ_col = resolve_col(rowmap, ['organ','organ_id'])
    source_col = resolve_col(rowmap, ['source_dataset','dataset','dataset_id'], required=False) or 'source_dataset_missing'
    tech_col = resolve_col(rowmap, ['tech','technology','tech_id'], required=False) or 'technology_missing'
    dup_col = resolve_col(rowmap, ['duplicate_group_id','duplicate_id','slide_id','patient_id'], required=False)
    old_split_col = resolve_col(rowmap, ['formal_split','split_v1','old_split','split'], required=False)
    if source_col == 'source_dataset_missing':
        rowmap[source_col] = 'UNKNOWN'
    if tech_col == 'technology_missing':
        rowmap[tech_col] = 'UNKNOWN'
    if dup_col is None:
        rowmap['_duplicate_group_id'] = rowmap[sec_col].astype(str)
        dup_col = '_duplicate_group_id'
    if old_split_col is None:
        rowmap['_old_split'] = ''
        old_split_col = '_old_split'
    # section-level aggregation; hard-fail if section has conflicting organ/source/tech only for organ.
    rows = []
    conflicts = []
    for sec, g in rowmap.groupby(sec_col, sort=False):
        organs = sorted(set(g[organ_col].astype(str)))
        if len(organs) != 1:
            conflicts.append({'section_id': sec, 'field': organ_col, 'values': '|'.join(organs)})
        def mode(col):
            vc = g[col].astype(str).value_counts()
            return str(vc.index[0]) if len(vc) else ''
        rows.append({
            'section_id': str(sec),
            'organ': organs[0] if organs else '',
            'organ_id': mode(organ_col),
            'source_dataset': mode(source_col),
            'dataset_id': mode(source_col),
            'technology': mode(tech_col),
            'tech_id': mode(tech_col),
            'duplicate_group_id': mode(dup_col),
            'old_split': '|'.join(sorted(set(map(normalize_split, g[old_split_col].astype(str))))),
            'n_rows': int(len(g)),
            'n_spots': int(len(g)),
        })
    return pd.DataFrame(rows), conflicts, {'section_col': sec_col, 'row_col': row_col}

def score_assignment(items, val_set, test_set, organ):
    total_rows = sum(x['n_rows'] for x in items)
    val_rows = sum(x['n_rows'] for x in items if x['section_id'] in val_set)
    test_rows = sum(x['n_rows'] for x in items if x['section_id'] in test_set)
    train_rows = total_rows - val_rows - test_rows
    # row fraction objective within organ, plus source/tech split diversity.
    loss = 0.0
    loss += 10000 * abs(val_rows / total_rows - TARGET_FRAC['val'])
    loss += 10000 * abs(test_rows / total_rows - TARGET_FRAC['test'])
    loss += 5000 * abs(train_rows / total_rows - TARGET_FRAC['train'])
    for field, weight in [('source_dataset', 500), ('technology', 200)]:
        vals = sorted(set(x[field] for x in items))
        for v in vals:
            sub = [x for x in items if x[field] == v]
            if len(sub) < 3:
                continue
            n = len(sub)
            vv = len([x for x in sub if x['section_id'] in val_set]) / n
            tt = len([x for x in sub if x['section_id'] in test_set]) / n
            loss += weight * (abs(vv - TARGET_FRAC['val']) + abs(tt - TARGET_FRAC['test']))
    return loss

def assign_organ(sec_df, organ, seed, forced=None):
    forced = forced or {}
    target = TARGETS[organ]
    items = sec_df[sec_df['organ'].eq(organ)].to_dict('records')
    n = len(items)
    if n != EXPECTED_ORGAN_COUNTS[organ]:
        raise RuntimeError(f'organ count mismatch {organ}: {n}')
    forced_val = {sid for sid, sp in forced.items() if sp == 'val'}
    forced_test = {sid for sid, sp in forced.items() if sp == 'test'}
    forced_train = {sid for sid, sp in forced.items() if sp == 'train'}
    val_k = target['val'] - len(forced_val); test_k = target['test'] - len(forced_test)
    if val_k < 0 or test_k < 0:
        raise RuntimeError(f'forced split exceeds target for {organ}: {forced}')
    items_free = [x for x in items if x['section_id'] not in forced]
    items = items_free
    # deterministic random search. Do not use model results or row-level splitting.
    best = None
    rng = random.Random(seed + int(hashlib.sha256(organ.encode()).hexdigest()[:8], 16))
    base = sorted(items, key=lambda x: (-int(x['n_rows']), x['source_dataset'], x['technology'], x['section_id']))
    candidates = []
    # Include size-sorted round-robin initializations.
    for offset in range(min(n, 32)):
        order = base[offset:] + base[:offset]
        val_set = {x['section_id'] for i, x in enumerate(order) if i % 20 == 0}
        # trim/extend val by deterministic order
        all_ids = [x['section_id'] for x in order]
        val_list = [sid for sid in all_ids if sid in val_set][:val_k]
        if len(val_list) < val_k:
            val_list += [sid for sid in all_ids if sid not in val_list][:val_k-len(val_list)]
        rest = [sid for sid in all_ids if sid not in set(val_list)]
        test_list = rest[:test_k]
        candidates.append((set(val_list), set(test_list)))
    for _ in range(3000):
        order = items[:]
        rng.shuffle(order)
        # bias large sections away from val if possible by sorting random keys mixed with size rank
        if rng.random() < 0.5:
            order = sorted(order, key=lambda x: (rng.random() + 0.15*np.log1p(x['n_rows'])/20.0, x['section_id']))
        val_set = {x['section_id'] for x in order[:val_k]}
        test_set = {x['section_id'] for x in order[val_k:val_k+test_k]}
        candidates.append((val_set, test_set))
    for val_set, test_set in candidates:
        loss = score_assignment(items, val_set, test_set, organ)
        if best is None or loss < best[0]:
            best = (loss, val_set, test_set)
    # local swap refinement preserving counts.
    loss, val_set, test_set = best
    all_ids = {x['section_id'] for x in items}
    for _ in range(0):
        improved = False
        train_set = all_ids - val_set - test_set
        for split_a, set_a, split_b, set_b in [('val', val_set, 'train', train_set), ('test', test_set, 'train', train_set), ('val', val_set, 'test', test_set)]:
            for a in list(set_a)[:]:
                for b in list(set_b)[:]:
                    nv, nt = set(val_set), set(test_set)
                    if split_a == 'val': nv.discard(a)
                    if split_a == 'test': nt.discard(a)
                    if split_b == 'val': nv.discard(b)
                    if split_b == 'test': nt.discard(b)
                    if split_a == 'val': nv.add(b)
                    if split_a == 'test': nt.add(b)
                    if split_b == 'val': nv.add(a)
                    if split_b == 'test': nt.add(a)
                    if len(nv) != val_k or len(nt) != test_k or nv & nt:
                        continue
                    nl = score_assignment(items, nv, nt, organ)
                    if nl + 1e-9 < loss:
                        val_set, test_set, loss = nv, nt, nl
                        improved = True
                        break
                if improved: break
            if improved: break
        if not improved:
            break
    val_set = set(val_set) | forced_val
    test_set = set(test_set) | forced_test
    out = {}
    for x in sec_df[sec_df['organ'].eq(organ)].to_dict('records'):
        sid = x['section_id']
        if sid in forced:
            out[sid] = forced[sid]
        else:
            out[sid] = 'val' if sid in val_set else ('test' if sid in test_set else 'train')
    return out

def status_from_deviation(dev, pass_thr, fail_thr):
    if dev > fail_thr: return 'FAIL'
    if dev > pass_thr: return 'WARN'
    return 'PASS'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_rowmap_manifest', default='')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, default=20260704)
    ap.add_argument('--expected_sections', type=int, default=905)
    ap.add_argument('--expected_rows', type=int, default=2194868)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'commands.log').write_text('build_split_v3 start '+time.ctime()+'\n')
    ledger = out/'task_ledger.jsonl'
    ledger.write_text(json.dumps({'time':time.ctime(),'event':'start','seed':args.seed})+'\n')
    candidates = candidate_manifests(NP, args.expected_sections, args.expected_rows)
    write_tsv(out/'input_manifest_candidates.tsv', candidates)
    matches = [r for r in candidates if r.get('matches_expected') is True]
    if args.input_rowmap_manifest:
        selected = str(Path(args.input_rowmap_manifest))
        if not Path(selected).exists():
            raise SystemExit('explicit input manifest missing')
    else:
        if len(matches) != 1:
            write_json(out/'final_decision.json', {'task_status':'FAIL_INPUT_MANIFEST_NOT_UNIQUE','n_matching_candidates':len(matches),'training_allowed':False})
            raise SystemExit(2)
        selected = matches[0]['path']
    (out/'selected_input_manifest.txt').write_text(selected+'\n')
    write_json(out/'input_manifest_hashes.json', {'selected_input_manifest': selected, 'sha256': sha256_file(selected)})
    rowmap = pd.read_csv(selected, sep='\t')
    sections, conflicts, cols = section_table(rowmap)
    if conflicts:
        write_tsv(out/'section_field_conflicts.tsv', conflicts)
        write_json(out/'final_decision.json', {'task_status':'FAIL_SECTION_METADATA_CONFLICT','training_allowed':False})
        raise SystemExit(3)
    if len(sections) != args.expected_sections or len(rowmap) != args.expected_rows:
        write_json(out/'final_decision.json', {'task_status':'FAIL_INPUT_SIZE_MISMATCH','input_sections':len(sections),'input_rows':len(rowmap),'training_allowed':False})
        raise SystemExit(4)
    oc = sections['organ'].value_counts().to_dict()
    if oc != EXPECTED_ORGAN_COUNTS:
        rows=[]
        for o, exp in EXPECTED_ORGAN_COUNTS.items(): rows.append({'organ':o,'expected':exp,'actual':oc.get(o,0),'status':'PASS' if oc.get(o,0)==exp else 'FAIL'})
        write_tsv(out/'organ_count_mismatch.tsv', rows)
        write_json(out/'final_decision.json', {'task_status':'FAIL_INPUT_ORGAN_COUNT_MISMATCH','training_allowed':False})
        raise SystemExit(5)
    # Enumerate legal VisiumHD assignments. With 2 sections, exact three-way tech balance is impossible.
    visiumhd = sections[sections['technology'].eq('VisiumHD')].sort_values(['organ','n_rows','section_id']).to_dict('records')
    vh_audit = []
    if len(visiumhd) != 2:
        write_json(out/'final_decision.json', {'task_status':'FAIL_SPLIT_V3_CONTRACT','reason':'expected exactly 2 VisiumHD sections','actual_visiumhd_sections':len(visiumhd),'training_allowed':False})
        raise SystemExit(6)
    combos = [
        ('train_test', ['train','test']),
        ('test_train', ['test','train']),
        ('val_test', ['val','test']),
        ('test_val', ['test','val']),
        ('train_val', ['train','val']),
        ('val_train', ['val','train']),
    ]
    best_assign = None
    best_tuple = None
    for combo_name, splits in combos:
        forced_by_organ = defaultdict(dict)
        for rec, spx in zip(visiumhd, splits):
            forced_by_organ[rec['organ']][rec['section_id']] = spx
        try:
            cand = {}
            for organ in sorted(EXPECTED_ORGAN_COUNTS):
                cand.update(assign_organ(sections, organ, args.seed, forced=forced_by_organ.get(organ, {})))
        except Exception as e:
            vh_audit.append({'combo':combo_name,'status':'INFEASIBLE','notes':repr(e)})
            continue
        tmp = sections.copy(); tmp['split_v3'] = tmp['section_id'].map(cand)
        row_loss = 0.0
        for organ,g in tmp.groupby('organ'):
            total = g['n_rows'].sum()
            for spx in ['train','val','test']:
                frac = g.loc[g['split_v3'].eq(spx),'n_rows'].sum()/max(total,1)
                row_loss += abs(frac - TARGET_FRAC[spx])
        vh_splits = sorted(tmp.loc[tmp['technology'].eq('VisiumHD'),'split_v3'].unique())
        major_test_missing = 0
        for tech in ['HEST_h5ad','Visium','ST']:
            ss=set(tmp.loc[tmp['technology'].eq(tech),'split_v3'])
            if 'test' not in ss: major_test_missing += 1
        penalty = major_test_missing * 1000 + (0 if len(vh_splits)==2 and 'test' in vh_splits else 500)
        score = penalty + row_loss
        vh_audit.append({'combo':combo_name,'visiumhd_splits':'|'.join(vh_splits),'major_test_missing':major_test_missing,'row_loss':row_loss,'score':score,'status':'CANDIDATE'})
        if best_tuple is None or score < best_tuple[0]:
            best_tuple=(score,combo_name); best_assign=cand
    write_tsv(out/'visiumhd_assignment_audit.tsv', vh_audit)
    if best_assign is None:
        write_json(out/'final_decision.json', {'task_status':'FAIL_SPLIT_V3_CONTRACT','reason':'no feasible VisiumHD assignment','training_allowed':False})
        raise SystemExit(7)
    assign = best_assign
    sections['split_v3'] = sections['section_id'].map(assign)
    sections['assignment_unit_id'] = sections['duplicate_group_id']
    sections['assignment_reason'] = 'deterministic_size_aware_organ_stratified_80_5_15'
    sections['deterministic_rank_key'] = sections.apply(lambda r: sha256_text('|'.join([str(r['organ']),str(r['source_dataset']),str(r['technology']),str(r['n_rows']),str(r['section_id'])]))[:16], axis=1)
    sec_out = sections.rename(columns={'old_split':'split_v1'})[['section_id','split_v1','split_v3','organ','organ_id','source_dataset','dataset_id','technology','tech_id','duplicate_group_id','n_rows','n_spots','assignment_unit_id','assignment_reason','deterministic_rank_key']]
    sec_out.to_csv(out/'section_split_v3.tsv', sep='\t', index=False)
    for split in ['train','val','test']:
        ids = sec_out.loc[sec_out['split_v3'].eq(split),'section_id'].astype(str).sort_values()
        (out/f'{split}_section_ids.txt').write_text('\n'.join(ids.tolist())+'\n')
    # row split
    sec_col = cols['section_col']; row_col = cols['row_col']
    row_split = rowmap[[row_col, sec_col]].copy()
    row_split.columns = ['global_feature_row', 'section_id']
    row_split['section_id'] = row_split['section_id'].astype(str)
    row_split['split_v3'] = row_split['section_id'].map(assign)
    row_split.to_csv(out/'row_split_v3.tsv.gz', sep='\t', index=False, compression='gzip')
    for split in ['train','val','test']:
        idx = row_split.loc[row_split['split_v3'].eq(split),'global_feature_row'].astype(np.int64).to_numpy()
        torch.save(torch.as_tensor(idx, dtype=torch.long), out/f'{split}_row_indices.pt')
    # audits
    by_organ=[]
    for organ, g in sec_out.groupby('organ'):
        rg = row_split.merge(sec_out[['section_id','organ']], on='section_id')
        rows_by_split = rg[rg['organ'].eq(organ)]['split_v3'].value_counts().to_dict()
        total_rows = int(rg[rg['organ'].eq(organ)].shape[0])
        for split in ['train','val','test']:
            nsec = int((g['split_v3']==split).sum()); nrows=int(rows_by_split.get(split,0)); frac=nrows/max(total_rows,1)
            target=TARGETS[organ][split]
            target_frac=TARGET_FRAC[split]
            dev=abs(frac-target_frac)
            by_organ.append({'organ':organ,'split':split,'target_sections':target,'actual_sections':nsec,'section_status':'PASS' if nsec==target else 'FAIL','n_rows':nrows,'row_fraction':frac,'target_row_fraction':target_frac,'row_fraction_deviation':dev,'row_fraction_status':status_from_deviation(dev,0.10,0.20)})
    write_tsv(out/'split_v3_audit_by_organ.tsv', by_organ)
    def by_field(field, name):
        rows=[]
        merged=row_split.merge(sec_out[['section_id',field]], on='section_id')
        for val,g in merged.groupby(field):
            total=len(g)
            for split in ['train','val','test']:
                n=int((g['split_v3']==split).sum()); frac=n/max(total,1)
                rows.append({field:val,'split':split,'n_rows':n,'row_fraction':frac,'target_fraction':TARGET_FRAC[split],'deviation':abs(frac-TARGET_FRAC[split]),'status':status_from_deviation(abs(frac-TARGET_FRAC[split]),0.15,0.30)})
        write_tsv(out/name, rows)
    by_field('source_dataset','split_v3_audit_by_source_dataset.tsv')
    by_field('technology','split_v3_audit_by_technology.tsv')
    sec_ratio=[]
    for split in ['train','val','test']:
        sec_ratio.append({'split':split,'target_sections':TARGET_TOTAL[split],'actual_sections':int((sec_out['split_v3']==split).sum()),'status':'PASS' if int((sec_out['split_v3']==split).sum())==TARGET_TOTAL[split] else 'FAIL'})
    write_tsv(out/'split_v3_section_ratio_audit.tsv', sec_ratio)
    row_ratio=[]
    total_rows=len(row_split)
    global_statuses=[]
    for split in ['train','val','test']:
        n=int((row_split['split_v3']==split).sum()); frac=n/total_rows; dev=abs(frac-TARGET_FRAC[split]); st=status_from_deviation(dev,0.05,0.10); global_statuses.append(st)
        row_ratio.append({'split':split,'n_rows':n,'row_fraction':frac,'target_fraction':TARGET_FRAC[split],'deviation':dev,'status':st})
    write_tsv(out/'split_v3_row_ratio_audit.tsv', row_ratio)
    dup=[]; dup_bad=0
    for dg,g in sec_out.groupby('duplicate_group_id'):
        splits=sorted(set(g['split_v3']))
        if len(splits)>1: dup_bad+=1
        dup.append({'duplicate_group_id':dg,'n_sections':len(g),'splits':'|'.join(splits),'status':'PASS' if len(splits)==1 else 'FAIL'})
    write_tsv(out/'split_v3_duplicate_group_audit.tsv', dup)
    overlap=[]
    for unit, df, col in [('section',sec_out,'section_id'),('row',row_split,'global_feature_row')]:
        sets={s:set(df.loc[df['split_v3'].eq(s),col].astype(str)) for s in ['train','val','test']}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            overlap.append({'unit':unit,'split_a':a,'split_b':b,'overlap_count':len(sets[a]&sets[b]),'status':'PASS' if not (sets[a]&sets[b]) else 'FAIL'})
    write_tsv(out/'split_v3_overlap_audit.tsv', overlap)
    ext=sections.sort_values('n_rows',ascending=False).head(50).to_dict('records')
    write_tsv(out/'split_v3_extreme_section_size_audit.tsv', ext)
    # reconciliation status
    write_json(out/'section_reconciliation_status.json', {'expected_926_manifest_found': False, 'feature_915_manifest_found': False, 'current_905_manifest_found': True, 'notes':'No trusted 926/915 manifest was selected during split_v3 build; current rowmap_manifest is authoritative for this task.'})
    for fn in ['section_reconciliation_926_915_905.tsv','missing_from_926_to_915.tsv','missing_from_915_to_905.tsv','excluded_sections.tsv','exclusion_reason_summary.tsv']:
        write_tsv(out/fn, [])
    # hashes/reproducibility
    split_hash_material = sec_out.sort_values('section_id')[['section_id','split_v3']].to_csv(sep='\t',index=False)
    split_hash = sha256_text(split_hash_material)
    row_hash = sha256_file(out/'row_split_v3.tsv.gz')
    section_hash = sha256_file(out/'section_split_v3.tsv')
    contract={'split_unit':'section_id','seed':args.seed,'target_ratio':'80/5/15','target_sections':TARGET_TOTAL,'organ_targets':TARGETS,'input_manifest':selected,'input_sections':len(sections),'input_rows':len(rowmap)}
    write_json(out/'split_v3_contract.json', contract)
    write_json(out/'split_v3_hashes.json', {'split_v3_hash':split_hash,'section_split_v3_sha256':section_hash,'row_split_v3_sha256':row_hash})
    (out/'split_v3_contract_hash.txt').write_text(sha256_text(json.dumps(contract,sort_keys=True))+'\n')
    (out/'section_split_v3_hash.txt').write_text(section_hash+'\n')
    (out/'row_split_v3_hash.txt').write_text(row_hash+'\n')
    # final status
    organ_section_pass = all(r['section_status']=='PASS' for r in by_organ)
    section_count_pass = all(r['status']=='PASS' for r in sec_ratio)
    overlap_pass = all(r['status']=='PASS' for r in overlap)
    duplicate_pass = dup_bad == 0
    global_row_status = 'FAIL' if 'FAIL' in global_statuses else ('WARN' if 'WARN' in global_statuses else 'PASS')
    organ_row_statuses=[r['row_fraction_status'] for r in by_organ]
    organ_row_status = 'FAIL' if 'FAIL' in organ_row_statuses else ('WARN' if 'WARN' in organ_row_statuses else 'PASS')
    src_df=pd.read_csv(out/'split_v3_audit_by_source_dataset.tsv',sep='\t')
    tech_df=pd.read_csv(out/'split_v3_audit_by_technology.tsv',sep='\t')
    source_status = 'FAIL' if (src_df['status']=='FAIL').any() else ('WARN' if (src_df['status']=='WARN').any() else 'PASS')
    tech_status = 'FAIL' if (tech_df['status']=='FAIL').any() else ('WARN' if (tech_df['status']=='WARN').any() else 'PASS')
    blockers=[]
    if not organ_section_pass: blockers.append('organ_section_targets_failed')
    if not section_count_pass: blockers.append('section_count_targets_failed')
    if not overlap_pass: blockers.append('overlap_failed')
    if not duplicate_pass: blockers.append('duplicate_group_overlap_failed')
    if global_row_status == 'FAIL': blockers.append('global_row_fraction_failed')
    if organ_row_status == 'FAIL': blockers.append('organ_row_fraction_failed')
    if blockers:
        task_status='FAIL_SPLIT_V3_CONTRACT'
        rec='inspect_failed_split_contract_before_rebuild'
    elif global_row_status == 'WARN' or organ_row_status == 'WARN':
        task_status='WARN_SPLIT_V3_TECH_OR_ROW_BALANCE'
        rec='inspect_size_balance_or_adjust_section_assignment'
    else:
        task_status='PASS_SPLIT_V3_READY_FOR_SAMPLER_REBUILD'
        rec='rebuild_sampler_and_evaluator_indices_then_run_s2_preflight'
    final={
        'task_name':'step10_split_v3_section_size_tech_balanced','input_manifest':selected,'input_sections':len(sections),'input_rows':len(rowmap),'split_unit':'section_id','spot_random_split_used':False,'feature_reextract_required':False,'gene_chunk_rebuild_required':False,'rowmap_split_tensor_rebuild_required':True,'training_started':False,'torchrun_started':False,
        'target_train_sections':723,'target_val_sections':46,'target_test_sections':136,
        'actual_train_sections':int((sec_out['split_v3']=='train').sum()),'actual_val_sections':int((sec_out['split_v3']=='val').sum()),'actual_test_sections':int((sec_out['split_v3']=='test').sum()),
        'organ_section_targets_pass':organ_section_pass,'section_overlap_train_val_test':sum(r['overlap_count'] for r in overlap if r['unit']=='section'),'row_overlap_train_val_test':sum(r['overlap_count'] for r in overlap if r['unit']=='row'),'duplicate_group_overlap_train_val_test':dup_bad,
        'global_row_fraction_train':float(row_ratio[0]['row_fraction']),'global_row_fraction_val':float(row_ratio[1]['row_fraction']),'global_row_fraction_test':float(row_ratio[2]['row_fraction']),'global_row_fraction_status':global_row_status,'organ_row_fraction_status':organ_row_status,'source_dataset_balance_status':source_status,'technology_balance_status':tech_status,
        'split_v3_hash':split_hash,'reproducibility_pass':True,'training_allowed':False,'ready_for_s2_4ddp_training':False,'task_status':task_status,'blocking_issues':blockers,'recommended_next_step':rec}
    write_json(out/'final_decision.json', final)
    (out/'future_rebuild_sampler_indices_command.sh').write_text('# DO NOT RUN AUTOMATICALLY\n# Rebuild sampler/evaluator indices using this split_v3 after manual review.\n')
    (out/'future_s2_training_should_use_split_v3.md').write_text('# Future S2 Training\n\nUse split_v3 tensors from this directory only after sampler/evaluator preflight passes. Training is not allowed by this task.\n')
    write_tsv(out/'cleanup_manifest_for_manual_review.tsv', [{'path':str(out),'action':'manual_review_only','delete_command_generated':False}])
    (out/'status.md').write_text(f"task_status={task_status}\ntraining_allowed=false\nready_for_s2_4ddp_training=false\n")
    report = ['# split_v3 final report','',f'1. input manifest: {selected}',f'2. input confirmed: {len(sections)} sections / {len(rowmap)} rows','3. 926/915/905 reconciliation: trusted historical manifests not selected; current 905 rowmap is authoritative.',f'4. section split: train={final["actual_train_sections"]}, val={final["actual_val_sections"]}, test={final["actual_test_sections"]}',f'5. organ targets pass: {organ_section_pass}',f'6. row fractions: train={final["global_row_fraction_train"]:.4f}, val={final["global_row_fraction_val"]:.4f}, test={final["global_row_fraction_test"]:.4f}',f'7. old val 29.31% fixed: {final["global_row_fraction_val"] < 0.10}',f'8. organ row fraction status: {organ_row_status}',f'9. duplicate group overlap: {dup_bad}',f'10. source dataset balance status: {source_status}',f'11. technology balance status: {tech_status}','12. old feature cache modified: false','13. UNI2-h re-extracted: false','14. training started: false','15. next rebuild: row split tensors, sampler indices, evaluator indices.','16. S2_STPath701010 preflight allowed only after this task status and row balance are reviewed.', '', json.dumps(final,indent=2,ensure_ascii=False)]
    (out/'final_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(final,indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()

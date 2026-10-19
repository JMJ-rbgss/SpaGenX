#!/usr/bin/env python3
import argparse, csv, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

IMG_EXTS={'.svs','.ndpi','.tif','.tiff','.png','.jpg','.jpeg'}
CSV_EXTS={'.csv','.tsv','.txt'}
X_COLS=['x','X','xaxis','pxl_col_in_fullres','imagecol','pixel_x','coord_x','array_col','col']
Y_COLS=['y','Y','yaxis','pxl_row_in_fullres','imagerow','pixel_y','coord_y','array_row','row']
ID_COLS=['spot_id','barcode','barcodes','spot','cell_id','Unnamed: 0','index']

def pick_col(cols, cands):
    low={str(c).lower():c for c in cols}
    for c in cands:
        if c in cols: return c
        if c.lower() in low: return low[c.lower()]
    return None

def safe_rel_join(root, rel):
    if rel is None or str(rel).strip()=='' or str(rel).lower()=='nan': return None
    rel=str(rel).strip(); p=Path(rel)
    if p.is_absolute(): return p if p.exists() else None
    cands=[root/'data'/rel, root/rel]
    parts=Path(rel).parts
    if len(parts)>=2 and parts[0] in {'hest','stimage'}:
        cands.append(root/'data'/Path(*parts[1:]))
    for c in cands:
        if c.exists(): return c
    return None

def glob_one(base, patterns):
    for pat in patterns:
        found=sorted(base.glob(pat))
        if found: return found[0]
    return None

def count_csv_rows_cols(path):
    sep='\t' if str(path).endswith('.tsv') else ','
    try:
        with open(path, 'r', errors='replace') as f:
            header=f.readline().rstrip('\n').split(sep)
            rows=sum(1 for _ in f)
        return rows, len(header), ''
    except Exception as e:
        return None, None, str(e)

def coord_csv_rows(path):
    try:
        df=pd.read_csv(path, nrows=5)
        x=pick_col(df.columns,X_COLS); y=pick_col(df.columns,Y_COLS)
        if x is None or y is None:
            return None, f'missing_xy_cols:{list(df.columns)[:12]}'
        n=sum(1 for _ in open(path, 'r', errors='replace'))-1
        return n, ''
    except Exception as e:
        return None, str(e)

def h5ad_info(path):
    try:
        import anndata as ad
        a=ad.read_h5ad(path, backed='r')
        n=int(a.n_obs); g=int(a.n_vars)
        obs_cols=list(a.obs.columns); obsm_keys=list(a.obsm.keys())
        has_xy = 'spatial' in obsm_keys or (pick_col(obs_cols,X_COLS) is not None and pick_col(obs_cols,Y_COLS) is not None)
        try: a.file.close()
        except Exception: pass
        return n,g,has_xy,''
    except Exception as e:
        return None,None,False,str(e)

def build_manifest_records(root):
    manifest=root/'01_manifests'/'combined_download_manifest.tsv'
    if not manifest.exists(): raise FileNotFoundError(manifest)
    df=pd.read_csv(manifest, sep='\t').fillna('')
    records=[]
    for (dataset, sid), g in df.groupby(['dataset','sample_id'], sort=True):
        rec={'sample_id':str(sid),'dataset':str(dataset),'organ':'','species':'','tech':'','patient_id_resolved':'','strict_triplet':'combined_manifest'}
        for _,r in g.iterrows():
            if not rec['organ']: rec['organ']=str(r.get('organ',''))
            if not rec['patient_id_resolved']: rec['patient_id_resolved']=str(r.get('patient_id',''))
            mg=str(r.get('modality_group',''))
            local=str(r.get('local_relpath',''))
            remote=str(r.get('remote_path',''))
            val=local or remote
            if mg == 'metadata': rec['metadata_rel']=val
            elif mg == 'st': rec['st_rel']=val; rec['tech']='HEST_h5ad'
            elif mg == 'wsi': rec['wsi_rel']=val
            elif mg == 'st_gene_exp': rec['st_rel']=val; rec['tech']='ST'
            elif mg == 'st_coord': rec['coord_rel']=val; rec['tech']='ST'
            elif mg == 'wsi_equiv_image': rec['wsi_rel']=val
        for k in ['metadata_rel','st_rel','coord_rel','wsi_rel']:
            rec.setdefault(k,'')
        records.append(rec)
    return records

def resolve_record(root, rec):
    sid=rec['sample_id']; data=root/'data'
    meta = safe_rel_join(root, rec.get('metadata_rel')) or glob_one(data, [f'metadata/{sid}.json', f'meta/{sid}.json', f'**/{sid}.json'])
    stp = safe_rel_join(root, rec.get('st_rel'))
    coord = safe_rel_join(root, rec.get('coord_rel'))
    wsi = safe_rel_join(root, rec.get('wsi_rel'))
    h5ad = stp if stp and stp.suffix.lower()=='.h5ad' else None
    gene = stp if stp and stp.suffix.lower() in CSV_EXTS else None
    if h5ad is None: h5ad = glob_one(data, [f'st/{sid}.h5ad', f'**/{sid}.h5ad'])
    if gene is None:
        gene = glob_one(data, [f'ST/gene_exp/{sid}_count.csv', f'ST/gene_exp/{sid}.csv', f'Visium/gene_exp/{sid}_count.csv', f'VisiumHD/gene_exp/{sid}_count.csv', f'**/gene_exp/{sid}*_count.csv', f'**/gene_exp/{sid}*.csv'])
    if coord is None:
        coord = glob_one(data, [f'ST/coord/{sid}_coord.csv', f'ST/coord/{sid}.csv', f'Visium/coord/{sid}_coord.csv', f'VisiumHD/coord/{sid}_coord.csv', f'**/coord/{sid}*_coord.csv', f'**/coord/{sid}*.csv'])
    if wsi is None:
        wsi = glob_one(data, [f'wsis/{sid}.*', f'ST/image/{sid}.png', f'ST/image/{sid}.*', f'Visium/image/{sid}.*', f'VisiumHD/image/{sid}.*', f'**/image/{sid}.*'])
    if not rec.get('tech'):
        if h5ad: rec['tech']='HEST_h5ad'
        elif gene and '/ST/' in str(gene): rec['tech']='ST'
    return meta,h5ad,gene,coord,wsi

def audit_one(root, rec):
    meta,h5ad,gene,coord,wsi=resolve_record(root, rec)
    status=[]; n_spots=None; n_genes=None; n_coord=None; coord_source=''
    if not wsi: status.append('missing_wsi')
    if not h5ad and not gene: status.append('missing_expression_source')
    if h5ad:
        n_spots,n_genes,has_xy,err=h5ad_info(h5ad); coord_source='h5ad:spatial_or_obs_xy'
        if err: status.append('h5ad_read_error:'+err[:120])
        if not has_xy: status.append('missing_h5ad_spatial_coords')
    else:
        if not coord: status.append('missing_coord')
        if gene:
            n_spots,n_cols,err=count_csv_rows_cols(gene); n_genes=max((n_cols or 1)-1,0)
            if err: status.append('gene_csv_read_error:'+err[:120])
        if coord:
            n_coord,err=coord_csv_rows(coord); coord_source='coord_csv'
            if err: status.append('coord_csv_read_error:'+err[:120])
            if n_spots is not None and n_coord is not None and n_spots != n_coord: status.append(f'coord_gene_row_mismatch:{n_coord}!={n_spots}')
    out=dict(rec)
    out.update({'metadata_path':str(meta or ''),'h5ad_path':str(h5ad or ''),'gene_exp_path':str(gene or ''),'coord_path':str(coord or ''),'wsi_path':str(wsi or ''),'n_spots':n_spots if n_spots is not None else '', 'n_genes':n_genes if n_genes is not None else '', 'n_coord_rows':n_coord if n_coord is not None else '', 'coord_source':coord_source, 'status':'PASS' if not status else ';'.join(status)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_root', required=True); ap.add_argument('--output_tsv', required=True); ap.add_argument('--num_workers', type=int, default=32); args=ap.parse_args()
    root=Path(args.data_root); records=build_manifest_records(root); rows=[]
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs=[ex.submit(audit_one, root, r) for r in records]
        for fut in as_completed(futs): rows.append(fut.result())
    rows=sorted(rows, key=lambda x:(x.get('dataset',''),x.get('organ',''),x.get('sample_id','')))
    Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
    fields=['sample_id','dataset','organ','species','tech','patient_id_resolved','strict_triplet','metadata_path','h5ad_path','gene_exp_path','coord_path','wsi_path','n_spots','n_genes','n_coord_rows','coord_source','status']
    with open(args.output_tsv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    by_status={}
    for r in rows: by_status[r['status']]=by_status.get(r['status'],0)+1
    print(json.dumps({'output_tsv':args.output_tsv,'total':len(rows),'pass':sum(r['status']=='PASS' for r in rows),'by_status':by_status}, indent=2))
if __name__=='__main__': main()

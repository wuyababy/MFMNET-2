#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit the ORIGINAL DeepDDS handling of the 22RV1 cell line.
No model loading, no MolFormer, no inference, no tuning.

The script:
1) extracts the exact code around independent_cell_features_954.csv in
   load_data.py and creat_data_DC.py;
2) searches the entire repository for 22RV1 / 924100;
3) inspects the first-column identifiers of independent_cell_features_954.csv;
4) checks how many of the 668 rows use 22RV1;
5) reports whether the official prediction files contain all 668 rows.

Run this before changing any cell feature mapping.
"""

from __future__ import annotations
import argparse, json, re, shutil, tempfile, zipfile
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deepdds-repo", required=True)
    p.add_argument("--output", default="./result/deepdds_22rv1_audit")
    return p.parse_args()


def open_repo(path):
    p = Path(path)
    if p.is_dir():
        return p, None
    td = tempfile.mkdtemp(prefix="deepdds_22rv1_")
    root = Path(td)
    with zipfile.ZipFile(p) as z:
        z.extractall(root)
    return root, td


def locate(root, suffix):
    hits = [p for p in root.rglob("*") if p.is_file()
            and str(p).replace("\\","/").endswith(suffix)]
    if not hits:
        return None
    return hits[0]


def read_text(path):
    for enc in ("utf-8","utf-8-sig","gb18030","latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def code_context(path, patterns, radius=35):
    text = read_text(path)
    lines = text.splitlines()
    hit_lines = set()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(p.lower() in low for p in patterns):
            hit_lines.add(i)
    if not hit_lines:
        return ""
    keep = set()
    for i in hit_lines:
        keep.update(range(max(0,i-radius), min(len(lines),i+radius+1)))
    out=[]
    prev=None
    for i in sorted(keep):
        if prev is not None and i != prev+1:
            out.append("\n... omitted ...\n")
        out.append(f"{i+1:04d}: {lines[i]}")
        prev=i
    return "\n".join(out)


def main():
    args=parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    root,tmp=open_repo(args.deepdds_repo)

    try:
        load=locate(root,"load_data.py")
        create=locate(root,"creat_data_DC.py")
        feat=locate(root,"data/independent_set/independent_cell_features_954.csv")
        inp=locate(root,"data/independent_set/independent_input.csv")
        cellmeta=locate(root,"data/independent_set/independent_cell.csv")
        input_names=locate(root,"data/independent_set/independent_input_drug_cell.csv")

        for name,p in [("load_data.py",load),("creat_data_DC.py",create)]:
            if p:
                ctx=code_context(
                    p,
                    ["independent_cell_features_954","cell_file","cellfile",
                     "cell_feature","cell_features","independent_input"]
                )
                (out/f"{name}.relevant.txt").write_text(ctx,encoding="utf-8")

        # Full repo literal search.
        hits=[]
        text_ext={".py",".txt",".csv",".md",".ipynb",".json",".tsv"}
        pats=["22RV1","22Rv1","22rv1","924100"]
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in text_ext:
                continue
            try:
                txt=read_text(p)
            except Exception:
                continue
            for line_no,line in enumerate(txt.splitlines(),1):
                if any(x.lower() in line.lower() for x in pats):
                    hits.append({
                        "path":str(p.relative_to(root)).replace("\\","/"),
                        "line":line_no,
                        "text":line[:2000],
                    })
        pd.DataFrame(hits).to_csv(out/"repo_22rv1_924100_hits.csv",index=False)

        # Feature first-column diagnostics.
        firstcol_diag=[]
        feat_shape=None
        if feat:
            df=pd.read_csv(feat,low_memory=False)
            feat_shape=[int(df.shape[0]),int(df.shape[1])]
            ids=df.iloc[:,0].astype(str)
            for i,x in enumerate(ids):
                xl=x.upper()
                # Diagnostic only: broad strings are NOT used as mappings.
                if any(k in xl for k in ["22","RV","924","PROSTATE"]):
                    firstcol_diag.append({
                        "row_index":i,
                        "identifier":x,
                    })
            pd.DataFrame(firstcol_diag).to_csv(
                out/"feature_first_column_diagnostic_candidates.csv",index=False
            )

        # Exact 668 involvement.
        exact={}
        if inp:
            d=pd.read_csv(inp,low_memory=False)
            rows=d[d["cell"].astype(str).str.upper()=="22RV1"].copy()
            rows.to_csv(out/"independent_input_22rv1_rows.csv",index=False)
            exact={
                "independent_input_rows":int(len(d)),
                "rows_with_22RV1":int(len(rows)),
                "22RV1_positive":int((pd.to_numeric(rows["label"],errors="coerce")==1).sum()),
                "22RV1_negative":int((pd.to_numeric(rows["label"],errors="coerce")==0).sum()),
            }

        if cellmeta:
            cm=pd.read_csv(cellmeta,low_memory=False)
            c=cm[cm["Cell line name"].astype(str).str.upper()=="22RV1"].copy()
            c.to_csv(out/"independent_cell_22rv1_metadata.csv",index=False)

        if input_names:
            try:
                ndf=pd.read_csv(input_names,header=None,low_memory=False)
                mask=ndf.astype(str).apply(
                    lambda col: col.str.upper().eq("22RV1")
                ).any(axis=1)
                ndf[mask].to_csv(out/"independent_input_drug_cell_22rv1_rows.csv",
                                 index=False,header=False)
            except Exception as e:
                (out/"input_drug_cell_error.txt").write_text(repr(e),encoding="utf-8")

        pred_rows={}
        pred_dir=locate(root,"data/result/independent_set_predict/Independent_GATNet--result_new_labels_0_10.csv")
        if pred_dir:
            for p in pred_dir.parent.glob("*.csv"):
                try:
                    dfp=pd.read_csv(p,low_memory=False)
                    pred_rows[p.name]={
                        "header_inferred_rows":int(len(dfp)),
                        "cols":int(dfp.shape[1]),
                    }
                    # Also check headerless in case first prediction was consumed.
                    dfp0=pd.read_csv(p,header=None,low_memory=False)
                    pred_rows[p.name]["headerless_rows"]=int(len(dfp0))
                except Exception as e:
                    pred_rows[p.name]={"error":repr(e)}

        summary={
            "feature_file_shape":feat_shape,
            "literal_repo_hits":len(hits),
            "first_column_diagnostic_candidate_count":len(firstcol_diag),
            **exact,
            "prediction_file_row_counts":pred_rows,
            "note":"No mapping or model inference was performed."
        }
        (out/"deepdds_22rv1_audit_summary.json").write_text(
            json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
        )

        print(json.dumps(summary,indent=2,ensure_ascii=False))
        print("\nRelevant original code written to:")
        print(out/"load_data.py.relevant.txt")
        print(out/"creat_data_DC.py.relevant.txt")
        print("\nUpload the entire output folder:", out.resolve())

    finally:
        if tmp:
            shutil.rmtree(tmp,ignore_errors=True)


if __name__=="__main__":
    main()

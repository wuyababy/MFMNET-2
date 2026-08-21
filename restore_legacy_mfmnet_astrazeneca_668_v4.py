#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Restore the legacy MFMNet AstraZeneca 668-sample inference path.

This script is designed specifically to verify, not force, the manuscript's
legacy "93.3% accuracy on AstraZeneca's independent set" statement.

Workflow
--------
1. Load the exact 668-row independent_input.csv from the official DeepDDS repo.
2. Reconstruct the official 954-dimensional cell-line features.
3. Audit whether the supplied MFMNet 768-d drug embeddings are consistent with
   IBM MoLFormer pooler embeddings, using ONLY the O'Neil training feature files.
4. If the provenance audit passes, generate AstraZeneca embeddings with the
   source-matched MolFormer procedure.
5. Load the legacy project checkpoint result/models/Enhanced_best.pt and run
   the exact 668 records once, with no external tuning.
6. Report the actual accuracy and whether it rounds to 93.3%.

Important
---------
- No external labels are used to select the MolFormer input representation.
- No external labels are used to tune a threshold, seed, checkpoint, or ensemble.
- The primary MolFormer seed is fixed at 0 before external inference.
- If the embedding-provenance audit fails, model inference is stopped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn import metrics
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem


DRUG_DIM = 768
CELL_DIM = 954
ATOM_DIM = 78
MAX_ATOMS = 100


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--codes-root", default=".",
                   help="MFMNet codes directory containing enhanced_model.py, data/, result/")
    p.add_argument("--deepdds-repo", required=True,
                   help="Official DeepDDS repository ZIP or extracted directory")
    p.add_argument("--molformer-model",
                   default="ibm-research/MoLFormer-XL-both-10pct")
    p.add_argument("--molformer-revision", default="",
                   help='Use "compat-v4" if needed for Transformers v4')
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--provenance-min-mean-cosine", type=float, default=0.90)
    p.add_argument("--provenance-min-top1-rate", type=float, default=0.80)
    p.add_argument("--output", default="./result/astra_668_legacy_mfmnet_restore")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def choose_device(req):
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if req == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU requested but unavailable")
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def norm_token(x):
    s = str(x).strip().upper()
    s = s.replace("–", "-").replace("—", "-").replace("_", "-")
    return re.sub(r"[^A-Z0-9]+", "", s)


def canonical_nonisomeric_smiles(s):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {s!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def locate(root, suffix):
    hits = [
        p for p in root.rglob("*")
        if p.is_file() and str(p).replace("\\", "/").endswith(suffix)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one {suffix}, found {len(hits)}: {hits[:10]}")
    return hits[0]


def open_repo(path):
    path = Path(path)
    if path.is_dir():
        return path, None
    if path.is_file() and path.suffix.lower() == ".zip":
        td = tempfile.mkdtemp(prefix="deepdds_restore_")
        root = Path(td)
        with zipfile.ZipFile(path) as z:
            z.extractall(root)
        return root, td
    raise ValueError("--deepdds-repo must be a ZIP or extracted directory")


def cosine(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def build_empirical_embedding_centroids(ddi, emb):
    left = emb[:, :DRUG_DIM]
    right = emb[:, DRUG_DIM:]
    store = defaultdict(list)

    for i, s in enumerate(ddi["drug1"].astype(str)):
        store[s].append(left[i])
    for i, s in enumerate(ddi["drug2"].astype(str)):
        store[s].append(right[i])

    centroids = {}
    within_rows = []
    for s, vals in store.items():
        arr = np.stack(vals).astype(np.float64)
        # Unit-normalized centroid is robust to scale variation.
        unit = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        c = unit.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-12
        centroids[s] = c

        sims = unit @ c
        within_rows.append({
            "smiles": s,
            "n_occurrences": len(arr),
            "mean_cosine_to_empirical_centroid": float(sims.mean()),
            "min_cosine_to_empirical_centroid": float(sims.min()),
            "max_cosine_to_empirical_centroid": float(sims.max()),
        })
    return centroids, pd.DataFrame(within_rows)


def infer_embedding_batch_artifact(ddi, emb):
    rows = []
    for side, scol, arr in [
        ("left", "drug1", emb[:, :DRUG_DIM]),
        ("right", "drug2", emb[:, DRUG_DIM:]),
    ]:
        groups = defaultdict(list)
        for i, (s, v) in enumerate(zip(ddi[scol].astype(str), arr)):
            groups[(s, np.asarray(v, np.float32).tobytes())].append(i)

        dup = [idx for idx in groups.values() if len(idx) > 1]
        for bs in [8, 16, 32, 64, 128, 256]:
            same = sum(len(set(i // bs for i in idx)) == 1 for idx in dup)
            rows.append({
                "side": side,
                "candidate_batch_size": bs,
                "duplicate_vector_groups": len(dup),
                "groups_fully_within_one_batch": same,
                "fraction_fully_within_one_batch": (
                    same / len(dup) if dup else np.nan
                ),
            })
    return pd.DataFrame(rows)


def load_molformer(model_id, revision, device, deterministic_eval=False):
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Transformers is required. Install it first, e.g. "
            '`pip install -U transformers safetensors huggingface_hub`'
        ) from e

    kw = {"trust_remote_code": True}
    if revision:
        kw["revision"] = revision

    tok = AutoTokenizer.from_pretrained(model_id, **kw)

    model_kw = dict(kw)
    model_kw["deterministic_eval"] = deterministic_eval
    model = AutoModel.from_pretrained(model_id, **model_kw)
    model.to(device)
    model.eval()
    return tok, model


def molformer_pooler(smiles_list, tokenizer, model, device, batch_size, seed):
    set_seed(seed)
    outs = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start:start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        if not hasattr(out, "pooler_output") or out.pooler_output is None:
            raise RuntimeError("Loaded MolFormer did not return pooler_output")
        x = out.pooler_output.detach().cpu().numpy().astype(np.float32)
        if x.shape[1] != DRUG_DIM:
            raise RuntimeError(
                f"Expected MolFormer pooler dimension {DRUG_DIM}, got {x.shape}"
            )
        outs.append(x)
    return np.concatenate(outs, axis=0)


def provenance_score(smiles, generated, centroids, mode, repeat_seed):
    keys = list(centroids.keys())
    C = np.stack([centroids[k] for k in keys])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    rows = []
    top1 = 0
    correct_sims = []

    for s, v in zip(smiles, generated):
        u = v.astype(np.float64)
        u /= np.linalg.norm(u) + 1e-12
        sims = C @ u
        order = np.argsort(-sims)
        pred = keys[int(order[0])]
        true_i = keys.index(s)
        true_sim = float(sims[true_i])
        second_best = float(np.max(np.delete(sims, true_i))) if len(sims) > 1 else np.nan
        correct = pred == s
        top1 += int(correct)
        correct_sims.append(true_sim)
        rows.append({
            "mode": mode,
            "repeat_seed": repeat_seed,
            "smiles": s,
            "true_centroid_cosine": true_sim,
            "best_other_centroid_cosine": second_best,
            "margin_true_minus_best_other": true_sim - second_best,
            "nearest_empirical_smiles": pred,
            "top1_correct": correct,
        })

    summary = {
        "mode": mode,
        "repeat_seed": repeat_seed,
        "n_smiles": len(smiles),
        "mean_true_centroid_cosine": float(np.mean(correct_sims)),
        "median_true_centroid_cosine": float(np.median(correct_sims)),
        "min_true_centroid_cosine": float(np.min(correct_sims)),
        "top1_identification_rate": float(top1 / len(smiles)),
        "median_margin": float(pd.DataFrame(rows)["margin_true_minus_best_other"].median()),
    }
    return pd.DataFrame(rows), summary


def _cell_identifier_base_token(x: str) -> str:
    """
    Return a conservative source identifier token before common tissue suffixes.
    We do not use arbitrary string-prefix matching because SW48 must not match SW480.
    """
    s = str(x).strip()
    first = re.split(r"[_\s]+", s, maxsplit=1)[0]
    return norm_token(first)


def _identifier_tokens(x: str):
    """
    Conservative tokenization for source identifiers.  Keeps the full normalized
    identifier and delimiter-separated tokens so that a COSMIC ID such as 924100
    can be matched when it is embedded in a longer source row identifier.
    """
    s = str(x).strip()
    toks = {norm_token(s)}
    for part in re.split(r"[_\s|;,:/\\]+", s):
        n = norm_token(part)
        if n:
            toks.add(n)
    return toks


def _load_independent_cell_aliases(cell_meta_path, needed_cells):
    meta = pd.read_csv(cell_meta_path, low_memory=False)
    if "Cell line name" not in meta.columns:
        raise ValueError(
            f"independent_cell.csv missing 'Cell line name': {list(meta.columns)}"
        )

    # Only source metadata fields are used. No labels/predictions are involved.
    name_norm_to_row = {
        norm_token(r["Cell line name"]): r
        for _, r in meta.iterrows()
        if not pd.isna(r["Cell line name"])
    }

    alias_map = {}
    alias_audit = []
    for cell in needed_cells:
        cn = norm_token(cell)
        r = name_norm_to_row.get(cn)

        aliases = {cn}
        cosmic = ""
        if r is not None:
            if "COSMIC ID" in meta.columns and not pd.isna(r.get("COSMIC ID")):
                cosmic = str(r.get("COSMIC ID")).strip()
                # Excel/CSV sometimes renders integer IDs as 924100.0
                cosmic = re.sub(r"\.0$", "", cosmic)
                if cosmic:
                    aliases.add(norm_token(cosmic))

        alias_map[cell] = aliases
        alias_audit.append({
            "cell": cell,
            "cell_normalized": cn,
            "cosmic_id": cosmic,
            "source_aliases": " | ".join(sorted(aliases)),
            "found_in_independent_cell_metadata": r is not None,
        })

    return alias_map, pd.DataFrame(alias_audit)


def _try_cell_matrix_interpretation(df, exact_input, header_mode, alias_map):
    if df.shape[1] != CELL_DIM + 1:
        return {}, pd.DataFrame(), {
            "header_mode": header_mode,
            "usable": False,
            "reason": f"expected 955 columns, got {df.shape[1]}"
        }

    ids = df.iloc[:, 0].astype(str)
    row_norms = [norm_token(x) for x in ids]
    base_norms = [_cell_identifier_base_token(x) for x in ids]
    row_tokens = [_identifier_tokens(x) for x in ids]

    needed = sorted(set(exact_input["cell"].astype(str)))
    inventory = {}
    audit = []

    for cell in needed:
        cn = norm_token(cell)
        aliases = set(alias_map.get(cell, {cn}))

        # Rule 1: exact full identifier by cell name.
        exact_name_hits = [i for i, rn in enumerate(row_norms) if rn == cn]

        # Rule 2: exact source token before tissue suffix by cell name.
        base_name_hits = [i for i, bn in enumerate(base_norms) if bn == cn]

        # Rule 3: official COSMIC ID from independent_cell.csv, matched only
        # as an exact full identifier or delimiter-separated identifier token.
        cosmic_aliases = aliases - {cn}
        cosmic_hits = [
            i for i, toks in enumerate(row_tokens)
            if cosmic_aliases and bool(toks.intersection(cosmic_aliases))
        ]

        if len(exact_name_hits) == 1:
            hits = exact_name_hits
            match_rule = "normalized_exact_cell_name"
        elif len(exact_name_hits) == 0 and len(base_name_hits) == 1:
            hits = base_name_hits
            match_rule = "exact_cell_token_before_tissue_suffix"
        elif (
            len(exact_name_hits) == 0
            and len(base_name_hits) == 0
            and len(cosmic_hits) == 1
        ):
            hits = cosmic_hits
            match_rule = "official_cosmic_id_token"
        else:
            # Retain all source-defined candidates for diagnostics only.
            hits = sorted(set(exact_name_hits + base_name_hits + cosmic_hits))
            match_rule = "unresolved_or_ambiguous"

        finite = False
        matched_identifier = ""
        candidate_ids = " | ".join(str(ids.iloc[i]) for i in hits[:20])

        if len(hits) == 1:
            i = hits[0]
            vals = pd.to_numeric(df.iloc[i, 1:], errors="coerce").to_numpy(np.float32)
            finite = bool(len(vals) == CELL_DIM and np.isfinite(vals).all())
            matched_identifier = str(df.iloc[i, 0])
            if finite:
                inventory[cell] = vals

        audit.append({
            "header_mode": header_mode,
            "cell": cell,
            "source_aliases": " | ".join(sorted(aliases)),
            "n_matches": len(hits),
            "match_rule": match_rule,
            "matched_row_identifier": matched_identifier,
            "candidate_row_identifiers": candidate_ids,
            "finite_954": finite,
        })

    audit_df = pd.DataFrame(audit)
    summary = {
        "header_mode": header_mode,
        "usable": True,
        "resolved_finite_cells": int(audit_df["finite_954"].sum()),
        "needed_cells": int(len(needed)),
        "ambiguous_or_missing": int((audit_df["n_matches"] != 1).sum()),
    }
    return inventory, audit_df, summary


def extract_external_cells(cell954_path, cell_meta_path, exact_input):
    """
    Reconstruct the official 954-d AstraZeneca cell-line features.

    Source identifiers are resolved with two official DeepDDS files:
      * independent_input.csv: requested challenge cell names
      * independent_cell.csv: source cell name -> COSMIC ID metadata

    The 954-d feature file is tested with both header interpretations because
    its first rows contain metadata.  The chosen interpretation is the one that
    resolves the most requested cells using identifiers/features only.
    No external labels or model predictions are used.
    """
    needed = sorted(set(exact_input["cell"].astype(str)))
    alias_map, alias_audit = _load_independent_cell_aliases(
        cell_meta_path, needed
    )

    interpretations = []
    for header_mode, header in [("header_inferred", 0), ("headerless", None)]:
        try:
            df = pd.read_csv(cell954_path, header=header, low_memory=False)
            inventory, audit, summary = _try_cell_matrix_interpretation(
                df, exact_input, header_mode, alias_map
            )
            interpretations.append((inventory, audit, summary))
        except Exception as e:
            interpretations.append((
                {},
                pd.DataFrame(),
                {
                    "header_mode": header_mode,
                    "usable": False,
                    "reason": repr(e),
                    "resolved_finite_cells": 0,
                    "needed_cells": len(needed),
                    "ambiguous_or_missing": len(needed),
                }
            ))

    interpretations.sort(
        key=lambda x: (
            x[2].get("resolved_finite_cells", 0),
            -x[2].get("ambiguous_or_missing", 999),
        ),
        reverse=True,
    )
    best_inventory, best_audit, best_summary = interpretations[0]

    all_audits = []
    for _, adf, summ in interpretations:
        if len(adf):
            tmp = adf.copy()
            tmp["interpretation_resolved_finite_cells"] = summ.get(
                "resolved_finite_cells", 0
            )
            tmp["selected_interpretation"] = (
                summ.get("header_mode") == best_summary.get("header_mode")
            )
            all_audits.append(tmp)

    combined = (
        pd.concat(all_audits, ignore_index=True)
        if all_audits else pd.DataFrame()
    )

    print("\nCell-feature source-format audit:")
    for _, _, summ in interpretations:
        print(json.dumps(summ, ensure_ascii=False))

    print(
        "Selected cell-feature interpretation:",
        best_summary.get("header_mode"),
        f"({best_summary.get('resolved_finite_cells', 0)}/24 finite cells)"
    )

    # Return alias audit too so the exact 22RV1 -> COSMIC mapping is preserved.
    return best_inventory, combined, alias_audit

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"{x} not in allowable set")
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    feature = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
             'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
             'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
             'Pb', 'Unknown']
        )
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )
    arr = np.asarray(feature, np.float32)
    arr /= float(arr.sum())
    return arr


def smile_to_legacy_graph(smile):
    mol = Chem.MolFromSmiles(str(smile))
    if mol is None:
        raise ValueError(f"Cannot parse {smile!r}")
    if mol.GetNumAtoms() > MAX_ATOMS:
        raise ValueError(f"Molecule > {MAX_ATOMS} atoms: {smile!r}")

    x = torch.zeros((MAX_ATOMS, ATOM_DIM), dtype=torch.float32)
    mask = [1] * MAX_ATOMS
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = torch.from_numpy(atom_features(atom))
        mask[i] = 0

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.mask = torch.tensor([bool(v) for v in mask])
    return data


def build_external_graph_lists(inp, left_emb, right_emb, cell_inventory):
    left_list, right_list = [], []
    graph_cache = {}

    all_smiles = sorted(set(inp["drug1"].astype(str)) | set(inp["drug2"].astype(str)))
    for s in all_smiles:
        graph_cache[s] = smile_to_legacy_graph(s)

    for i, r in inp.reset_index(drop=True).iterrows():
        cell = str(r["cell"])
        if cell not in cell_inventory:
            raise RuntimeError(f"No 954-d cell feature for {cell}")

        # Clone graph tensors so PyG batching remains independent.
        lg0 = graph_cache[str(r["drug1"])]
        rg0 = graph_cache[str(r["drug2"])]

        lg = Data(x=lg0.x.clone(), edge_index=lg0.edge_index.clone())
        lg.mask = lg0.mask.clone()
        lg.y = torch.tensor(int(r["label"]))
        lg.smiles_embedding = torch.tensor(left_emb[i][None, :], dtype=torch.float32)
        lg.ccle_embedding = torch.tensor(cell_inventory[cell][None, :], dtype=torch.float32)

        rg = Data(x=rg0.x.clone(), edge_index=rg0.edge_index.clone())
        rg.mask = rg0.mask.clone()
        rg.y = torch.tensor(int(r["label"]))
        rg.smiles_embedding = torch.tensor(right_emb[i][None, :], dtype=torch.float32)

        left_list.append(lg)
        right_list.append(rg)

    return left_list, right_list


def import_legacy_model(codes_root):
    codes_root = Path(codes_root).resolve()
    if str(codes_root) not in sys.path:
        sys.path.insert(0, str(codes_root))
    import enhanced_model
    return enhanced_model


def predict_legacy(model, device, left_list, right_list, linc, batch_size):
    lloader = DataLoader(left_list, batch_size=batch_size, shuffle=False)
    rloader = DataLoader(right_list, batch_size=batch_size, shuffle=False)

    ys, scores = [], []
    model.eval()
    with torch.no_grad():
        for dl, dr in zip(lloader, rloader):
            dl = dl.to(device)
            dr = dr.to(device)
            logits, _ = model(dl, dr, linc.to(device))
            prob = F.softmax(logits, dim=1)[:, 1]
            ys.append(dl.y.view(-1).detach().cpu().numpy())
            scores.append(prob.detach().cpu().numpy())
    return np.concatenate(ys).astype(int), np.concatenate(scores)


def main():
    args = parse_args()
    codes_root = Path(args.codes_root).resolve()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    ddi_path = codes_root / "data/raw/all_ddi.csv"
    emb_path = codes_root / "data/raw/all_drug_embedding.csv"
    checkpoint = codes_root / "result/models/Enhanced_best.pt"
    gene_vector = codes_root / "data/raw/gene_vector.csv"
    enhanced_model_path = codes_root / "enhanced_model.py"

    for p in [ddi_path, emb_path, checkpoint, gene_vector, enhanced_model_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    repo_root, temp_repo = open_repo(args.deepdds_repo)

    try:
        exact_path = locate(repo_root, "data/independent_set/independent_input.csv")
        cell954_path = locate(
            repo_root, "data/independent_set/independent_cell_features_954.csv"
        )
        drug_meta_path = locate(repo_root, "data/independent_set/independent_drug.csv")
        cell_meta_path = locate(repo_root, "data/independent_set/independent_cell.csv")

        inp = pd.read_csv(exact_path, low_memory=False)
        if len(inp) != 668:
            raise RuntimeError(f"Expected exact 668 rows, got {len(inp)}")
        if not {"drug1", "drug2", "cell", "label"}.issubset(inp.columns):
            raise RuntimeError("Unexpected independent_input.csv schema")
        inp["label"] = pd.to_numeric(inp["label"], errors="raise").astype(int)

        # Save exact-source diagnostics.
        exact_summary = {
            "rows": int(len(inp)),
            "positive": int((inp["label"] == 1).sum()),
            "negative": int((inp["label"] == 0).sum()),
            "unique_input_smiles": int(
                len(set(inp["drug1"].astype(str)) | set(inp["drug2"].astype(str)))
            ),
            "unique_cells": int(inp["cell"].astype(str).nunique()),
            "metadata_drug_rows": int(len(pd.read_csv(drug_meta_path))),
            "metadata_cell_rows": int(len(pd.read_csv(cell_meta_path))),
        }

        # ------------------------------------------------------------------
        # O'Neil embedding provenance audit
        # ------------------------------------------------------------------
        ddi = pd.read_csv(ddi_path, low_memory=False)
        emb_df = pd.read_csv(emb_path, low_memory=False)
        emb = emb_df.to_numpy(np.float32)
        if len(ddi) != len(emb) or emb.shape[1] != 1536:
            raise RuntimeError("MFMNet all_ddi/all_drug_embedding mismatch")

        centroids, within = build_empirical_embedding_centroids(ddi, emb)
        within.to_csv(outdir / "oneil_empirical_embedding_within_smiles.csv", index=False)

        batch_artifact = infer_embedding_batch_artifact(ddi, emb)
        batch_artifact.to_csv(
            outdir / "oneil_embedding_batch_artifact_audit.csv", index=False
        )

        train_smiles = sorted(centroids)
        device = choose_device(args.device)
        print("Using device:", device)

        print("Loading MolFormer. First run may download model files...")
        tokenizer, molformer = load_molformer(
            args.molformer_model, args.molformer_revision,
            device=device, deterministic_eval=False
        )

        candidate_inputs = {
            "raw_smiles": train_smiles,
            "rdkit_canonical_nonisomeric": [
                canonical_nonisomeric_smiles(x) for x in train_smiles
            ],
        }

        detail_parts = []
        provenance_summaries = []

        # Three fixed repeats are used only to characterize feature provenance
        # on the O'Neil training inputs, never to select based on external labels.
        for mode, smi in candidate_inputs.items():
            for repeat_seed in [0, 1, 2]:
                generated = molformer_pooler(
                    smi, tokenizer, molformer, device,
                    batch_size=min(args.batch_size, len(smi)),
                    seed=repeat_seed
                )
                detail, summ = provenance_score(
                    train_smiles, generated, centroids, mode, repeat_seed
                )
                detail_parts.append(detail)
                provenance_summaries.append(summ)

        detail_df = pd.concat(detail_parts, ignore_index=True)
        prov_df = pd.DataFrame(provenance_summaries)
        detail_df.to_csv(outdir / "molformer_provenance_per_smiles.csv", index=False)
        prov_df.to_csv(outdir / "molformer_provenance_summary.csv", index=False)

        # Choose representation solely by mean training-feature agreement.
        mode_agg = (
            prov_df.groupby("mode", as_index=False)
            .agg(
                mean_true_centroid_cosine=("mean_true_centroid_cosine", "mean"),
                top1_identification_rate=("top1_identification_rate", "mean"),
                median_margin=("median_margin", "median"),
            )
            .sort_values(
                ["mean_true_centroid_cosine", "top1_identification_rate"],
                ascending=False
            )
            .reset_index(drop=True)
        )
        mode_agg.to_csv(outdir / "molformer_mode_selection_training_only.csv", index=False)
        best = mode_agg.iloc[0]
        best_mode = str(best["mode"])

        provenance_pass = bool(
            best["mean_true_centroid_cosine"] >= args.provenance_min_mean_cosine
            and best["top1_identification_rate"] >= args.provenance_min_top1_rate
        )

        audit_summary = {
            "exact_astrazeneca": exact_summary,
            "mfmnet_training_unique_smiles": int(len(train_smiles)),
            "best_molformer_input_mode_selected_from_training_features_only": best_mode,
            "best_mean_true_centroid_cosine": float(best["mean_true_centroid_cosine"]),
            "best_top1_identification_rate": float(best["top1_identification_rate"]),
            "best_median_margin": float(best["median_margin"]),
            "provenance_min_mean_cosine": args.provenance_min_mean_cosine,
            "provenance_min_top1_rate": args.provenance_min_top1_rate,
            "provenance_pass": provenance_pass,
            "primary_external_embedding_seed": args.seed,
            "external_labels_used_for_embedding_selection": False,
        }
        (outdir / "pre_inference_provenance_summary.json").write_text(
            json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(json.dumps(audit_summary, indent=2, ensure_ascii=False))

        if not provenance_pass:
            print(
                "\nSTOP: MolFormer provenance did not pass the pre-specified "
                "training-feature audit. No AstraZeneca model inference was performed."
            )
            return

        # ------------------------------------------------------------------
        # Build official 954-d external cell inputs.
        # ------------------------------------------------------------------
        cell_inventory, cell_audit, cell_alias_audit = extract_external_cells(
            cell954_path, cell_meta_path, inp
        )
        cell_audit.to_csv(outdir / "astra_668_cell_feature_audit.csv", index=False)
        cell_alias_audit.to_csv(
            outdir / "astra_668_cell_source_alias_audit.csv", index=False
        )

        selected_audit = cell_audit[
            cell_audit["selected_interpretation"] == True
        ].copy() if len(cell_audit) else cell_audit

        if (
            len(cell_inventory) != inp["cell"].nunique()
            or len(selected_audit) != inp["cell"].nunique()
            or not selected_audit["finite_954"].all()
        ):
            unresolved = selected_audit.loc[
                ~selected_audit["finite_954"],
                [
                    "cell", "source_aliases", "n_matches",
                    "match_rule", "candidate_row_identifiers"
                ]
            ]
            print("\nUnresolved/ambiguous cell-feature rows in selected interpretation:")
            print(unresolved.to_string(index=False))
            raise RuntimeError(
                "Could not reconstruct all 24 official 954-d cell-line vectors. "
                "See astra_668_cell_feature_audit.csv"
            )

        # ------------------------------------------------------------------
        # Generate external MolFormer embeddings once with fixed seed 0.
        # The training file shows a batch-structured stochastic signature; we
        # preserve sequential batch_size=64 generation rather than constructing
        # one static vector per drug.
        # ------------------------------------------------------------------
        raw_left = inp["drug1"].astype(str).tolist()
        raw_right = inp["drug2"].astype(str).tolist()

        if best_mode == "rdkit_canonical_nonisomeric":
            ext_left_smi = [canonical_nonisomeric_smiles(x) for x in raw_left]
            ext_right_smi = [canonical_nonisomeric_smiles(x) for x in raw_right]
        else:
            ext_left_smi = raw_left
            ext_right_smi = raw_right

        set_seed(args.seed)
        left_external = molformer_pooler(
            ext_left_smi, tokenizer, molformer, device,
            batch_size=args.batch_size, seed=args.seed
        )
        # Continue with a separately fixed but deterministic stream for drug2.
        # We use seed+1 rather than selecting a seed by external performance.
        right_external = molformer_pooler(
            ext_right_smi, tokenizer, molformer, device,
            batch_size=args.batch_size, seed=args.seed + 1
        )

        pd.DataFrame(left_external).to_csv(
            outdir / "astra_668_molformer_drug1_768.csv", index=False
        )
        pd.DataFrame(right_external).to_csv(
            outdir / "astra_668_molformer_drug2_768.csv", index=False
        )

        # ------------------------------------------------------------------
        # Restore legacy Enhanced_best.pt inference.
        # ------------------------------------------------------------------
        legacy = import_legacy_model(codes_root)
        # enhanced_model.py stores a module-global device used during forward.
        legacy.device = device

        model = legacy.EnhancedMDSyn().to(device)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()

        row_lincs = pd.read_csv(gene_vector, index_col=0, header=None)
        linc = torch.tensor(row_lincs.to_numpy(), dtype=torch.float32)

        left_list, right_list = build_external_graph_lists(
            inp, left_external, right_external, cell_inventory
        )

        y, score = predict_legacy(
            model, device, left_list, right_list, linc, args.batch_size
        )
        pred = (score > 0.5).astype(int)

        precision_curve, recall_curve, _ = metrics.precision_recall_curve(y, score)
        pr_auc = float(metrics.auc(recall_curve, precision_curve))
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

        result = {
            "dataset": "DeepDDS official AstraZeneca independent_input.csv",
            "n": int(len(y)),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "checkpoint": str(checkpoint),
            "checkpoint_type": "legacy supplied Enhanced_best.pt",
            "molformer_model": args.molformer_model,
            "molformer_input_mode": best_mode,
            "molformer_external_seed_drug1": int(args.seed),
            "molformer_external_seed_drug2": int(args.seed + 1),
            "molformer_batch_size": int(args.batch_size),
            "external_tuning": False,
            "threshold": 0.5,
            "correct": int((pred == y).sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "accuracy_percent": float(100 * accuracy_score(y, pred)),
            "accuracy_percent_one_decimal": float(round(100 * accuracy_score(y, pred), 1)),
            "rounds_to_93_3_percent": bool(
                round(100 * accuracy_score(y, pred), 1) == 93.3
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "AUROC": float(roc_auc_score(y, score)),
            "PR_AUC": pr_auc,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "kappa": float(cohen_kappa_score(y, pred)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        }

        pred_df = inp[["drug1", "drug2", "cell", "label"]].copy()
        pred_df["score"] = score
        pred_df["predicted_label"] = pred
        pred_df["correct"] = pred_df["label"].to_numpy() == pred
        pred_df.to_csv(outdir / "astra_668_legacy_mfmnet_predictions.csv", index=False)

        (outdir / "astra_668_legacy_mfmnet_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\nFINAL LEGACY ASTRAZENECA RESULT")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\nUpload the entire output folder:")
        print(outdir.resolve())

    finally:
        if temp_repo:
            shutil.rmtree(temp_repo, ignore_errors=True)


if __name__ == "__main__":
    main()

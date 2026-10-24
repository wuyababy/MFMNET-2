"""
make_fig2_atomic_similarity.py — atomic-feature similarity heatmaps
(before vs. after training, VINBLASTINE / L778123)

What it does
------------
1. Look up SMILES for the two drugs in the DrugComb identifier table
   (DrugComb_drug_identifiers.xlsx) and build the molecular
   graphs with the same pipeline used during training (loader.smile_to_graph,
   100-node padded graphs + real-atom mask).
2. Build MFMNet (variant="full") twice:
   - "before training": the same architecture randomly initialized with seed 0;
   - "after training":  loaded from a released checkpoint
     (checkpoints/mfmnet_full_foldN.pt).
3. Extract the 128-d fused atom representation (GCN/GIN/GAT branches +
   graph_fusion) for every real atom (mask == 0) and compute atom-atom
   cosine similarity matrices: cross (A x B) and self (A x A, B x B).
4. Plot a 3 x 2 panel figure (rows: A x B, B x B, A x A; columns:
   before/after), one shared colorbar per row, and save
   300 dpi LZW TIFF + vector PDF.

Usage
-----
python figures/make_fig2_atomic_similarity.py \
    --codes-dir  . \
    --checkpoint checkpoints/mfmnet_full_fold1.pt \
    --xlsx       DrugComb_drug_identifiers.xlsx \
    --outdir     figures/output/fig2

Display ranges (row 1: 0-1, rows 2-3: -0.5 to 1.0) are fixed for
comparability across panels; underlying similarity values are not modified.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
TICK_EVERY = 10         # sample ticks: 29-59 full ticks would overlap in-panel
CB_TICKS = [-0.5, 0.0, 0.5, 1.0]


def lookup_smiles(xlsx_path: str, drug_name: str) -> str:
    """Look up the SMILES of a drug name (case-insensitive) in the identifier table."""
    df = pd.read_excel(xlsx_path)
    hit = df[df["dname"].astype(str).str.upper() == drug_name.upper()]
    if hit.empty:
        raise SystemExit(
            f"Drug {drug_name!r} not found in {xlsx_path}. "
            f"Available dname examples: {df['dname'].head(10).tolist()} ... "
            f"Alternatively pass --smiles1/--smiles2 directly."
        )
    return str(hit.iloc[0]["smiles"])


def molecule_graph(smiles: str, codes_dir: str):
    """Build the padded molecular graph with the training pipeline (smile_to_graph)."""
    sys.path.insert(0, str(codes_dir))
    from loader import smile_to_graph  # noqa: PLC0415 (lazy import; needs rdkit)

    data, mask = smile_to_graph(smiles)
    if data is None:
        raise SystemExit(f"Failed to parse SMILES: {smiles!r}")
    real_idx = np.array([i for i, m in enumerate(mask) if m == 0], dtype=int)
    return data, real_idx


def load_model(variant: str, checkpoint_path, codes_dir: str, seed: int = 0):
    """Build the model; checkpoint_path=None returns an untrained (random init) model."""
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(codes_dir))
    from MFMNet_reviewer_clean_save_ckpt import MFMNetReviewerModel  # noqa: PLC0415

    torch.manual_seed(seed)
    model = MFMNetReviewerModel(variant=variant)
    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    model.eval()
    return model


def atom_embeddings(model, data, real_idx):
    """128-d fused representation of every real atom (graph_fusion output)."""
    import torch  # noqa: PLC0415
    from torch_geometric.data import Batch  # noqa: PLC0415

    with torch.no_grad():
        batch = Batch.from_data_list([data])
        gcn = model._gcn(batch)
        gin = model._gin(batch)
        gat = model._gat(batch)
        fused = model.graph_fusion(torch.cat([gcn, gin, gat], dim=1))  # [100, 128]
        return fused[torch.as_tensor(real_idx)].cpu().numpy()


def self_similarity(emb):
    """n x n cosine self-similarity."""
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    e = emb / norm
    return e @ e.T


def cross_similarity(emb_a, emb_b):
    """n x m cross-molecule cosine similarity (atoms of A vs atoms of B)."""
    na = np.linalg.norm(emb_a, axis=1, keepdims=True)
    nb = np.linalg.norm(emb_b, axis=1, keepdims=True)
    na[na == 0], nb[nb == 0] = 1.0, 1.0
    return (emb_a / na) @ (emb_b / nb).T


def nice_ticks(n: int, step: int = TICK_EVERY):
    """Sampled ticks in [0, n-1]; append n-1 if it is not close to the last tick."""
    ticks = list(range(0, n, step))
    if n - 1 - ticks[-1] >= step - 2:
        ticks.append(n - 1)
    return ticks


def plot_heatmaps(pairs, outdir: Path, row_labels, col_labels):
    """pairs: [(mat_before, mat_after) x3]; row/col labels = drug names per row."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3, 2, figsize=(7.5, 10.2),
        gridspec_kw={"wspace": 0.32, "hspace": 0.28},
    )

    for r, (mat_b, mat_a) in enumerate(pairs):
        mats = (mat_b, mat_a)
        lo, hi = [(0.0, 1.0), (-0.5, 1.0), (-0.5, 1.0)][r]
        for c in range(2):
            ax = axes[r][c]
            mat = mats[c]
            im = ax.imshow(mat, cmap="YlOrRd", vmin=lo, vmax=hi, aspect="auto")

            ax.set_xticks(nice_ticks(mat.shape[1]))
            ax.set_yticks(nice_ticks(mat.shape[0]))
            ax.tick_params(direction="out", length=3, pad=2)

            ax.set_xlabel(col_labels[r], labelpad=6)
            ax.set_ylabel(row_labels[r], labelpad=6)

            if r == 0:
                ax.set_title("Before Training" if c == 0 else "After Training",
                             fontsize=11, fontweight="bold", pad=8)

    for c, tag in enumerate(("A", "B")):
        axes[0][c].text(-0.22, 1.08, tag, transform=axes[0][c].transAxes,
                        fontsize=11, fontweight="bold", va="bottom", ha="left")

    # one shared colorbar per row (range fixed per row as defined above)
    for r in range(3):
        im = axes[r][1].get_images()[0]
        cax = fig.add_axes(axes[r][1].get_position().bounds)
        cax.set_position([axes[r][1].get_position().x1 + 0.015,
                          axes[r][1].get_position().y0,
                          0.015,
                          axes[r][1].get_position().height])
        cb = fig.colorbar(im, cax=cax,
                          ticks=(CB_TICKS if im.get_clim()[0] < 0
                                 else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Value", fontsize=9)

    tif = outdir / "fig2_atomic_similarity.tif"
    pdf = outdir / "fig2_atomic_similarity.pdf"
    fig.savefig(tif, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return tif, pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes-dir", default=".",
                    help="directory containing loader.py and the model definition")
    ap.add_argument("--checkpoint", default="checkpoints/mfmnet_full_fold1.pt")
    ap.add_argument("--drug-name1", default="VINBLASTINE")
    ap.add_argument("--drug-name2", default="L-778123 free base")
    ap.add_argument("--label1", default="VINBLASTINE", help="display name in the figure")
    ap.add_argument("--label2", default="L778123", help="display name in the figure")
    ap.add_argument("--smiles1", default=None, help="optional: pass SMILES directly")
    ap.add_argument("--smiles2", default=None)
    ap.add_argument("--xlsx", default="DrugComb_drug_identifiers.xlsx")
    ap.add_argument("--outdir", default="figures/output/fig2")
    args = ap.parse_args()

    sys.path.insert(0, str(args.codes_dir))

    if args.smiles1:
        smiles1 = args.smiles1
    else:
        smiles1 = lookup_smiles(args.xlsx, args.drug_name1)
    if args.smiles2:
        smiles2 = args.smiles2
    else:
        smiles2 = lookup_smiles(args.xlsx, args.drug_name2)
    print(f"SMILES:\n  {args.label1}: {smiles1}\n  {args.label2}: {smiles2}")

    a, b = args.label1, args.label2
    model_before = load_model("full", None, args.codes_dir, seed=0)
    model_after = load_model("full", args.checkpoint, args.codes_dir)

    embs = {}
    for label, smiles in ((a, smiles1), (b, smiles2)):
        data, real_idx = molecule_graph(smiles, args.codes_dir)
        embs[label] = (
            atom_embeddings(model_before, data, real_idx),
            atom_embeddings(model_after, data, real_idx),
        )
        print(f"{label}: {len(real_idx)} real atoms")

    (ea_b, ea_a), (eb_b, eb_a) = embs[a], embs[b]
    pairs = [
        (cross_similarity(ea_b, eb_b), cross_similarity(ea_a, eb_a)),  # A x B
        (self_similarity(eb_b), self_similarity(eb_a)),                # B x B
        (self_similarity(ea_b), self_similarity(ea_a)),                # A x A
    ]
    row_labels = [a, b, a]
    col_labels = [b, b, a]

    tif, pdf = plot_heatmaps(pairs, Path(args.outdir), row_labels, col_labels)
    print(f"Written:\n  {tif}\n  {pdf}")


if __name__ == "__main__":
    main()

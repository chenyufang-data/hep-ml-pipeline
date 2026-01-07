#!/usr/bin/env python3
"""
jet_selection.py

Select exactly one b1 jet, one b2 jet, and one c1 jet per event from Delphes
ROOT files and write a flattened, event-level dataset.

Assumptions (Delphes format):
  - Tree name: "Delphes"
  - Jet branches: Jet.PT, Jet.Eta, Jet.Phi, Jet.Mass, Jet.BTag
  - b-jet identification: BTag == 1
  - c-jet identification: BTag == 16

This module is used internally by make_dataset.py and is not intended
to be run as a standalone script.

Outputs:
  - Flat table with one row per event containing selected jet features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import awkward as ak
import uproot
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.sanity import sanity_root_parquet, SanityError

# Physics helpers (no extra deps)
def _px_py_pz_e(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px * px + py * py + pz * pz
    e = np.sqrt(np.maximum(p2 + mass * mass, 0.0))
    return px, py, pz, e


def inv_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1, py1, pz1, e1 = _px_py_pz_e(pt1, eta1, phi1, m1)
    px2, py2, pz2, e2 = _px_py_pz_e(pt2, eta2, phi2, m2)
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    m2_tot = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2_tot, 0.0))

def inv_mass_3(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2, pt3, eta3, phi3, m3):
    px1, py1, pz1, e1 = _px_py_pz_e(pt1, eta1, phi1, m1)
    px2, py2, pz2, e2 = _px_py_pz_e(pt2, eta2, phi2, m2)
    px3, py3, pz3, e3 = _px_py_pz_e(pt3, eta3, phi3, m3)
    e = e1 + e2 + e3
    px = px1 + px2 + px3
    py = py1 + py2 + py3
    pz = pz1 + pz2 + pz3
    m2_tot = e * e - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2_tot, 0.0))


def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    deta = eta1 - eta2
    return np.sqrt(deta * deta + dphi * dphi)


# Selection config
@dataclass
class Cuts:
    b_pt: float = 25.0
    b_eta: float = 2.5
    c_pt: float = 25.0
    c_eta: float = 2.5


# Read a branch from an awkward record returned by uproot.iterate.
# Works whether arrs behaves like a record or mapping.
def get_branch(arrs, name, default=None):
    try:
        # awkward record has .fields
        if hasattr(arrs, "fields") and name in arrs.fields:
            return arrs[name]
        # some cases behave more like dict
        if isinstance(arrs, dict) and name in arrs:
            return arrs[name]
    except Exception:
        pass
    return default

# Core chunk processing
def process_chunk(arrs: dict, label: int, require_exact: bool, cuts: Cuts) -> pd.DataFrame:
    # uproot returns jagged arrays under names like "Jet.PTa
    pt   = get_branch(arrs, "Jet/Jet.PT")
    eta  = get_branch(arrs, "Jet/Jet.Eta")
    phi  = get_branch(arrs, "Jet/Jet.Phi")
    btag = get_branch(arrs, "Jet/Jet.BTag")
    if pt is None or eta is None or phi is None or btag is None:
        return pd.DataFrame()   # chunk doesn't have what we need

    mass = get_branch(arrs, "Jet/Jet.Mass", default=ak.zeros_like(pt))


    # masks
    bmask = (btag == 1) & (pt > cuts.b_pt) & (abs(eta) < cuts.b_eta)
    cmask = (btag == 16) & (pt > cuts.c_pt) & (abs(eta) < cuts.c_eta)

    b_pt = pt[bmask]
    b_eta = eta[bmask]
    b_phi = phi[bmask]
    b_m = mass[bmask]

    c_pt = pt[cmask]
    c_eta = eta[cmask]
    c_phi = phi[cmask]
    c_m = mass[cmask]

    nb = ak.num(b_pt)
    nc = ak.num(c_pt)

    if require_exact:
        keep = (nb == 2) & (nc == 1)
    else:
        keep = (nb >= 2) & (nc >= 1)

    # apply keep
    b_pt, b_eta, b_phi, b_m = b_pt[keep], b_eta[keep], b_phi[keep], b_m[keep]
    c_pt, c_eta, c_phi, c_m = c_pt[keep], c_eta[keep], c_phi[keep], c_m[keep]

    # choose b1,b2 by pT (descending), c1 is the (only) one (or highest pT if nc>1)
    b_order = ak.argsort(b_pt, axis=1, ascending=False)
    b_pt = b_pt[b_order]
    b_eta = b_eta[b_order]
    b_phi = b_phi[b_order]
    b_m = b_m[b_order]

    c_order = ak.argsort(c_pt, axis=1, ascending=False)
    c_pt = c_pt[c_order]
    c_eta = c_eta[c_order]
    c_phi = c_phi[c_order]
    c_m = c_m[c_order]

    # slice to needed multiplicity
    b1_pt = ak.to_numpy(b_pt[:, 0]).astype(np.float32)
    b1_eta = ak.to_numpy(b_eta[:, 0]).astype(np.float32)
    b1_phi = ak.to_numpy(b_phi[:, 0]).astype(np.float32)
    b1_m = ak.to_numpy(b_m[:, 0]).astype(np.float32)

    b2_pt = ak.to_numpy(b_pt[:, 1]).astype(np.float32)
    b2_eta = ak.to_numpy(b_eta[:, 1]).astype(np.float32)
    b2_phi = ak.to_numpy(b_phi[:, 1]).astype(np.float32)
    b2_m = ak.to_numpy(b_m[:, 1]).astype(np.float32)

    c1_pt = ak.to_numpy(c_pt[:, 0]).astype(np.float32)
    c1_eta = ak.to_numpy(c_eta[:, 0]).astype(np.float32)
    c1_phi = ak.to_numpy(c_phi[:, 0]).astype(np.float32)
    c1_m = ak.to_numpy(c_m[:, 0]).astype(np.float32)

    mbc1 = inv_mass(b1_pt, b1_eta, b1_phi, b1_m, c1_pt, c1_eta, c1_phi, c1_m).astype(np.float32)
    mbc2 = inv_mass(b2_pt, b2_eta, b2_phi, b2_m, c1_pt, c1_eta, c1_phi, c1_m).astype(np.float32)

    dr13 = delta_r(b1_eta, b1_phi, c1_eta, c1_phi).astype(np.float32)
    dr23 = delta_r(b2_eta, b2_phi, c1_eta, c1_phi).astype(np.float32)
    dr12 = delta_r(b1_eta, b1_phi, b2_eta, b2_phi).astype(np.float32)

    ratio_ptcb = c1_pt / b1_pt
    mbb = inv_mass(b1_pt, b1_eta, b1_phi, b1_m, b2_pt, b2_eta, b2_phi, b2_m).astype(np.float32)
    mcbb = inv_mass_3(c1_pt, c1_eta, c1_phi, c1_m, b1_pt, b1_eta, b1_phi, b1_m, b2_pt, b2_eta, b2_phi, b2_m).astype(np.float32)
    ht = b1_pt + b2_pt + c1_pt
    

    out = {
        "label": np.full_like(b1_pt, label, dtype=np.int32),
        "ptb1": b1_pt,
        "etab1": b1_eta,
        "ptb2": b2_pt,
        "etab2": b2_eta,
        "ptc1": c1_pt,
        "etac1": c1_eta,
        "mbc1": mbc1,
        "mbc2": mbc2,
        "dr13": dr13,
        "dr23": dr23,
        "ratio_ptcb": ratio_ptcb,
        "mbb": mbb,
        "mcbb": mcbb,
        "ht": ht,
        "dr12": dr12,
    }

    evt_weight = get_branch(arrs, "Event/Event.Weight", default=None)
    evt_xs = get_branch(arrs, "Event/Event.CrossSection", default=None)

    if evt_weight is None:
        weight = np.ones(len(b1_pt), dtype=np.float32)
    else:
        w_evnt = ak.firsts(evt_weight)
        w_evnt = w_evnt[keep]
        weight = ak.to_numpy(ak.fill_none(w_evnt,1.0)).astype(np.float32)

    out["weight"] = weight

    if evt_xs is None:
        xs = np.ones(len(b1_pt), dtype=np.float32)
    else:
        xs_evnt = ak.firsts(evt_xs)
        xs_evnt = xs_evnt[keep]
        xs = ak.to_numpy(ak.fill_none(xs_evnt,1.0)).astype(np.float32)

    out["xs"] = xs

    return pd.DataFrame(out)


# IO: iterate ROOT in chunks
def iter_chunks(files, tree_name="Delphes", step_size="200 MB"):
    branches = [
        "Jet/Jet.PT", "Jet/Jet.Eta", "Jet/Jet.Phi", "Jet/Jet.Mass", "Jet/Jet.BTag",
        "Event/Event.Weight", "Event/Event.CrossSection",
    ]

    for fpath in files:
        with uproot.open(fpath) as f:
            t = f[tree_name]
            # Only request branches that exist (ScalarHT may be missing)
            avail = set(t.keys())
            use = [b for b in branches if b in avail]
            for arrs in t.iterate(expressions=use, step_size=step_size, library="ak"):
                yield fpath, arrs

# --------------------------------------
# Main
# --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", action="append", required=True, help="Input ROOT file (repeatable)")
    ap.add_argument("-o", "--output", required=True, help="Output CSV/Parquet path")
    ap.add_argument("--tree", default="Delphes", help="Tree name (default: Delphes)")
    ap.add_argument("--label", type=int, required=True, help="Class label (signal=1, background=0)")
    ap.add_argument("--require-exact", action="store_true", help="Require exactly 2 b-jets and 1 c-jet")
    ap.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    ap.add_argument("--step-size", default="200 MB", help='uproot step_size (e.g. "200 MB")')
    ap.add_argument("--b-pt", type=float, default=25.0)
    ap.add_argument("--b-eta", type=float, default=2.5)
    ap.add_argument("--c-pt", type=float, default=25.0)
    ap.add_argument("--c-eta", type=float, default=2.5)
    ap.add_argument("--no-sanity", action="store_true", help="Disable per-chunk sanity checks")

    args = ap.parse_args()

    cuts = Cuts(b_pt=args.b_pt, b_eta=args.b_eta, c_pt=args.c_pt, c_eta=args.c_eta)

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    dfs = []

    total_rows = 0
    part = 0
    sum_w_all = 0.0
    sum_xsw_all = 0.0

    for fpath, arrs in tqdm(iter_chunks(args.input, tree_name=args.tree, step_size=args.step_size), desc="Reading chunks"):
        evt_weight = get_branch(arrs, "Event/Event.Weight", default=None)
        evt_xs     = get_branch(arrs, "Event/Event.CrossSection", default=None)

        if evt_weight is not None and evt_xs is not None:
            w_all  = ak.to_numpy(ak.fill_none(ak.firsts(evt_weight), 1.0)).astype(np.float64)
            xs_all = ak.to_numpy(ak.fill_none(ak.firsts(evt_xs),     1.0)).astype(np.float64)

            sum_w_all   += float(w_all.sum())
            sum_xsw_all += float((w_all * xs_all).sum())

        df = process_chunk(arrs, label=args.label, require_exact=args.require_exact, cuts=cuts)

        if df.empty:
            continue

        # ---- sanity check (per chunk, fail fast) ----
        if not args.no_sanity:
            try:
                sanity_root_parquet(df, name=f"{Path(fpath).name}:part{part:05d}")
            except SanityError as e:
                # include file context if you want: args.input
                raise SystemExit(f"[SANITY FAIL] {e}") from e

        total_rows += len(df)

        if args.format == "csv":
            # append CSV chunk
            header = not outpath.exists()
            df.to_csv(outpath, mode="a", header=header, index=False)
        else:
            # write chunk parquet part
            part_path = outpath.parent / f"{outpath.stem}.part{part:05d}.parquet"
            df.to_parquet(part_path, index=False)
            part += 1

    if sum_w_all <= 0:
        raise SystemExit("sum_w_all is zero/negative; cannot compute xs_total. Check Event.Weight.")
    xs_total = 2.0 * sum_xsw_all / sum_w_all

    import json

    meta = {
        "sum_w_all": sum_w_all,
        "sum_xsw_all": sum_xsw_all,
        "xs_total": xs_total,
        "note": "xs_total = 2*sum(w*xs)/sum(w) computed pre-selection",
        "inputs": [str(p) for p in args.input],
        "cuts": {"b_pt": args.b_pt, "b_eta": args.b_eta, "c_pt": args.c_pt, "c_eta": args.c_eta},
        "require_exact": bool(args.require_exact),
        "label": int(args.label),
    }
    meta_path = outpath.parent / f"{outpath.stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))



    print(
        f"Sanity summary: rows={total_rows}, "
        f"label={args.label}, "
        f"require_exact={args.require_exact}"
    )
    print(f"\nWrote {total_rows} rows -> {outpath} {args.format}")
    print(f"\n{outpath} meta.json Saved")


if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""Merge all scan results and plot the best-known N_mix vs D* curve.

This script:
  1. loads every nmix_scan_512*.npz result file in data/validation/;
  2. keeps the best (largest) D* per N_mix across all files/seeds;
  3. plots a direct line plot through the remaining best-known points.

Usage: python plot_nmix_scan.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'data', 'validation')


def load(path):
    d = np.load(path)
    return d['n_mix'], d['D'], d['d_R'], d['d_B'], d['m']


def main():
    # Load all SHA3-512 scan files.  The merged output is intentionally not
    # named nmix_scan_512*.npz, so it is not fed back into itself.
    files = sorted(glob.glob(os.path.join(OUT_DIR, 'nmix_scan_512*.npz')))
    if not files:
        print(f'No nmix_scan_512*.npz files found under {OUT_DIR}')
        return

    best = {}
    for f in files:
        try:
            n_mix, D, dR, dB, m = load(f)
        except Exception as exc:
            print(f'skip {f}: {exc}')
            continue
        for nm, dd, dr, db, mm in zip(n_mix, D, dR, dB, m):
            nm = int(nm)
            dd = int(dd)
            if nm <= 0:
                # N_mix = 0 is outside the training domain and unreliable here.
                continue
            if dd < 0:
                continue
            if nm not in best or dd > best[nm][0]:
                best[nm] = (dd, int(dr), int(db), int(mm), os.path.basename(f))

    if not best:
        print('No valid N_mix > 0 results found.')
        return

    # Sort by N_mix and extract the best-known values.
    uniq = np.array(sorted(best), dtype=int)
    D_u = np.array([best[nm][0] for nm in uniq], dtype=int)
    dR_u = np.array([best[nm][1] for nm in uniq], dtype=int)
    dB_u = np.array([best[nm][2] for nm in uniq], dtype=int)
    m_u = np.array([best[nm][3] for nm in uniq], dtype=int)

    print('loaded files:', [os.path.basename(f) for f in files])
    print('merged levels:', uniq.tolist())
    print('D*:', D_u.tolist())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(uniq, D_u, 'o-', color='#1f77b4', linewidth=1.8, markersize=6,
            label='best-known D*')
    # Training-domain marker
    ax.axvspan(41, 245, color='#ffcccc', alpha=0.35,
               label='N$_{mix}$ domain of training data [41, 245]')
    ax.set_xlabel('N$_{mix}$ (number of mixed bits, fixed)', fontsize=12)
    ax.set_ylabel('attack degree D = min(d$_R$, d$_B$, m)', fontsize=12)
    ax.set_title('N$_{mix}$ vs. attack degree (SHA3-512)',
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, 'nmix_vs_D_curve.png')
    pdf = os.path.join(OUT_DIR, 'nmix_vs_D_curve.pdf')
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f'figure saved: {png} / {pdf}')

    np.savez(os.path.join(OUT_DIR, 'nmix_scan_merged.npz'),
             n_mix=uniq, D=D_u,
             d_R=dR_u, d_B=dB_u, m=m_u)


if __name__ == '__main__':
    main()

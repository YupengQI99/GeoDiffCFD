#!/usr/bin/env python3
"""User-supplied independent planar-shear calibration (2026-09-27).

Reference-free planar-shear wall calibration for the quintic compact Heaviside.
Same 1D finite-volume problem as brinkman_mask_calibration, with a dense solve
independent of that module's Thomas algorithm. No force/reference data enters.
Coordinate > 0 is fluid; mask evaluated at coordinate-shift. A negative root
means an inward material correction phi_eff = phi + abs(shift).
Run with one BLAS thread under the bounded CPU test supervisor.
"""
import numpy as np


def quintic(phi, eps):
    s = np.clip(phi / eps, -1.0, 1.0)
    return 0.5 + s * (15 / 16 + s * s * (-5 / 8 + s * s * 3 / 16))


def cosine(phi, eps):
    s = phi / eps
    inner = 0.5 * (1 + s + np.sin(np.pi * s) / np.pi)
    return np.where(phi < -eps, 0.0, np.where(phi > eps, 1.0, inner))


def displacement(db_over_h, eps_over_h, shift_over_h, profile=quintic, sampling="tangential-cell",
                 solid=96, fluid=128, fit=16):
    a = 1.0 / db_over_h ** 2
    n = solid + fluid
    x = np.arange(n) - solid + 0.5
    if sampling == "tangential-cell":
        frac = profile(x - shift_over_h, eps_over_h)
    else:  # normal-face-average sensitivity; original expert implementation
        frac = 0.5 * (profile(x - 0.5 - shift_over_h, eps_over_h) + profile(x + 0.5 - shift_over_h, eps_over_h))
    alpha = a * (1 - frac)
    diag = 2 + alpha; diag[0] = 3 + alpha[0]; diag[-1] = 1 + alpha[-1]
    A = np.diag(diag) + np.diag(-np.ones(n - 1), -1) + np.diag(-np.ones(n - 1), 1)
    rhs = np.zeros(n); rhs[-1] = 1.0
    u = np.linalg.solve(A, rhs)
    i = np.arange(n - fit, n)
    slope, intercept = np.linalg.lstsq(np.c_[x[i], np.ones(fit)], u[i], rcond=None)[0]
    return intercept / slope


def calibrate(db_over_h, eps_over_h, **kw):
    lo, hi = -4.0, 4.0
    flo = displacement(db_over_h, eps_over_h, lo, **kw)
    for _ in range(200):
        mid = 0.5 * (lo + hi); f = displacement(db_over_h, eps_over_h, mid, **kw)
        if f * flo <= 0: hi = mid
        else: lo, flo = mid, f
        if hi - lo < 1e-13: break
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("check vs repository (compact-cosine, dB/h=1, eps/h=1.5): shift* = %.4f h  [paper DFG 2D-1: 0.78 h]"
          % calibrate(1.0, 1.5, profile=cosine))
    nu, D = 1e-3, 0.1
    for name, h, a, e in [("P 440x82", 0.005, 160, 0.015), ("A 440x82", 0.005, 640, 0.015),
                          ("B 440x82", 0.005, 160, 0.0075), ("C 880x164", 0.0025, 2560, 0.0075),
                          ("design 220x41", 0.01, 160, 0.015)]:
        db, eh = np.sqrt(nu / a) / h, e / h
        ell = displacement(db, eh, 0.0) * h
        s = calibrate(db, eh); sf = calibrate(db, eh, sampling="normal-face")
        print(f"{name:14s} dB/h={db:.3f} eps/h={eh:.2f} unshifted displacement={ell*1e3:+.3f} mm "
              f"(D_eff/D={1 - 2*ell/D:.4f})  shift*={s*h*1e3:+.4f} mm ({s:+.4f} h)  normal-face={sf*h*1e3:+.4f} mm")

"""
WKB greybody factor models and amplitude-only SXS fitting.

Convention
----------
f0 = Re(omega_0) > 0
t0 = Im(omega_0) < 0   (QNM decay rate, negative)
f1 = Re(omega_1) > 0
t1 = Im(omega_1) < 0

Equations (21), (22), (23) of Konoplya & Zhidenko (2408.11162).
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import minimize, curve_fit

_EXP_CLIP = 500.0

WKB_KEYS: dict[int, tuple[str, ...]] = {
    1: ("f0", "t0"),
    2: ("f0", "t0", "f1"),
    3: ("f0", "t0", "f1", "t1"),
}


def greybody_wkb(omega: np.ndarray, num_wkb: int, pars: dict) -> np.ndarray:
    """WKB reflectivity R(omega) = 1 - Gamma(omega).

    Parameters
    ----------
    omega : dimensionless frequency array (M omega)
    num_wkb : WKB order (1, 2, or 3)
    pars : dict with keys from WKB_KEYS[num_wkb]

    Returns
    -------
    R : reflectivity, real array in [0, 1].  R -> 1 at low freq, R -> 0 at high freq.
    """
    omega = np.asarray(omega, dtype=float)
    f0 = pars["f0"]
    t0 = pars["t0"]
    u = omega**2 - f0**2

    if num_wkb == 1:
        # Eq (21): eikonal
        iK = u / (4.0 * f0 * t0)

    elif num_wkb == 2:
        # Eq (22): first order beyond eikonal
        f1 = pars["f1"]
        iK = u / (4.0 * f0 * t0) - (f0 - f1) / (16.0 * t0)

    elif num_wkb == 3:
        # Eq (23): second order beyond eikonal
        f1 = pars["f1"]
        t1 = pars["t1"]
        df = f0 - f1

        corr_eik = 1.0 + df**2 / (32.0 * t0**2) - (3.0 * t0 - t1) / (24.0 * t0)
        corr_high = 1.0 + f0 * df / (4.0 * t0**2)

        iK = (
            u / (4.0 * f0 * t0) * corr_eik
            - df / (16.0 * t0)
            - u**2 / (16.0 * f0**3 * t0) * corr_high
            + u**3 / (32.0 * f0**5 * t0) * corr_high
            + f0**2 * (df**2 / (16.0 * t0**4) - (3.0 * t0 - t1) / (12.0 * t0))
        )
    else:
        raise ValueError("num_wkb must be 1, 2, or 3")

    exponent = np.clip(2.0 * np.pi * iK, -_EXP_CLIP, _EXP_CLIP)
    return 1.0 / (1.0 + np.exp(-exponent))


def _amp_model(omega: np.ndarray, R: np.ndarray, A: float, p: float) -> np.ndarray:
    """Amplitude model: |h(omega)| = A * R(omega) / omega^p."""
    return A * R / omega**p


def model_wkb(
    omega: np.ndarray,
    num_wkb: int,
    pars_wkb: dict,
    A: float,
    p: float,
) -> np.ndarray:
    """Amplitude model: |h(omega)| = A * R(omega) / omega^p."""
    R = greybody_wkb(omega, num_wkb, pars_wkb)
    return _amp_model(omega, R, A, p)


def amp_mismatch(omega: np.ndarray, abs1: np.ndarray, abs2: np.ndarray) -> float:
    """Amplitude-only mismatch: 1 - <|h1|,|h2|> / sqrt(<|h1|^2> <|h2|^2>)."""
    domega = np.mean(np.diff(omega))
    inner12 = np.sum(abs1 * abs2) * domega
    inner11 = np.sum(abs1**2) * domega
    inner22 = np.sum(abs2**2) * domega
    if inner11 * inner22 <= 0:
        return 1.0
    return 1.0 - inner12 / np.sqrt(inner11 * inner22)


def fit_wkb_amplitude(
    omega_fit: np.ndarray,
    H_fit: np.ndarray,
    num_wkb: int,
    x0_wkb: list[float],
) -> dict:
    """Fit WKB greybody model to |h(omega)| via profile likelihood.

    For each trial of WKB parameters, (A, p) are optimally determined
    via curve_fit.  The WKB parameters are optimized with Nelder-Mead.

    Parameters
    ----------
    omega_fit : frequency array in the fit interval
    H_fit : complex waveform in the fit interval
    num_wkb : WKB order (1, 2, or 3)
    x0_wkb : initial guess for WKB parameters, ordered as WKB_KEYS[num_wkb]

    Returns
    -------
    dict with keys: wkb_params, A, p, mismatch, model_amplitude, greybody
    """
    abs_num = np.abs(H_fit)
    keys = WKB_KEYS[num_wkb]

    def _objective(wkb_vec):
        pars = dict(zip(keys, wkb_vec))
        if pars["f0"] <= 0 or pars["t0"] >= 0:
            return 1.0
        if num_wkb >= 2 and pars.get("f1", 1.0) <= 0:
            return 1.0
        if num_wkb >= 3 and pars.get("t1", -1.0) >= 0:
            return 1.0
        try:
            R = greybody_wkb(omega_fit, num_wkb, pars)
        except Exception:
            return 1.0
        if not np.all(np.isfinite(R)) or np.max(R) < 1e-30:
            return 1.0
        try:
            popt, _ = curve_fit(
                lambda og, A, p: _amp_model(og, R, A, p),
                omega_fit, abs_num, p0=[1.0, 0.5], maxfev=5000,
            )
        except RuntimeError:
            return 1.0
        mdl = _amp_model(omega_fit, R, *popt)
        return amp_mismatch(omega_fit, abs_num, mdl)

    res = minimize(
        _objective,
        x0=x0_wkb,
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 5000},
    )

    pars_best = dict(zip(keys, res.x))
    R = greybody_wkb(omega_fit, num_wkb, pars_best)
    popt, _ = curve_fit(
        lambda og, A, p: _amp_model(og, R, A, p),
        omega_fit, abs_num, p0=[1.0, 0.5], maxfev=5000,
    )
    A_fit, p_fit = popt
    mdl = _amp_model(omega_fit, R, A_fit, p_fit)
    mm = amp_mismatch(omega_fit, abs_num, mdl)

    return {
        "wkb_params": pars_best,
        "A": A_fit,
        "p": p_fit,
        "mismatch": mm,
        "model_amplitude": mdl,
        "reflectivity": R,
    }


def fit_amplitude_tabulated(
    omega_fit: np.ndarray,
    H_fit: np.ndarray,
    f_abs,
) -> dict:
    """Amplitude-only fit using the tabulated reflectivity."""
    abs_num = np.abs(H_fit)
    R_th = f_abs(omega_fit)
    popt, _ = curve_fit(
        lambda og, A, p: _amp_model(og, R_th, A, p),
        omega_fit, abs_num, p0=[1.0, 0.5], maxfev=5000,
    )
    A_fit, p_fit = popt
    mdl = _amp_model(omega_fit, R_th, A_fit, p_fit)
    mm = amp_mismatch(omega_fit, abs_num, mdl)
    return {
        "A": A_fit,
        "p": p_fit,
        "mismatch": mm,
        "model_amplitude": mdl,
        "reflectivity": R_th,
    }


def load_kerr_qnm(chi_final: float, ell: int, m: int, theory_dir: Path) -> dict:
    """Load Kerr QNM frequencies from tabulated data.

    Returns dict with f0, t0, f1, t1 in the Im(omega) < 0 convention.
    """
    m_abs = abs(m)
    qnm = {}
    for n, prefix in [(1, "0"), (2, "1")]:
        fname = theory_dir / f"l{ell}" / f"n{n}l{ell}m{m_abs}.dat"
        data = np.loadtxt(str(fname))
        f_re = interp1d(data[:, 0], data[:, 1], kind="cubic")
        f_im = interp1d(data[:, 0], data[:, 2], kind="cubic")
        qnm[f"f{prefix}"] = float(f_re(chi_final))
        qnm[f"t{prefix}"] = float(f_im(chi_final))
    return qnm

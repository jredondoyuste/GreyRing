"""
WKB bilby-compatible waveform models for injection-recovery.

Each function follows the bilby source-model interface:
    f(frequency_array, ..., **kwargs) -> {"plus": ndarray, "cross": ndarray}
"""

import numpy as np
import lal

from .agnostic import greybody_wkb

twopi = 2.0 * np.pi
C_mt = (lal.MSUN_SI * lal.G_SI) / (lal.C_SI**3)
C_md = (lal.MSUN_SI * lal.G_SI) / (1e6 * lal.PC_SI * lal.C_SI**2)


def _wkb_model(frequency_array, final_mass, f0, t0, A, p, c,
               luminosity_distance, num_wkb=1, f1=None, t1=None):
    """Core WKB amplitude+phase model in physical units.

    Returns (H2, phi) on the positive-frequency mask,
    plus the boolean mask itself.
    """
    f = np.asarray(frequency_array, dtype=float)
    m = f > 0.0

    omega = twopi * final_mass * C_mt * f[m]

    pars = {"f0": f0, "t0": t0}
    if num_wkb >= 2 and f1 is not None:
        pars["f1"] = f1
    if num_wkb >= 3 and t1 is not None:
        pars["t1"] = t1

    R = 1.0 - greybody_wkb(omega, num_wkb, pars)

    H2 = (A * R) / (omega ** p)
    phi = c / omega

    H2 *= final_mass**2 * C_md * C_mt / luminosity_distance

    return f, m, H2, -phi


def wkb_eikonal_22(
    frequency_array,
    final_mass,
    f0, t0,
    A, p, c,
    luminosity_distance,
    theta_jn,
    **kwargs,
):
    """WKB eikonal (order 1) waveform model for the (2,2) mode.

    Parameters
    ----------
    frequency_array : array-like
        Frequencies in Hz.
    final_mass : float
        Final mass in solar masses (fixed; needed for Hz -> Momega).
    f0, t0 : float
        Dimensionless QNM frequency and damping rate (t0 < 0).
    A, p : float
        Phenomenological amplitude parameters.
    c : float
        Phase parameter.
    luminosity_distance : float
        Luminosity distance in Mpc.
    theta_jn : float
        Inclination angle in radians.

    Returns
    -------
    dict with keys "plus", "cross" (complex ndarrays).
    """
    f, m, H2, phi = _wkb_model(
        frequency_array, final_mass, f0, t0, A, p, c,
        luminosity_distance, num_wkb=1,
    )

    wave_plus = np.zeros_like(f, dtype=complex)
    wave_cross = np.zeros_like(f, dtype=complex)
    if not np.any(m):
        return {"plus": wave_plus, "cross": wave_cross}

    H2 = H2 / 2.0
    ampl_plus = H2 * (1.0 + np.cos(theta_jn)**2) / 2.0
    ampl_cross = H2 * np.cos(theta_jn)

    wave_plus[m] = np.sqrt(5.0 / (4.0 * np.pi)) * ampl_plus * np.exp(1j * phi)
    wave_cross[m] = np.sqrt(5.0 / (4.0 * np.pi)) * ampl_cross * (-1j) * np.exp(1j * phi)

    return {"plus": wave_plus, "cross": wave_cross}

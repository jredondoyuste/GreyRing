"""
Bilby-compatible waveform models using WKB greybody factors.

Four agnostic models that replace the tabulated Kerr reflectivity with
analytical WKB formulas parametrized by light-ring or QNM quantities.
"""

import numpy as np
import lal

from .wkb_reflectivity import (
    greybody_eikonal_lr,
    greybody_eikonal_qnm,
    greybody_second_order_lr,
    greybody_second_order_qnm,
)

twopi = 2.0 * np.pi
C_mt = (lal.MSUN_SI * lal.G_SI) / (lal.C_SI**3)
C_md = (lal.MSUN_SI * lal.G_SI) / (1e6 * lal.PC_SI * lal.C_SI**2)


def _build_waveform(frequency_array, final_mass, luminosity_distance, theta_jn, gamma_complex, A, p, c):
    """Shared logic: from complex greybody factor to plus/cross polarizations."""
    f = np.asarray(frequency_array, dtype=float)
    wave_plus = np.zeros_like(f, dtype=complex)
    wave_cross = np.zeros_like(f, dtype=complex)
    m = f > 0.0
    if not np.any(m):
        return {"plus": wave_plus, "cross": wave_cross}

    omega = twopi * final_mass * C_mt * f[m]

    R = np.abs(gamma_complex)
    Rphase = np.angle(gamma_complex)

    H2 = (A * R) / (omega**p)
    phi = c / omega + Rphase

    H2 *= final_mass**2 * C_md * C_mt / luminosity_distance
    phi = -phi

    H2 = H2 / 2.0
    ampl_plus = H2 * (1.0 + np.cos(theta_jn) ** 2) / 2.0
    ampl_cross = H2 * np.cos(theta_jn)

    wave_plus[m] = np.sqrt(5.0 / (4.0 * np.pi)) * ampl_plus * np.exp(1j * phi)
    wave_cross[m] = np.sqrt(5.0 / (4.0 * np.pi)) * ampl_cross * (-1j) * np.exp(1j * phi)
    return {"plus": wave_plus, "cross": wave_cross}


# ── Model 1: eikonal, light-ring parametrization ────────────────────────

def wkb_eikonal_lr_22(
    frequency_array,
    final_mass,
    omega_lr,
    lambda_lr,
    A,
    p,
    c,
    luminosity_distance,
    theta_jn,
    **kwargs,
):
    f = np.asarray(frequency_array, dtype=float)
    m = f > 0.0
    if not np.any(m):
        return {"plus": np.zeros_like(f, dtype=complex), "cross": np.zeros_like(f, dtype=complex)}

    omega = twopi * final_mass * C_mt * f[m]
    gamma = greybody_eikonal_lr(omega, omega_lr, lambda_lr)
    return _build_waveform(f, final_mass, luminosity_distance, theta_jn, gamma, A, p, c)


# ── Model 2: eikonal, QNM parametrization ───────────────────────────────

def wkb_eikonal_qnm_22(
    frequency_array,
    final_mass,
    re_omega0,
    im_omega0,
    A,
    p,
    c,
    luminosity_distance,
    theta_jn,
    **kwargs,
):
    f = np.asarray(frequency_array, dtype=float)
    m = f > 0.0
    if not np.any(m):
        return {"plus": np.zeros_like(f, dtype=complex), "cross": np.zeros_like(f, dtype=complex)}

    omega = twopi * final_mass * C_mt * f[m]
    gamma = greybody_eikonal_qnm(omega, re_omega0, im_omega0)
    return _build_waveform(f, final_mass, luminosity_distance, theta_jn, gamma, A, p, c)


# ── Model 3: second-order WKB, light-ring + potential derivatives ───────

def wkb_second_order_lr_22(
    frequency_array,
    final_mass,
    omega_lr,
    lambda_lr,
    v3,
    v4,
    A,
    p,
    c,
    luminosity_distance,
    theta_jn,
    **kwargs,
):
    f = np.asarray(frequency_array, dtype=float)
    m = f > 0.0
    if not np.any(m):
        return {"plus": np.zeros_like(f, dtype=complex), "cross": np.zeros_like(f, dtype=complex)}

    omega = twopi * final_mass * C_mt * f[m]
    gamma = greybody_second_order_lr(omega, omega_lr, lambda_lr, v3, v4)
    return _build_waveform(f, final_mass, luminosity_distance, theta_jn, gamma, A, p, c)


# ── Model 4: second-order WKB, QNM parametrization ─────────────────────

def wkb_second_order_qnm_22(
    frequency_array,
    final_mass,
    re_omega0,
    im_omega0,
    re_omega1,
    im_omega1,
    A,
    p,
    c,
    luminosity_distance,
    theta_jn,
    **kwargs,
):
    f = np.asarray(frequency_array, dtype=float)
    m = f > 0.0
    if not np.any(m):
        return {"plus": np.zeros_like(f, dtype=complex), "cross": np.zeros_like(f, dtype=complex)}

    omega = twopi * final_mass * C_mt * f[m]
    gamma = greybody_second_order_qnm(omega, re_omega0, im_omega0, re_omega1, im_omega1)
    return _build_waveform(f, final_mass, luminosity_distance, theta_jn, gamma, A, p, c)

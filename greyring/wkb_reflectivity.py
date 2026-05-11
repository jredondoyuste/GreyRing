"""
WKB greybody factors parametrized by light-ring or QNM quantities.

Four models at two WKB orders (eikonal and second-order) with two
parametrizations each (potential/LR derivatives and QNM frequencies).

Convention: Gamma = 1 / (1 + exp(2 pi i K)), consistent with
Konoplya & Zhidenko (2408.11162) eq (18) and the wkb_code paper eq (13).
"""

import numpy as np

_EXP_CLIP = 500.0


def _safe_gamma(exponent: np.ndarray) -> np.ndarray:
    """Compute 1/(1+exp(exponent)) with overflow protection."""
    exponent = np.asarray(exponent, dtype=complex)
    re = np.real(exponent)
    re_clipped = np.clip(re, -_EXP_CLIP, _EXP_CLIP)
    return 1.0 / (1.0 + np.exp(re_clipped + 1j * np.imag(exponent)))


def greybody_eikonal_lr(omega: np.ndarray, omega_lr: float, lambda_lr: float) -> np.ndarray:
    """Eikonal greybody factor from light-ring quantities.

    iK = (V_0 - omega^2) / sqrt(-2 V_2)
       = (Omega_LR^2 - omega^2) / (2 Omega_LR lambda_LR)

    Parameters
    ----------
    omega : dimensionless frequency (M omega in geom. units)
    omega_lr : light-ring orbital frequency Omega_LR
    lambda_lr : Lyapunov exponent lambda_LR
    """
    iK = (omega_lr**2 - omega**2) / (2.0 * omega_lr * lambda_lr)
    return _safe_gamma(2.0 * np.pi * iK)


def greybody_eikonal_qnm(omega: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """Eikonal greybody factor from fundamental QNM frequency.

    iK = (omega^2 - Re(omega_0)^2) / (4 Re(omega_0) Im(omega_0))

    with f0 = Re(omega_0), t0 = -Im(omega_0) > 0.

    Parameters
    ----------
    omega : dimensionless frequency
    f0 : real part of the fundamental QNM frequency
    t0 : (minus) imaginary part of the fundamental QNM frequency
    """
    k_factor = -1j * (omega**2 - f0**2) / (4.0 * f0 * t0)
    return _safe_gamma(-2.0 * np.pi * 1j * k_factor)


def greybody_second_order_lr(
    omega: np.ndarray,
    omega_lr: float,
    lambda_lr: float,
    v3: float,
    v4: float,
) -> np.ndarray:
    """Second-order WKB greybody factor from LR quantities + potential derivatives.

    Includes the Iyer-Will A_2 correction built from V_0, V_2, V_3, V_4
    where V_n = d^n V / dx^n at the potential maximum.

    V_0 = Omega_LR^2,  V_2 = -2 Omega_LR^2 lambda_LR^2,
    V_3 = v3,  V_4 = v4.

    Parameters
    ----------
    omega : dimensionless frequency
    omega_lr : light-ring orbital frequency
    lambda_lr : Lyapunov exponent
    v3 : third derivative of the potential at its maximum
    v4 : fourth derivative of the potential at its maximum
    """
    V0 = omega_lr**2
    V2 = -2.0 * omega_lr**2 * lambda_lr**2
    sqrt_neg2V2 = 2.0 * omega_lr * lambda_lr

    U0 = V0 - omega**2
    iK0 = U0 / sqrt_neg2V2

    # K_0^2 = -(iK_0)^2 for the scattering problem (K purely imaginary)
    K0_sq = -(iK0**2)

    # Iyer-Will A_2 correction: Schutz, Iyer & Will (1987)
    A2 = (1.0 / (-2.0 * V2)) * (
        (1.0 / 8.0) * (v4 / V2) * (0.25 + K0_sq)
        - (1.0 / 288.0) * (v3 / V2) ** 2 * (7.0 + 60.0 * K0_sq)
    )

    iK = iK0 + A2 / sqrt_neg2V2
    return _safe_gamma(2.0 * np.pi * iK)


def greybody_second_order_qnm(
    omega: np.ndarray,
    f0: float,
    t0: float,
    f1: float,
    t1: float,
) -> np.ndarray:
    """Second-order WKB greybody factor from QNM frequencies.

    iK from eq (23) of Konoplya & Zhidenko (2408.11162), parametrized
    by the fundamental mode (f0, t0) and first overtone (f1, t1).

    f_n = Re(omega_n),  t_n = -Im(omega_n) > 0.

    Parameters
    ----------
    omega : dimensionless frequency
    f0 : real part of fundamental QNM frequency
    t0 : (minus) imaginary part of fundamental QNM frequency
    f1 : real part of first overtone
    t1 : (minus) imaginary part of first overtone
    """
    w2_f02 = omega**2 - f0**2

    Delta1 = (f0 - f1) / (16.0 * f0)

    Delta2 = (
        -w2_f02 / (32.0 * f0 * t0)
        * ((f0 - f1) ** 2 / (4.0 * t0**2) - (3.0 * t0 - t1) / (3.0 * t0))
        + w2_f02**2 / (16.0 * f0**3 * t0)
        * (1.0 + f0 * (f0 - f1) / (4.0 * t0**2))
    )

    Deltaf = (
        -w2_f02**3 / (32.0 * f0**5 * t0)
        * (
            1.0
            + f0 * (f0 - f1) / (4.0 * t0**2)
            + f0**2
            * ((f0 - f1) ** 2 / (16.0 * t0**4) - (3.0 * t0 - t1) / (12.0 * t0))
        )
    )

    k_factor = -1j * w2_f02 / (4.0 * f0 * t0) + Delta1 + Delta2 + Deltaf
    return _safe_gamma(-2.0 * np.pi * 1j * k_factor)

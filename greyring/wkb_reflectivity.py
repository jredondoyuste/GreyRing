"""
WKB reflectivity parametrized by light-ring or QNM quantities.

Returns the REFLECTIVITY R = 1/(1 + exp(-2*pi*i*nu)), consistent with
the GreyRing convention: |R| -> 1 below the barrier, |R| -> 0 above.

The transmission coefficient is T = 1/(1 + exp(+2*pi*i*nu));
|T|^2 + |R|^2 = 1 for real potentials.

nu = iQ0/sqrt(2Q0'') - Lambda - Omega  (Iyer & Will 1987).
Lambda (2nd order, eq 1.5a) is REAL  -> introduces arg(R).
Omega  (3rd order, eq 1.5b) is IMAG  -> corrects |R|.
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
    return _safe_gamma(-2.0 * np.pi * iK)


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
    return _safe_gamma(2.0 * np.pi * 1j * k_factor)


def _lambda_iw(V2, v3, v4, K0_sq, sqrt_neg2V2):
    """Iyer-Will Lambda correction (eq 1.5a). Real-valued."""
    return (1.0 / sqrt_neg2V2) * (
        (1.0 / 8.0) * (v4 / V2) * (0.25 + K0_sq)
        - (1.0 / 288.0) * (v3 / V2) ** 2 * (7.0 + 60.0 * K0_sq)
    )


def _omega_iw(V2, v3, v4, v5, v6, K0, K0_sq):
    """Iyer-Will Omega correction (eq 1.5b). Purely imaginary."""
    bracket = (
        (5.0 / 6912.0) * (v3 / V2) ** 4 * (77.0 + 188.0 * K0_sq)
        - (1.0 / 384.0) * v3**2 * v4 / V2**3 * (51.0 + 100.0 * K0_sq)
        + (1.0 / 2304.0) * (v4 / V2) ** 2 * (67.0 + 68.0 * K0_sq)
        + (1.0 / 288.0) * v3 * v5 / V2**2 * (19.0 + 28.0 * K0_sq)
        - (1.0 / 288.0) * (v6 / V2) * (5.0 + 4.0 * K0_sq)
    )
    return K0 / (-2.0 * V2) * bracket


def _lr_common(omega, omega_lr, lambda_lr):
    """Shared quantities for LR-parametrized models."""
    V0 = omega_lr**2
    V2 = -2.0 * omega_lr**2 * lambda_lr**2
    sqrt_neg2V2 = 2.0 * omega_lr * lambda_lr
    iK0 = (V0 - omega**2) / sqrt_neg2V2
    K0 = -1j * iK0
    K0_sq = K0**2
    return V2, sqrt_neg2V2, iK0, K0, K0_sq


def greybody_second_order_lr(
    omega: np.ndarray,
    omega_lr: float,
    lambda_lr: float,
    v3: float,
    v4: float,
) -> np.ndarray:
    """Second-order WKB greybody factor from LR quantities + potential derivatives.

    Includes the Iyer-Will Lambda correction (eq 1.5a).
    Lambda is real, so -2i*pi*Lambda introduces arg(Gamma).

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
    V2, sqrt_neg2V2, iK0, K0, K0_sq = _lr_common(omega, omega_lr, lambda_lr)
    Lambda = _lambda_iw(V2, v3, v4, K0_sq, sqrt_neg2V2)
    exponent = 2.0 * np.pi * iK0 - 2j * np.pi * Lambda
    return _safe_gamma(-exponent)


def greybody_third_order_lr(
    omega: np.ndarray,
    omega_lr: float,
    lambda_lr: float,
    v3: float,
    v4: float,
    v5: float,
    v6: float,
) -> np.ndarray:
    """Third-order WKB greybody factor from LR quantities + potential derivatives.

    Includes both Iyer-Will corrections:
    Lambda (eq 1.5a, real)  -> introduces arg(Gamma)
    Omega  (eq 1.5b, imag) -> corrects |Gamma|

    Parameters
    ----------
    omega : dimensionless frequency
    omega_lr : light-ring orbital frequency
    lambda_lr : Lyapunov exponent
    v3 : third derivative of the potential at its maximum
    v4 : fourth derivative of the potential at its maximum
    v5 : fifth derivative of the potential at its maximum
    v6 : sixth derivative of the potential at its maximum
    """
    V2, sqrt_neg2V2, iK0, K0, K0_sq = _lr_common(omega, omega_lr, lambda_lr)
    Lambda = _lambda_iw(V2, v3, v4, K0_sq, sqrt_neg2V2)
    Omega = _omega_iw(V2, v3, v4, v5, v6, K0, K0_sq)
    exponent = 2.0 * np.pi * iK0 - 2j * np.pi * Lambda - 2j * np.pi * Omega
    return _safe_gamma(-exponent)


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
    return _safe_gamma(2.0 * np.pi * 1j * k_factor)

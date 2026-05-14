# Fourier transform of the SXS:BBH signal

############################# IMPORTS #############################

import numpy as np
import sxs
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d



############################## UTILS ##############################
"""
Here we define omega_star (omega_x in https://arxiv.org/pdf/2604.11895, supplemental material section B) via the real part of the fundamental QNM associated to the remnant mass and spin. One can also define it in a simulation dependent way, results will be similar. The QNMs are taken from https://pages.jh.edu/eberti2/ringdown/.
"""
OMEGA_STAR_TEMPLATE = "theory/l{ell}/n1l{ell}m{m}.dat"


############################### FFT ###############################

def load_simulation(sim_number, download=False):
    """
    Loading the sxs simulation
    """
    return sxs.load(f"SXS:BBH:{sim_number}", download=download)


def tukey_left(N, alpha=0.15):
    """
    Tukey window
    """
    N_taper = int(alpha * N)
    w = np.ones(N)
    if N_taper > 0:
        x = np.linspace(0, np.pi, 2 * N_taper, endpoint=False)
        left = 0.5 * (1 - np.cos(x))[:N_taper]
        w[:N_taper] = left
    return w



def fft_sxs_app(sim, omega_max=1.0, ell=2, m=2):
    """
     Input:
       sim: SXS simulation object
       omega_max: maximum absolute angular frequency to keep, float
     Output:
       omega_all[mask]: filtered angular frequency array, ndarray
       H[mask]: Fourier transform of the (2,2) mode, ndarray. To adapt for different multipoles.
    """
    w = sim.h

    h_preprocessed = w.preprocess()
    h22 = h_preprocessed[:, sim.h.index(ell,m)] #change here to use a different multipole

    t = np.asarray(h22.t, float)
    dt = np.min(np.diff(t))

    freqs = np.fft.fftfreq(t.size, dt)
    omega_all = 2 * np.pi * freqs

    H = np.fft.fft(h22.ndarray) * dt

    t0 = t[0]
    H = H * np.exp(1j * omega_all * t0)

    idx = np.argsort(omega_all)
    omega_all = omega_all[idx]
    H = H[idx]

    mask = (omega_all > -omega_max) & (omega_all < omega_max)
    return omega_all[mask], H[mask]


def _extract_final_spin(sim):
    """
    Input:
        sim: SXS simulation object
    Output:
        abs_chi_f: magnitude of the final dimensionless spin, float
    If the metadata contains a spin vector, its norm is used.
    """
    chi = sim.metadata.remnant_dimensionless_spin
    chi = np.asarray(chi)

    if chi.ndim == 0:
        return abs(float(chi))

    return float(np.linalg.norm(chi))


def build_dimensionless_positive_spectrum(sim, omega_max_search=1.0, ell=2, m=2):
    """
    Input:
        sim: SXS simulation object
        omega_max_search: maximum angular frequency used in the FFT, float
    Output:
        Momega: positive dimensionless frequency array, ndarray
        H: Fourier-domain waveform on the positive Momega branch, ndarray
        M_final: remnant mass, float
        chi_final: magnitude of the final dimensionless spin, float
        
    The spectrum is expressed in the dimensionless variable Momega = - M_final * omega_all.
    This is done to follow the convention of https://arxiv.org/pdf/2604.11895, details in Supplemental Material Sec.B.
    """
    omega_all, H_all = fft_sxs_app(sim, omega_max=omega_max_search, ell=ell, m=m)

    M_final = float(sim.metadata.remnant_mass)
    chi_final = _extract_final_spin(sim)

    if m >= 0:
        Momega_all = -M_final * omega_all
    elif m < 0:
        Momega_all = M_final * omega_all
    else:
        raise ValueError("The m=0 case is not implemented in this convention.")

    idx = np.argsort(Momega_all)

    Momega_all = Momega_all[idx]
    H_all = H_all[idx]

    mask = Momega_all > 0
    return Momega_all[mask], H_all[mask], M_final, chi_final


############################## KNEE ###############################

def omega_star_from_file(spin_star, ell=2, m=2, template=OMEGA_STAR_TEMPLATE):
    """
    We extract the omega_x frequency from the file, as explained in line 15.
    """
    filename = template.format(ell=ell, m=np.abs(m))
    data = np.loadtxt(filename)

    spin_vals = np.asarray(data[:, 0], float)
    omegaR_vals = np.asarray(data[:, 1], float)

    idx = np.argsort(spin_vals)
    spin_vals = spin_vals[idx]
    omegaR_vals = omegaR_vals[idx]

    omega_of_spin = interp1d(spin_vals,omegaR_vals,kind="cubic",fill_value="extrapolate")

    return float(omega_of_spin(spin_star))


def find_knee_frequency(sim=None, spin_star=None, ell=2, m=2,
                        template=OMEGA_STAR_TEMPLATE):

    return omega_star_from_file(spin_star=spin_star,ell=ell,m=m,template=template)

########################## OMEGA RANGE ############################

def prepare_positive_branch(omega, H, omega_min, omega_max_search=1.0):
    """
    Input:
        omega: frequency array, ndarray
        H: Fourier-domain waveform, ndarray
        omega_min: lower frequency bound, float
        omega_max_search: upper frequency bound, float
    Output:
        omega: filtered and sorted positive-frequency array, ndarray
        H: filtered and sorted Fourier-domain waveform, ndarray
    Non-finite entries are removed, the arrays are sorted in omega,
    and only the interval omega_min < omega <= omega_max is kept.
    """
    omega = np.asarray(omega, float)
    H = np.asarray(H, complex)

    mask = np.isfinite(omega) & np.isfinite(H.real) & np.isfinite(H.imag)
    omega = omega[mask]
    H = H[mask]

    idx = np.argsort(omega)
    omega = omega[idx]
    H = H[idx]

    mask = (omega > omega_min) & (omega <= omega_max_search)
    omega = omega[mask]
    H = H[mask]

    if len(omega) < 5:
        raise ValueError("omega array too short")

    return omega, H


def find_omegarange(omega,H,omega_x,omega_i_factor=0.7,ratio_factor=20.0,omega_max_search=1.0):
    """
    Input:
        omega: frequency array, ndarray
        H: Fourier-domain waveform, ndarray
        omega_x: reference frequency, float
        omega_i_factor: factor defining omega_i = omega_i_factor * omega_x, float
        ratio_factor: factor used to define the amplitude threshold, float
        omega_max_search: upper frequency bound for the search, float
    Output:
        omega_i: initial frequency of the fit interval, float
        omega_f: final frequency selected from the amplitude-ratio criterion, float
        info: dictionary with diagnostic quantities, dict
    The function defines omega_i from omega_x, smooths the waveform amplitude,
    and finds the first frequency omega_f >= omega_x such that the smoothed
    amplitude drops below h(omega_x) / ratio_factor.
    """
    omega_i = omega_i_factor * omega_x

    omega, H = prepare_positive_branch(omega, H, omega_i, omega_max_search=omega_max_search)
    amp = np.abs(H)

    hx = np.interp(omega_x, omega, amp)
    threshold = hx / ratio_factor

    mask_after = omega >= omega_x
    omega2 = omega[mask_after]
    amp2 = amp[mask_after]

    if len(omega2) == 0:
        omega_f = omega[-1]
    else:
        idx_ok = np.where(amp2 <= threshold)[0]
        if len(idx_ok) == 0:
            omega_f = omega2[-1]
        else:
            omega_f = omega2[idx_ok[0]]

    info = {
        "mode": "amp_ratio",
        "h_omega_x": hx,
        "threshold": threshold,
    }
    return omega_i, omega_f, info
    
def prepare_fit_data(sim_number,omega_max_search=1.0,ell=2,m=2,omega_i_factor=0.7,ratio_factor=20.0):
    """
    Input:
        sim_number: simulation number, int
        omega_max_search: maximum frequency used in the FFT, float
        ell: multipole index, int
        m: azimuthal index, int
        omega_i_factor: factor defining omega_i = omega_i_factor * omega_x, float
        ratio_factor: factor used in the amplitude-ratio criterion, float
    Output:
        omega_fit: frequency array in the fit interval, ndarray
        H_fit: waveform in the fit interval, ndarray
        omega_all: full positive-frequency array, ndarray
        H_all: full positive-frequency waveform, ndarray
        M_final: remnant mass, float
        chi_final: remnant spin magnitude, float
        omega_x: reference frequency, float
        omega_i: initial fit frequency, float
        omega_f: final fit frequency, float
    """
    sim = load_simulation(sim_number, download=False)

    omega_all, H_all, M_final, chi_final = build_dimensionless_positive_spectrum(sim,omega_max_search=omega_max_search,ell=ell,m=m)

    omega_x = find_knee_frequency(spin_star=chi_final,ell=ell,m=m)

    omega_i, omega_f, _ = find_omegarange(omega_all,H_all,omega_x,omega_i_factor=omega_i_factor,ratio_factor=ratio_factor,omega_max_search=omega_max_search)

    mask_fit = (omega_all >= omega_i) & (omega_all <= omega_f)

    return (
        omega_all[mask_fit],
        H_all[mask_fit],
        omega_all,
        H_all,
        M_final,
        chi_final,
        omega_x,
        omega_i,
        omega_f,
    )

######################## LINEAR DETRENDING ########################

def detrend_linear(x, y):
    """
    Input:
        x: independent variable array, ndarray
        y: dependent variable array, ndarray
    Output:
        y_detrended: array with the best-fit linear trend removed, ndarray
        m: slope of the best-fit line, float
        q: intercept of the best-fit line, float
    The function fits y with a linear model m * x + q
    and returns y - (m * x + q).
    """
    m, q = np.polyfit(x, y, 1)
    return y - (m * x + q), m, q

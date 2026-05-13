from dataclasses import dataclass, field
from pathlib import Path

from .model_utils import mismatch_complex
from .sxs_processing import prepare_fit_data
from .fitting import (
    build_theory_interpolants,
    fit_full_model,
    build_complex_model,
)
from .plotting import (
    set_latex_style,
    make_final_figure,
    build_plot_data,
)
from .agnostic import (
    fit_wkb_amplitude,
    fit_amplitude_tabulated,
    load_kerr_qnm,
    WKB_KEYS,
)


@dataclass
class GreyRingResult:
    """
    Container for the output of a GreyRing fit.
    """
    sim_number: int
    ell: int
    m: int

    M_final: float
    chi_final: float

    omega_x: float
    omega_i: float
    omega_f: float

    A: float
    p: float
    a: float
    b: float
    c: float

    mismatch: float
    output_file: str | None = None


def fit(
    sim_number,
    ell,
    m,
    abs_file,
    phase_file,
    omega_theory,
    omega_i_factor=0.7,
    omega_f_amp_ratio=20.0,
    omega_max_search=1.0,
    left_plot=0.15,
    make_plot=True,
    output_file=None,
    use_latex=True,
):
    """
    Fit one SXS multipole with the GreyRing frequency-domain model.

    Parameters
    ----------
    sim_number : int
        SXS simulation number, e.g. 3617.
    ell, m : int
        Multipole indices.
    abs_file : str
        Path to the file containing |R_lm| or |Zed_lm|.
    phase_file : str
        Path to the file containing arg(R_lm) or arg(Zed_lm).
    omega_theory : array_like
        Frequency array corresponding to the reflectivity values in
        abs_file and phase_file.
    omega_i_factor : float, optional
        Defines omega_i = omega_i_factor * omega_x.
    omega_f_amp_ratio : float, optional
        Defines omega_f as the first frequency after omega_x where
        |h(omega_f)| <= |h(omega_x)| / omega_f_amp_ratio.
    omega_max_search : float, optional
        Maximum dimensionless frequency used in the SXS FFT.
    left_plot : float, optional
        Left boundary of the plot.
    make_plot : bool, optional
        If True, build and save the amplitude/phase plot.
    output_file : str or None, optional
        Name of the output plot file. If None, a default name is used.
    use_latex : bool, optional
        If True, use the matplotlib LaTeX style defined in plotting.py.

    Returns
    -------
    GreyRingResult
        Best-fit parameters, remnant data, selected frequency interval,
        mismatch, and output plot filename.
    """
    if use_latex:
        set_latex_style()

    if output_file is None:
        output_file = f"greyring_fit_SXS{sim_number}_l{ell}m{m}.pdf"

    (
        omega_fit,
        H_fit,
        omega_all,
        H_all,
        M_final,
        chi_final,
        omega_x,
        omega_i,
        omega_f,
    ) = prepare_fit_data(
        sim_number=sim_number,
        omega_max_search=omega_max_search,
        ell=ell,
        m=m,
        omega_i_factor=omega_i_factor,
        ratio_factor=omega_f_amp_ratio,
    )

    f_abs, f_phase = build_theory_interpolants(
        sim_number=sim_number,
        abs_file=abs_file,
        phase_file=phase_file,
        omega_theory=omega_theory,
        ell=ell,
        m=m,
    )

    fit_data = fit_full_model(
        omega_fit=omega_fit,
        H_fit=H_fit,
        f_abs=f_abs,
        f_phase=f_phase,
        M_final=M_final,
    )

    H_model = build_complex_model(
        omega=omega_fit,
        f_abs=f_abs,
        f_phase=f_phase,
        M_final=M_final,
        fit_data=fit_data,
    )

    mismatch = mismatch_complex(omega_fit, H_fit, H_model)

    if make_plot:
        plot_data = build_plot_data(
            omega_all=omega_all,
            H_all=H_all,
            omega_fit=omega_fit,
            omega_f=omega_f,
            left_plot=left_plot,
            f_abs=f_abs,
            f_phase=f_phase,
            M_final=M_final,
            fit_data=fit_data,
        )

        make_final_figure(
            sim_number=sim_number,
            ell=ell,
            m=m,
            omega_plot=plot_data["omega_plot"],
            abs_num_plot=plot_data["abs_num_plot"],
            amp_fit_plot=plot_data["amp_fit_plot"],
            phi_th_det_plot=plot_data["phi_th_det_plot"],
            phi_sxs_det_plot=plot_data["phi_num_det_plot"],
            phi_fit_plot=plot_data["phi_fit_plot"],
            omega_i=omega_i,
            omega_f=omega_f,
            omega_x=omega_x,
            left_plot=left_plot,
            output_file=output_file,
        )
    else:
        output_file = None

    return GreyRingResult(
        sim_number=sim_number,
        ell=ell,
        m=m,
        M_final=M_final,
        chi_final=chi_final,
        omega_x=omega_x,
        omega_i=omega_i,
        omega_f=omega_f,
        A=fit_data["A_fit"],
        p=fit_data["p_fit"],
        a=fit_data["a_fit"],
        b=fit_data["b_fit"],
        c=fit_data["c_fit"],
        mismatch=mismatch,
        output_file=output_file,
    )


@dataclass
class WKBFitResult:
    """Container for an amplitude-only WKB fit."""
    sim_number: int
    ell: int
    m: int

    M_final: float
    chi_final: float

    omega_x: float
    omega_i: float
    omega_f: float

    num_wkb: int
    wkb_params: dict = field(default_factory=dict)

    A: float = 0.0
    p: float = 0.0

    mismatch: float = 1.0


def fit_wkb(
    sim_number: int,
    ell: int,
    m: int,
    num_wkb: int,
    theory_dir: str | Path | None = None,
    x0_wkb: list[float] | None = None,
    omega_i_factor: float = 0.7,
    omega_f_amp_ratio: float = 20.0,
    omega_max_search: float = 1.0,
) -> WKBFitResult:
    """Fit one SXS multipole with a WKB greybody amplitude model.

    Fits only |h(omega)|.  WKB parameters and phenomenological (A, p)
    are determined jointly via profile likelihood + Nelder-Mead.

    Parameters
    ----------
    sim_number : int
        SXS simulation number.
    ell, m : int
        Multipole indices.
    num_wkb : int
        WKB order (1, 2, or 3).
    theory_dir : path, optional
        Directory containing QNM data files (l{ell}/n{n}l{ell}m{m}.dat).
        Defaults to examples/sxs_fit/theory/ inside the GreyRing package.
    x0_wkb : list, optional
        Initial guess for WKB parameters.  If None, Kerr QNM values
        are loaded from theory_dir.
    omega_i_factor : float
        Factor defining omega_i = omega_i_factor * omega_x.
    omega_f_amp_ratio : float
        Amplitude ratio for omega_f selection.
    omega_max_search : float
        Maximum dimensionless frequency in the SXS FFT.

    Returns
    -------
    WKBFitResult
    """
    if theory_dir is None:
        theory_dir = Path(__file__).resolve().parent.parent / "examples" / "sxs_fit" / "theory"
    theory_dir = Path(theory_dir)

    import os
    saved_cwd = os.getcwd()
    try:
        os.chdir(str(theory_dir.parent))

        (
            omega_fit, H_fit, omega_all, H_all,
            M_final, chi_final, omega_x, omega_i, omega_f,
        ) = prepare_fit_data(
            sim_number=sim_number,
            omega_max_search=omega_max_search,
            ell=ell,
            m=m,
            omega_i_factor=omega_i_factor,
            ratio_factor=omega_f_amp_ratio,
        )
    finally:
        os.chdir(saved_cwd)

    if x0_wkb is None:
        qnm = load_kerr_qnm(chi_final, ell, m, theory_dir)
        keys = WKB_KEYS[num_wkb]
        x0_wkb = [qnm[k] for k in keys]

    result = fit_wkb_amplitude(omega_fit, H_fit, num_wkb, x0_wkb)

    return WKBFitResult(
        sim_number=sim_number,
        ell=ell,
        m=m,
        M_final=M_final,
        chi_final=chi_final,
        omega_x=omega_x,
        omega_i=omega_i,
        omega_f=omega_f,
        num_wkb=num_wkb,
        wkb_params=result["wkb_params"],
        A=result["A"],
        p=result["p"],
        mismatch=result["mismatch"],
    )
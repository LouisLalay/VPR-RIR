from constants import (
    ALMOST_ZERO,
    DELTA_MODE_MEAN,
    DELTA_MODE_PERCENT,
    DELTA_MODE_SUM,
    METRICS_DELTA_MODE,
    METRICS_NAMES,
)
from scipy.signal import find_peaks
from signal_processing_utils import linear_regression
from torch import Tensor
import pandas as pd
import torch


def to_dB(x: Tensor) -> Tensor:
    """
    Returns the dB value of x. No preprocessing is done.

    Args
    ----
        x: Tensor
            Input tensor.
    Returns
    -------
        x_dB: Tensor:
            dB value of x.
    """
    return 10 * torch.log10(x)


def find_first_peak(power: Tensor, sr: int) -> tuple[int, int]:
    """
    Finds the first peak using scipy.

    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        first_peak_index: int
            Index of the first peak.
        half_width: int
            Half width of the peak.
    """
    # Ignore noisy peaks
    max_power = power.max()
    power = (power - 0.5 * max_power).clamp(0)

    # Ignore peaks closer than 0.5 ms => 17cm
    minimal_distance_between_peaks = int(0.0005 * sr)
    peaks, properties = find_peaks(power, distance=minimal_distance_between_peaks)
    match len(peaks):
        case 0:
            # No peaks are found, either because the signal
            # is constant or because the peak is at 0 index
            if power.min() == power.max():
                # Signal is constant, no first peak
                return torch.tensor(0.0)
            else:
                # The max is the only peak
                first_peak_index = power.argmax()
                half_width = minimal_distance_between_peaks
        case 1:
            # Only one peak is found
            first_peak_index = peaks[0]
            half_width = minimal_distance_between_peaks
        case _:
            # General case
            first_peak_index = peaks[0]
            half_width = int((peaks[1] - peaks[0]) / 2)

    return first_peak_index, half_width


def c80(rir: Tensor, sr: int) -> Tensor:
    """
    Computes the clarity at 80 ms.

    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        c80: Tensor
            Clarity at 80 ms in dB.
    """

    # Find the sample index corresponding to 80 ms
    index_80_ms = int(0.08 * sr)

    # Compute the energy in the first 80 ms and the total energy
    energy_before = rir[:index_80_ms] @ rir[:index_80_ms]
    energy_after = rir[index_80_ms:] @ rir[index_80_ms:]

    c80 = to_dB(energy_before.clamp(min=ALMOST_ZERO)) - to_dB(
        energy_after.clamp(min=ALMOST_ZERO)
    )
    return c80


def d50(rir: Tensor, sr: int) -> Tensor:
    """
    Computes the Definition at 50 ms.

    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        d50: Tensor
            Definition at 50 ms in dB.
    """
    # Find the sample index corresponding to 50 ms
    index_50_ms = int(0.05 * sr)

    # Compute the energy in the first 50 ms and the total energy
    energy_before = rir[:index_50_ms] @ rir[:index_50_ms]
    energy_total = rir @ rir

    d50 = to_dB(energy_before.clamp(min=ALMOST_ZERO)) - to_dB(
        energy_total.clamp(min=ALMOST_ZERO)
    )
    return d50


def drr(rir: Tensor, sr: int) -> Tensor:
    """
    Computes the Direct-to-Reverberant Ratio.

    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        drr: Tensor
            Direct-to-Reverberant Ratio in dB.
    """
    power = rir.square()
    # Find the first peak
    first_peak_index, half_width = find_first_peak(power.cpu(), sr)
    # Select the energy around the first peak
    start = max(0, first_peak_index - half_width)
    end = min(first_peak_index + half_width, power.size(0))

    direct_energy = power[start:end].sum()
    reverberant_energy = power.sum() - direct_energy

    drr = to_dB(direct_energy.clamp(min=ALMOST_ZERO)) - to_dB(
        reverberant_energy.clamp(min=ALMOST_ZERO)
    )
    return drr


def edc(rir: Tensor, sr: int) -> Tensor:
    """
    Computes the Energy Decay Curve.

    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        edc: Tensor
            Energy Decay Curve.
    """
    power = rir.square()
    edc = power.flip(0).cumsum(0).flip(0)
    edc = edc / edc[0]

    return edc


def edr(rir: Tensor, sr: int, nfft: int = 512) -> Tensor:
    """
    Computes the Energy Decay Relief.
    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        edr: Tensor
            Energy Decay Relief in dB.
    """
    stft = torch.stft(
        rir,
        n_fft=nfft,
        hop_length=nfft // 2,
        window=torch.hann_window(nfft).to(rir),
        return_complex=True,
    )
    power = stft.real.square() + stft.imag.square()
    edr = power.flip(-1).cumsum(-1).flip(-1).clamp(min=1e-10)

    return to_dB(edr)


def rt30(rir: Tensor, sr: int) -> float:
    """
    Computes the RT30.
    Args
    ----
        rir: Tensor
            Impulse response.
        sr: int
            Sampling rate.
    Returns
    -------
        rt30: Tensor
            RT30 in s
    """

    Lh = rir.shape[0]
    log_edc = to_dB(edc(rir, sr).clamp(min=1e-10))

    win_size = Lh // 25
    hop_size = win_size // 2

    x = torch.arange(win_size).to(rir)
    angles = []

    # Find the angles along the EDC
    for i in range(0, Lh - win_size + 1, hop_size):
        # Compute the linear regression
        a, b = linear_regression(x + i, log_edc[i : i + win_size])
        angles.append(a)

    # Compute the derivative of the angles to find
    # where the decay is linear in log scale
    angle_derivative = torch.diff(torch.tensor(angles))

    # Select the longest segment of 0s in the derivative
    # where the slope is nearly constant
    threshold = 0.1 * angle_derivative.abs().max()
    start = 0
    l = 0
    max_l = 0
    for k, d_a in enumerate(angle_derivative):
        if d_a.abs() > threshold:
            start = k
            l = 0
        else:
            l += 1
            if l > max_l:
                max_l = l
                bounds = (start, k)

    # Transform slices idx to time steps
    start = int((bounds[0] + 1) * hop_size)
    end = int((bounds[1] + 1) * hop_size)

    # First segment - Before exponential decay
    a, b = linear_regression(torch.arange(start).to(rir), log_edc[:start])
    if b < -30:
        # First segment is already under -30 dB
        rt30_1 = 0
    elif torch.isclose(torch.tensor(a), torch.zeros(1)):
        # First segment is flat and above -30 dB
        # Probably due to direct path being long
        rt30_1 = Lh
    else:
        # General case, direct path negligible and exponential decay
        # Estimate RT30 from the linear regression
        rt30_1 = abs((-b - 30) / a)

    # Second segment - Exponential decay
    a, b = linear_regression(torch.arange(start, end).to(rir), log_edc[start:end])
    if torch.isclose(torch.tensor(a), torch.zeros(1)):
        # Second segment is flat
        if b < -30:
            # First segment probably already crossed -30 dB
            rt30_2 = start
        else:
            # Either no decay or noise floor is too high
            rt30_2 = Lh
    else:
        # General case when the direct path is not negligible
        rt30_2 = abs((-b - 30) / a)

    rt30_idx = int(min(rt30_1, rt30_2))
    rt30_idx = min(rt30_idx, Lh)

    if rt30_idx == Lh:
        return Lh / sr
    else:
        arg_max = rir.cpu().abs().argmax().item()
        return max((rt30_idx - arg_max) / sr, 0.0)


def mse(ref: Tensor, est: Tensor) -> float:
    """
    Computes the Mean Squared Error between two tensors.

    Args
    ----
        ref: Tensor
            First tensor, shape (N)
        est: Tensor
            Second tensor, shape (N)
    Returns
    -------
        mse: Float
            Mean Squared Error between ref and est in %.
    """
    return 100 * (ref - est).square().mean().item() / ref.square().mean().item()


def delta_m(ref: float | Tensor, est: float | Tensor, mode: str) -> float:
    if mode == DELTA_MODE_PERCENT:
        return 100 * abs(ref - est) / ref
    elif mode == DELTA_MODE_MEAN:
        return (ref - est).abs().mean().item()
    elif mode == DELTA_MODE_SUM:
        return (ref - est).abs().sum().item()
    else:
        return abs(ref - est)


def evaluate(rir: Tensor, sr: int) -> dict[str, float | Tensor]:
    """
    Build a dictionary containing the metrics of the RIR.
    The keys are the full names of the metrics.

    Args
    ----
        rir: Tensor
            Impulse response, shape (Lh)
        sr: int
            Sampling rate.
    Returns
    -------
        metrics: dict
            Dictionary containing the metrics of the RIR.
    """
    metrics = {}
    rir = rir.detach()

    r = c80(rir, sr)
    metrics[METRICS_NAMES["c80"]] = r.item()

    r = d50(rir, sr)
    metrics[METRICS_NAMES["d50"]] = r.item()

    r = drr(rir, sr)
    metrics[METRICS_NAMES["drr"]] = r.item()

    r = edc(rir, sr)
    metrics[METRICS_NAMES["edc"]] = r.cpu()

    r = edr(rir, sr)
    metrics[METRICS_NAMES["edr"]] = r.cpu()

    r = rt30(rir, sr)
    metrics[METRICS_NAMES["rt30"]] = r

    return metrics


def compare(rir_ref: Tensor, rir_est: Tensor, sr: int) -> dict[str, float]:
    """
    Compares two RIRs and returns the differences in metrics.
    Truncate the RIRs to the same length before comparison.

    Args
    ----
        rir_ref: Tensor
            Reference impulse response, shape (Lh_1)
        rir_est: Tensor
            Estimated impulse response, shape (Lh_2)
        sr: int
            Sampling rate.
    Returns
    -------
        diff: dict
            Dictionary containing the differences in metrics.
    """
    # Truncate the RIRs to the same length
    rir_est = rir_est.to(rir_ref.device)
    Lh = min(rir_ref.shape[0], rir_est.shape[0])
    rir_ref = rir_ref[:Lh]
    rir_est = rir_est[:Lh]

    m_ref = evaluate(rir_ref, sr)
    m_est = evaluate(rir_est, sr)

    diff = {
        METRICS_NAMES[key]: delta_m(
            m_ref[METRICS_NAMES[key]],
            m_est[METRICS_NAMES[key]],
            METRICS_DELTA_MODE[key],
        )
        for key in METRICS_NAMES.keys()
        if METRICS_NAMES[key] in m_ref
    }
    diff[METRICS_NAMES["mse"]] = mse(rir_ref, rir_est)

    return diff

from matplotlib import pyplot as plt
from scipy.signal import freqz
from torch import Tensor, tensor, exp
import numpy as np


def filter_plot(
    denominator: Tensor,
    numerator: Tensor,
    axis: plt.Axes,
    label: str = None,
):
    if numerator is None:
        numerator = tensor(1)
    freq, transfer = freqz(
        b=numerator.detach().cpu(),
        a=denominator.detach().cpu(),
        fs=1,
    )
    mod = np.abs(transfer)
    mod[mod == 0] = np.partition(mod, 1)[1]
    mod_dB = 20 * np.log10(mod)

    axis.plot(freq, mod_dB, label=label)
    axis.grid()
    axis.set_xlabel("Frequency (normalized)")
    axis.set_ylabel("Amplitude (dB)")
    axis.set_title("Filter g")


def filter_fig(g: Tensor, g0_inv: Tensor, a: Tensor, p: Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    g_axis: plt.Axes = axes[0]
    p_axis: plt.Axes = axes[1]
    filter_plot(g, g0_inv, g_axis)
    g_axis.set_title("Filter g")
    filter_plot(p * exp(a), tensor(1), p_axis)
    p_axis.set_title("Filter p * exp(a)")

    return fig


def rir_fig(mu_h: Tensor, r_h: Tensor, sr: int, ref_rir: Tensor = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mu_h_axis: plt.Axes = axes[0]
    r_h_axis: plt.Axes = axes[1]

    time = np.arange(mu_h.shape[0]) / sr

    if ref_rir is not None:
        mu_h_axis.plot(time, mu_h.detach().cpu(), label="mu_h")
        mu_h_axis.plot(time, ref_rir.detach().cpu(), "C3", label="Reference", alpha=0.5)
        mu_h_axis.set_title("Estimated RIR vs Reference")
        mu_h_axis.legend()
    else:
        mu_h_axis.plot(time, mu_h.detach().cpu())
        mu_h_axis.set_title("Estimated RIR")

    mu_h_axis.set_xlabel("Time (s)")
    mu_h_axis.set_ylabel("Amplitude")
    mu_h_axis.grid()

    r_h_axis.plot(time, r_h.detach().cpu())
    r_h_axis.set_title("Variance")
    r_h_axis.set_xlabel("Time (s)")
    r_h_axis.set_ylabel("Variance")
    r_h_axis.grid()

    return fig

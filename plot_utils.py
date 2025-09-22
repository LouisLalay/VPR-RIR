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
    axis.set_title("Filter $g$")


def filter_fig(g: Tensor, g0_inv: Tensor, a: Tensor, p: Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    g_axis: plt.Axes = axes[0]
    p_axis: plt.Axes = axes[1]
    filter_plot(g, g0_inv, g_axis)
    g_axis.set_title("Filter g")
    filter_plot(p * exp(a), tensor(1), p_axis)
    p_axis.set_title(r"Filter $p \cdot e^{a}$")
    fig.tight_layout()

    return fig


def rir_fig(mu_h: Tensor, r_h: Tensor, sr: int, ref_rir: Tensor = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mu_h_axis: plt.Axes = axes[0]
    r_h_axis: plt.Axes = axes[1]

    time = np.arange(mu_h.shape[0]) / sr

    if ref_rir is not None:
        mu_h_axis.plot(time, mu_h.detach().cpu(), label="mu_h")
        mu_h_axis.plot(
            np.arange(ref_rir.shape[0]) / sr,
            ref_rir.detach().cpu(),
            "C3",
            label="Reference",
            alpha=0.5,
        )
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
    fig.tight_layout()

    return fig


def rir_spectrograms(
    mu_h: Tensor,
    generated_rir: Tensor,
    sr: int,
    ref_rir: Tensor = None,
):
    if ref_rir is not None:
        rirs = {
            "Estimated RIR": mu_h,
            "Reference RIR": ref_rir,
            "Generated RIR": generated_rir,
        }
    else:
        rirs = {
            "Estimated RIR": mu_h,
            "Generated RIR": generated_rir,
        }

    axes: list[plt.Axes]
    fig, axes = plt.subplots(1, len(rirs), figsize=(len(rirs) * 6, 5))

    nfft = 1024 if mu_h.shape[0] > 1024 else mu_h.shape[0] // 10
    for k, (key, h) in enumerate(rirs.items()):
        axis = axes[k]
        _, _, _, cax = axis.specgram(
            (h / h.abs().max()).detach().cpu().numpy(),
            Fs=sr,
            NFFT=nfft,
            noverlap=nfft // 2,
            cmap="inferno",
            scale="dB",
            vmin=-80,
        )
        axis.set_title(key)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")
        fig.colorbar(cax, ax=axis, label="Amplitude (dB)")
        fig.tight_layout()

    return fig


def generated_rir_fig(generated_rir: Tensor, sr: int, reference_rir: Tensor = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    time = np.arange(generated_rir.shape[0]) / sr

    if reference_rir is not None:
        ax.plot(time, generated_rir.detach().cpu(), label="Generated RIR")
        ax.plot(
            np.arange(reference_rir.shape[0]) / sr,
            reference_rir.detach().cpu(),
            "C3",
            label="Reference RIR",
            alpha=0.5,
        )
        ax.set_title("Generated RIR vs Reference RIR")
        ax.legend()
    else:
        ax.plot(time, generated_rir.detach().cpu())
        ax.set_title("Generated RIR")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid()
    fig.tight_layout()

    return fig

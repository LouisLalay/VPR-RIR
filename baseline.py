from constants import BASELINE_AUDIO_NAMES, MODEL_NAMES
from crossband_filtering import CTF
from pathlib import Path
from torch import Tensor
from torchaudio import save, load
import torch


def dec(source: Tensor, reverberant_signal: Tensor) -> Tensor:
    """Deconvolution of the reverberant signal with the source signal.

    Args
    ----
    source: Tensor
        Known dry source signal
    reverberant_signal: Tensor
        Mesured signal with echoes

    Returns
    -------
    h_est: Tensor
        Estimated RIR
    """
    X = torch.fft.fft(source, reverberant_signal.shape[0])
    Y = torch.fft.fft(reverberant_signal)

    H_est = Y / X
    Lh = reverberant_signal.shape[0] - source.shape[0] + 1

    return torch.fft.ifft(H_est)[:Lh].real


def cbf(source: Tensor, reverberant_signal: Tensor) -> Tensor:
    """Adapted from https://github.com/Louis-Bahrman/SD-cRIRc

    Args
    ----
    source: Tensor
        Dry source signal
    reverberant_signal: Tensor
        Mesured signal with echoes

    Returns
    -------
    h_est: Tensor
        Estimated RIR
    """
    nfft = 512
    Y = (
        torch.stft(
            reverberant_signal,
            n_fft=nfft,
            hop_length=nfft // 2,
            win_length=nfft,
            window=torch.hann_window(nfft).to(reverberant_signal),
            return_complex=True,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    S = (
        torch.stft(
            source,
            n_fft=nfft,
            hop_length=nfft // 2,
            win_length=nfft,
            window=torch.hann_window(nfft).to(source),
            return_complex=True,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    L_h = reverberant_signal.size(0) - source.size(0) + 1
    dummy = torch.stft(
        torch.zeros(L_h),
        n_fft=nfft,
        hop_length=nfft // 2,
        win_length=nfft,
        window=torch.hann_window(nfft),
        return_complex=True,
    )
    # Use one crossband, meaning f-1, f and f+1 are used
    ctf_instance = CTF(output_len=dummy.shape[-1] + 1, num_absolute_cross_bands=1)

    H_est = ctf_instance.forward(Y, S).squeeze()
    h_est: Tensor = torch.istft(
        H_est,
        nfft,
        nfft // 2,
        nfft,
        torch.hann_window(nfft).to(H_est.device),
    )
    return h_est[:L_h]


# Add the corresponding full name in the constants.py file
# when adding a new baseline function
# Be sure the keys in BASELINES_NAMES match the keys in BASELINES_FUNCTIONS
BASELINES_FUNCTIONS = {
    "dec": dec,
    "cbf": cbf,
}


def compute_baselines(
    source: Tensor,
    reverberant_signal: Tensor,
    sr: int,
    run_path: Path,
):
    """
    Compute the estimated RIR using the baseline methods and save them to the run path.

    Args
    ----
    source: Tensor
        Known dry source signal
    reverberant_signal: Tensor
        Mesured signal with echoes
    sr: int
        Sampling rate of the audio signals
    run_path: Path
        Path to the run folder where the estimated RIRs will be saved
    """
    for key, func in BASELINES_FUNCTIONS.items():
        filename = run_path / BASELINE_AUDIO_NAMES[key]
        if filename.exists():
            continue
        estimated_rir = func(source, reverberant_signal).unsqueeze(0)
        save(filename, estimated_rir, sample_rate=sr)


def load_baselines(run_path: Path) -> dict:
    """
    Load the estimated RIRs from the run path.

    Args
    ----
    run_path: Path
        Path to the run folder where the estimated RIRs are saved

    Returns
    -------
    dict
        A dictionary with the estimated RIRs for each baseline method
    """
    baselines = {}
    for key in BASELINE_AUDIO_NAMES.keys():
        filename = run_path / BASELINE_AUDIO_NAMES[key]
        assert filename.exists(), f"Baseline file {filename} does not exist."
        baselines[MODEL_NAMES[key]] = load(filename)[0].squeeze()
    return baselines

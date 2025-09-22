from constants import ALMOST_ZERO
from metrics import edc, to_dB
from signal_processing_utils import linear_regression
from torch import Tensor
from torch.nn import Module
from torchaudio.transforms import Spectrogram
from tqdm import tqdm
from typing import Callable
import torch


class RT60Estimator(Module):
    def __init__(self, sr: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Minimal number of frames for a decay to be considered
        self.minimal_decay_width = 3

        self.sr = sr
        # 50 ms window
        self.window_size = int(50e-3 * sr)
        self.n_overlap = self.window_size // 4
        self.hop_size = self.window_size - self.n_overlap
        self.nfft = 1 << (self.window_size - 1).bit_length()
        # Limit to 4000 Hz
        # TODO: verify if it is necessary
        self.n_freqs_max = self.sr // 2
        # Only look for pack of windows around 1 second
        self.L_max = int(1.0 * sr / self.hop_size)

        spectro = Spectrogram(
            n_fft=self.nfft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            power=2.0,
        )
        self.power_spectro_function: Callable[[Tensor], Tensor] = lambda x: spectro.to(
            x.device
        )(x)

    def detect_FDR(
        self, power_spectrogram: torch.Tensor
    ) -> dict[int, list[tuple[int, int]]]:
        """
        Detects the free decay regions from the stft of a reverberant signal.
        Returns zones of the signal where FDR(s) are detected.

        Args
        --
            power_spectrogram: (K, T) power spectrogram of the reverberant signal

        Returns
        --
            fdr: (K, T') power spectrogram windows where energy is strictly decreasing

        Note
        --
            Unfold: windows_{L-1} # (K, T - L + 2, L-1) is included in windows_{L} # (K, T - L + 1, L)
            except for the last window which we have to compute. windows_{L-1} = windows_{L}[...,:-1] + the last window.
            Compare the speed of unfold to this method.

        """
        processed_freqs = set()
        fdr_zones: dict[int, list[tuple[int, int]]] = {}

        # Take a window size of L frames, decreasing from T to minimal_decay_width
        for L in range(self.L_max, self.minimal_decay_width - 1, -1):
            # Compute the sliding windows of size L with a hope of 1 frame
            # The resulting tensor is of size (K, T - L + 1, L)
            # One window contains L stft frames
            windows = power_spectrogram.unfold(dimension=1, size=L, step=1)

            # Compute the diff along the last dimension to compare consecutive windows
            diffs = windows.diff(dim=-1)  # (K, T - L + 1, L - 1)

            # If the difference is negative in all the L-1 differences, then we have a FDR
            is_fdr = (diffs < 0).all(dim=-1)  # (K, T - L + 1) boolean

            # As we keep the frequency axis untouched, we can see which frequencies have FDRs
            # Store the STFT parts - keep only the largest window that has a FDR
            for f, start_idx in is_fdr.nonzero():
                f = int(f.item())
                start_idx = int(start_idx.item())
                if f not in processed_freqs:
                    if L not in fdr_zones.keys():
                        fdr_zones[L] = []
                    fdr_zones[L].append((f, start_idx))
            # Add the frequency afterwards, because the more windows the better
            for f, start_idx in is_fdr.nonzero():
                processed_freqs.add(int(f.item()))

            # Loop until all frequencies have a FDR
            if len(processed_freqs) == self.n_freqs_max:
                break
        self.processed_freqs = processed_freqs
        return fdr_zones

    def rt60_from_stft_frame(
        self,
        stft_frame: Tensor,
        min_dB_decrease: int = 10,
    ) -> float:
        """
        Computes the RT60 time from a single STFT frame

        Args
        --
            stft_frame: (T,) single frequency bin STFT frame
            min_dB_decrease: only perform regression for edc under this value

        Returns
        --
            rt60: float RT60 estimate in seconds
        """
        # Time frequency domain energy decay curve
        # TODO: change the args to *args, **kwargs
        log_edc = to_dB(edc(stft_frame, 0))
        # By construction -edc is sorted and increasing
        # Find the first index below the min_dB_decrease threshold
        idx = torch.searchsorted(-log_edc, min_dB_decrease)
        regression_idx = torch.arange(
            idx.item(), log_edc.shape[0], device=log_edc.device
        )
        if len(regression_idx) < 2:
            return idx.item() * self.hop_size / self.sr
        slope, bias = linear_regression(regression_idx, log_edc[idx:])
        if slope < ALMOST_ZERO:
            rt = log_edc.shape[0] * self.hop_size / self.sr
        else:
            rt = -60.0 / slope * self.hop_size / self.sr
        return rt

    def get_all_rt60s(
        self,
        fdr_mask: dict[int, list[tuple[int, int]]],
        cropped_power: Tensor,
    ) -> dict[int, list[float]]:
        """
        Get the RT60 estimates for each frequency band.

        Args
        --
            fdr_mask: dict of FDR zones detected
            cropped_power: (K, T) power spectrogram cropped to 4000 Hz
        Returns
        --
            all_rt60s: dict of RT60 estimates for each frequency band
        """
        # RT60s for each window where a FDR was detected
        all_rt60s = {f: [] for f in self.processed_freqs}
        for L, zones in fdr_mask.items():
            for f, start_idx in zones:
                stft_frame = cropped_power[f, start_idx : start_idx + L]
                rt60 = self.rt60_from_stft_frame(stft_frame)
                all_rt60s[f].append(rt60)
        return all_rt60s

    @staticmethod
    def smooth_rt60s(rt60s: Tensor, target_n_bands: int) -> Tensor:
        """
        Smooth the RT60 estimates to the target number of bands using averaging.

        Args
        --
            rt60s: (K,) per-band RT60 estimates
            target_n_bands: int target number of bands
        Returns
        --
            smoothed_rt60s: (target_n_bands,) smoothed per-band RT60 estimates
        """
        if target_n_bands >= rt60s.shape[0]:
            return rt60s
        factor = rt60s.shape[0] / target_n_bands
        smoothed_rt60s = torch.zeros(target_n_bands, device=rt60s.device)
        for k in range(target_n_bands):
            start = int(k * factor)
            end = int((k + 1) * factor)
            smoothed_rt60s[k] = rt60s[start:end].mean()
        return smoothed_rt60s

    def forward(self, reverberant_signal: Tensor) -> Tensor:
        """
        Get the RT60 estimates for each frequency band.
        Args
        --
            reverberant_signal: (N,) reverberant signal
        Returns
        --
            rt60s: (K,) per-band RT60 estimates
        """
        power_spectrogram = self.power_spectro_function(reverberant_signal)
        # Limit to 4000 Hz
        cropped_power = power_spectrogram[: self.n_freqs_max]
        fdr_mask = self.detect_FDR(cropped_power)
        all_rt60s = self.get_all_rt60s(fdr_mask, cropped_power)
        rt60s = torch.zeros(self.n_freqs_max)
        for f, rts in all_rt60s.items():
            rt60s[f] = torch.tensor(rts).median()

        for f in set(range(self.n_freqs_max)) - self.processed_freqs:
            if f == 0:
                rt60s[f] = rt60s[:2].mean()
            elif f == self.n_freqs_max - 1:
                rt60s[f] = rt60s[-2:].mean()
            else:
                rt60s[f] = rt60s[f - 1 : f + 2].mean()
        return rt60s


def test():
    print("Initialisation")
    from signal_processing_utils import rt_to_ap_coeffs, ap_coefs_to_rt
    from time import time
    from torchaudio import load
    from torchaudio.functional import convolve, resample
    import matplotlib.pyplot as plt

    source, sr_source = load("data/example_audios/source.flac")
    rir, sr_rir = load("data/example_audios/rir.wav")
    sr = min(sr_source, sr_rir)

    rir = resample(rir[0] / rir[0].abs().max(), sr_rir, sr)
    source = resample(source[0], sr_source, sr)
    y = convolve(source, rir)

    rt60_estimator = RT60Estimator(sr=sr)
    print("Estimating RT60")
    tik = time()
    rt60 = rt60_estimator.forward(y)
    tok = time()
    print(f"RT60 estimation took {tok - tik:.2e} s")
    mean_rt60 = rt60.median()
    print(f"Estimated RT60: {mean_rt60:.3f} s")

    Lp = 10
    a, p = rt_to_ap_coeffs(rt60, sr, Lp=Lp)
    print(f"a: {a}, p[0]: {p[0]}")
    rt60_reconstructed = ap_coefs_to_rt(a, p, sr, rt60.shape[0])

    plt.plot(rt60.cpu(), label="Estimated")
    plt.plot(
        torch.linspace(0, len(rt60), Lp),
        rt60_estimator.smooth_rt60s(rt60, Lp).cpu(),
        label="Smoothed",
    )
    plt.plot(rt60_reconstructed.cpu(), label="Reconstructed")
    plt.legend()
    plt.show()

    plt.plot(torch.arange(rir.shape[0]) / sr, rir.cpu())
    plt.vlines(mean_rt60, -1, 1, color="tab:red")
    plt.title(f"Estimated RT60: {mean_rt60:.3f} s")
    plt.xlabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    test()

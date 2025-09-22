from pandas import DataFrame, read_csv
from pathlib import Path
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchaudio import load
from torchaudio.functional import convolve, resample
import torch


class SimpleDataset(Dataset):
    """
    Load audios from a list of absolute paths.
    """

    def __init__(self, metadata_file: Path, sr: int = 16000, **kwargs):
        """
        Args
        ----
        metadata_file: Path
            The path to the CSV file containing at least the column "file_path".
            The complete row is returned along with the audio.
        sr: int
            The sampling rate to resample the audio files to.
        """
        super().__init__()
        self.df = read_csv(metadata_file)
        self.df["sampling_rate"] = sr
        self.target_sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple[Tensor, DataFrame]:
        df_item = self.df.iloc[index]
        x, original_sr = load(df_item["file_path"])
        return (
            resample(
                x[0],
                orig_freq=original_sr,
                new_freq=self.target_sr,
            ).to(torch.get_default_dtype()),
            df_item,
        )


def prepare_audios(
    source: Tensor,
    rir: Tensor,
    noise: Tensor,
    Lh: int,
    snr_dB: float = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Pad RIR to Lh
    rir = torch.nn.functional.pad(
        rir,
        (0, Lh - rir.shape[0]),
        mode="constant",
        value=0,
    )

    y = convolve(source, rir, mode="full")
    # Scale the reverberant signal to compensate for RIR or source energy
    scale = y.abs().max() * 5 / 4
    y = y / scale

    noise = pad(
        noise,
        (0, y.shape[0] - noise.shape[0]),
        mode="constant",
        value=0,
    )

    # We adapt the noise power to the desired SNR
    sigma_noise_adapted = ((y @ y) / (noise @ noise)).sqrt() * 10 ** (-snr_dB / 20)
    noise = noise * sigma_noise_adapted

    noisy_y = y + noise

    return rir / scale, noise, y, noisy_y


def mixing_model(
    source: Tensor,
    rir: Tensor,
    noise: Tensor,
    snr_dB: float,
    sr: int,
    max_duration_s: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Returns y = source * rir + noise adapted to the desired SNR.
    Truncate y to the desired max duration
    """
    # Reverberant signal
    y = convolve(source, rir, mode="full")

    max_samples = int(max_duration_s * sr)
    y = y[:max_samples]

    noise = pad(
        noise,
        (0, max_samples - noise.shape[0]),
        mode="constant",
        value=0,
    )

    noise_power = (noise @ noise) / noise.shape[0]
    signal_power = (y @ y) / max_samples
    target_noise_power = signal_power * 10 ** (-snr_dB / 10)

    noise = noise * (target_noise_power / noise_power).sqrt()

    noisy_y = y + noise

    return noise, y, noisy_y


def test():
    import matplotlib.pyplot as plt

    source_file = Path("data/example_audios/source.flac")
    rir_file = Path("data/example_audios/rir.wav")
    noise_file = Path("data/example_audios/noise.wav")

    source, sr1 = load(source_file)
    rir, sr2 = load(rir_file)
    noise, sr3 = load(noise_file)

    sr = min(sr1, sr2, sr3)
    source = resample(source[0], orig_freq=sr1, new_freq=sr)
    rir = resample(rir[0], orig_freq=sr2, new_freq=sr)
    noise = resample(noise[0], orig_freq=sr3, new_freq=sr)

    max_duration = 4.0
    snr_dB = 0

    noise, y, noisy_y = mixing_model(
        source=source,
        rir=rir,
        noise=noise,
        snr_dB=snr_dB,
        sr=sr,
        max_duration_s=max_duration,
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(source)
    axes[0, 0].set_title("Source")
    axes[0, 1].plot(rir)
    axes[0, 1].set_title("RIR")
    axes[1, 0].plot(noise)
    axes[1, 0].set_title("Noise")
    axes[1, 1].plot(noisy_y)
    axes[1, 1].set_title(f"Noisy reverberant signal (SNR={snr_dB} dB)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()

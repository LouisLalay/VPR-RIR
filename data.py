from pandas import DataFrame, read_csv
from pathlib import Path
from torch import Tensor
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

    noise = torch.nn.functional.pad(
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

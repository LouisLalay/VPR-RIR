from data import SimpleDataset, prepare_audios
from pathlib import Path
from torch import set_default_dtype, float64
from vprir import VPRIR
from yaml import safe_load
import constants
import sys


def main(*args):
    # Parse CLI arguments
    config_file = Path(args[0]) if args else Path("config.yaml")
    print(f"Using config file: {config_file}")
    config = safe_load(config_file.read_text())

    source_set = SimpleDataset(**config["source"])
    rir_set = SimpleDataset(**config["rir"])
    noise_set = SimpleDataset(**config["noise"])

    source, source_df = source_set[0]
    rir, rir_df = rir_set[0]
    noise, noise_df = noise_set[0]

    (
        trimmed_rir,
        adapted_noise,
        reverberant_signal,
        noisy_reverberant_signal,
    ) = prepare_audios(
        source,
        rir,
        noise,
        config["model"]["Lh"],
        config["noise"]["snr_dB"],
    )
    original_signals = {
        constants.SOURCE_KEY: source,
        constants.RIR_KEY: trimmed_rir,
        constants.NOISE_KEY: adapted_noise,
        constants.REVERBERANT_SIGNAL_KEY: reverberant_signal,
        constants.NOISY_REVERBERANT_SIGNAL_KEY: noisy_reverberant_signal,
    }

    model = VPRIR(
        source=source,
        reverberant_signal=noisy_reverberant_signal,
        reference_h=trimmed_rir,
        **config["model"],
    )
    # model.load("runs/test/2025-05-14 17h14-45")
    model.save(original_signals, config_file)
    model.fit_autodiff(**config["experiment"])


if __name__ == "__main__":
    set_default_dtype(float64)
    main(*sys.argv[1:])

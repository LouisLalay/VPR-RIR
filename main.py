from data import SimpleDataset, prepare_audios, mixing_model
from pathlib import Path
from torch import set_default_dtype, float64
from vprir import VPRIR
from vprir_prego import VPRIRPrego
from yaml import safe_load
import constants
import sys


def autodiff(*args):
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
    try:
        model.load("runs/test/2025-06-05 16h21-58")
    except AssertionError:
        print("Starting a new experiment")
    model.save(original_signals, config_file)
    model.fit_autodiff(**config["experiment"])


def custom_recipe(*args):
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
    model.save(original_signals, config_file)
    model.fit_custom_recipe(**config["experiment"])


def prego(*args):
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
        adapted_noise,
        reverberant_signal,
        noisy_reverberant_signal,
    ) = mixing_model(
        source,
        rir,
        noise,
        config["noise"]["snr_dB"],
        config["sr"],
        4.0,
    )
    original_signals = {
        constants.SOURCE_KEY: source,
        constants.RIR_KEY: rir,
        constants.NOISE_KEY: adapted_noise,
        constants.REVERBERANT_SIGNAL_KEY: reverberant_signal,
        constants.NOISY_REVERBERANT_SIGNAL_KEY: noisy_reverberant_signal,
    }

    model = VPRIRPrego(
        source=source,
        reverberant_signal=noisy_reverberant_signal,
        reference_h=rir,
        **config["model"],
    )
    # try:
    #     model.load("runs/test_prego/2025-09-16 09h48-20")
    # except AssertionError:
    #     print("Starting a new experiment")
    model.save(original_signals, config_file)
    model.fit(**config["experiment"])


def dummy_exp(*args):
    # Parse CLI arguments
    config_file = Path(args[0]) if args else Path("config.yaml")
    print(f"Using config file: {config_file}")
    config = safe_load(config_file.read_text())
    rir_set = SimpleDataset(**config["rir"])
    rir, rir_df = rir_set[0]
    print(rir_df)

    rir = rir[: config["sr"]] / rir.abs().max()
    y = rir
    from torch import ones

    s = ones(1)
    model = VPRIRPrego(
        source=s,
        reverberant_signal=y,
        reference_h=rir,
        **config["model"],
    )
    model.fit_parameters_only(n_steps=2000, log_freq=10)


if __name__ == "__main__":
    set_default_dtype(float64)
    # autodiff(*sys.argv[1:])
    # custom_recipe(*sys.argv[1:])
    dummy_exp(*sys.argv[1:])
    print("Il faut absolument scale y")
    exit(0)
    prego(*sys.argv[1:])

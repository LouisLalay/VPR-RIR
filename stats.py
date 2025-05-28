from baseline import compute_baselines, load_baselines
from metrics import compare
from pandas import DataFrame, concat
from pathlib import Path
from torchaudio import load
from yaml import safe_load
import constants


def analyse_experiment(exp_root: Path):
    """
    Compute the metrics for all the runs in one experiment.
    Runs in the same folder are considered to be part of the same experiment.
    Saves the results in a CSV file named after the experiment root folder.

    Args
    ----
    exp_root : Path
        Path to the experiment root folder.
    """
    # Walk through the experiment root folder
    exp_root = Path(exp_root)
    assert exp_root.is_dir(), f"{exp_root} is not a directory."
    df = DataFrame()
    for run_root in exp_root.iterdir():
        if run_root.is_dir():
            df_run = analyse_run(run_root)
            df = concat([df, df_run], ignore_index=True)
    return df


def analyse_run(run_root: Path) -> DataFrame:
    """
    Computes the metrics for our model and the baselines for a single run.

    Args
    ----
    run_root : Path
        Path to the run folder.
    model : str
        The name of the model used for the run.
    Returns
    -------
    DataFrame
        A DataFrame containing the metrics for the run.
    """
    # Ensure the run_root is a path instance
    run_root = Path(run_root)
    assert run_root.is_dir(), f"{run_root} is not a directory."

    # Load the audios
    source, source_sr = load(run_root / (constants.SOURCE_KEY + ".wav"))
    source = source.squeeze()
    noisy_y, noisy_y_sr = load(
        run_root / (constants.NOISY_REVERBERANT_SIGNAL_KEY + ".wav")
    )
    noisy_y = noisy_y.squeeze()
    our_rir, est_rir_sr = load(run_root / constants.ESTIMATED_RIR_NAME)
    our_rir = our_rir.squeeze()
    rir_ref, ref_rir_sr = load(run_root / (constants.RIR_KEY + ".wav"))
    rir_ref = rir_ref.squeeze()
    # Load the config
    config_path = run_root / constants.SELF_CONFIG_NAME
    config = safe_load(config_path.read_text())
    # Verify SR consistency
    assert (
        source_sr == config["sr"]
        and noisy_y_sr == config["sr"]
        and est_rir_sr == config["sr"]
        and ref_rir_sr == config["sr"]
    ), f"Sampling rates do not match: ({source_sr}, {noisy_y_sr}, {est_rir_sr}, {ref_rir_sr})"

    # Get the estimated RIR from baselines
    compute_baselines(source, noisy_y, source_sr, run_root)
    estimated_rirs = load_baselines(run_root)
    estimated_rirs[constants.MODEL_NAMES["our"]] = our_rir

    # Compute the metrics for all models
    df = DataFrame()
    for model_name, estimated_rir in estimated_rirs.items():
        metrics = compare(rir_ref, estimated_rir, config["sr"])
        metrics["model"] = model_name
        metrics["SNR_dB"] = config["noise"]["snr_dB"]
        metrics["nb_steps"] = config["experiment"]["n_steps"]
        df = concat([df, DataFrame(metrics, index=[0])], ignore_index=True)

    return df

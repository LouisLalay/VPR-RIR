# Dict structure for the orignial audios saved at the beginning of the experiment
NOISE_KEY = "noise"
NOISY_REVERBERANT_SIGNAL_KEY = "noisy_reverberant_signal"
REVERBERANT_SIGNAL_KEY = "reverberant_signal"
RIR_KEY = "rir"
SOURCE_KEY = "source"

# Names of the files saved at runtime
ESTIMATED_RIR_NAME = "estimated_rir.wav"
MODEL_STATE_DICT_NAME = "state_dict.pkl"
OPTIMIZER_STATE_DICT_NAME = "optimizer_state_dict.pkl"
SELF_CONFIG_NAME = "config.yaml"

# Names of the columns in the metrics dataframe
METRICS_NAMES = {
    "c80": "Clarity 80 ms",
    "d50": "Definition 50 ms",
    "drr": "Direct to Reverberant Ratio",
    "edc": "Energy Decay Curve",
    "edr": "Energy Decay Relief",
    "rt30": "Reverberation Time 30 dB",
    "mse": "Mean Squared Error",
}
METRICS_UNITS = {
    METRICS_NAMES["c80"]: "dB",
    METRICS_NAMES["d50"]: "dB",
    METRICS_NAMES["drr"]: "dB",
    METRICS_NAMES["edc"]: "-",
    METRICS_NAMES["edr"]: "-",
    METRICS_NAMES["rt30"]: "s",
    METRICS_NAMES["mse"]: "%",
}
DELTA_MODE_MEAN = "mean"
DELTA_MODE_PERCENT = "percent"
DELTA_MODE_SUM = "sum"
METRICS_DELTA_MODE = {
    "c80": "",
    "d50": "",
    "drr": "",
    "edc": DELTA_MODE_MEAN,
    "edr": DELTA_MODE_MEAN,
    "rt30": "",
    "mse": DELTA_MODE_PERCENT,
}

ALMOST_ZERO = 1e-10

MODEL_NAMES = {
    "our": "VPR-RIR",
    "dec": "Deconvolution in Fourier domain",
    "cbf": "Crossband filter deconvolution",
}
BASELINE_AUDIO_NAMES = {
    "dec": "deconvolved.wav",
    "cbf": "crossband_filter_deconvolved.wav",
}

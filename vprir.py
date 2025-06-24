from constants import ALMOST_ZERO
from datetime import datetime
from metrics import compare, evaluate
from pathlib import Path
from physical_model import PhysicalRIRModel
from plot_utils import filter_fig, rir_fig, rir_spectrograms
from signal_processing_utils import stabilize_filter
from torch import Tensor
from torch.nn import Module, Parameter
from torch.utils.tensorboard import SummaryWriter
from torchaudio import save, load
from tqdm import tqdm
import constants
import torch


class VPRIR(Module):
    """
    This Module implements the VPRIR model ready to be fit to data.
    """

    y: Tensor
    s: Tensor
    epsilon_sample: Tensor

    def __init__(
        self,
        source: Tensor,
        reverberant_signal: Tensor,
        Lh: int,
        Lg: int,
        Lp: int,
        sr: int,
        force_zeros: bool,
        device: str = "cuda",
        reference_h: Tensor = None,
        experiment_directory: Path = None,
        *args,
        **kwargs,
    ):
        """
        Args
        ----
        source: Tensor
            The known source signal.
        reverberant_signal: Tensor
            The possibly noisy reverberant signal.
        Lh: int
            The length of the RIR to be estimated.
        Lg: int
            The length of the g filter.
        Lp: int
            The length of the p filter.
        force_zeros: bool
            If True, the g filter is forced to have zeros at 0 and 0.5 normalized frequency.
        device: str
            The device to use for the model. Default is "cuda".
        """
        super().__init__(*args, **kwargs)
        self.sr = sr
        self.reference_h = reference_h

        self.physical_model = PhysicalRIRModel(
            Lg=Lg,
            Lp=Lp,
            Lh=Lh,
            force_zeros=force_zeros,
        )
        self.mu_h = Parameter(torch.zeros(Lh))
        self.mu_h.data[0] = 1.0
        self.r_h = Parameter(torch.ones(Lh))

        self.register_buffer("y", reverberant_signal)
        self.register_buffer("s", source)
        self.register_buffer("epsilon_sample", torch.randn(Lh))

        self.to(device)

        # Constants for the loss function
        S: Tensor = torch.fft.fft(self.s, n=self.y.shape[0])
        self.S2 = S.real**2 + S.imag**2
        Y: Tensor = torch.fft.fft(self.y, n=self.y.shape[0])
        self.Y_conj_S = Y.conj() * S

        self.last_loss = 0

        exp_name = datetime.now().strftime("%Y-%m-%d %Hh%M-%S")
        self.full_exp_path = Path(experiment_directory) / exp_name
        self.logger = SummaryWriter(self.full_exp_path)

        self.start_new = True

    def expectation_diff(self):
        MU_h: Tensor = torch.fft.fft(self.mu_h, n=self.y.shape[0])
        expectation_diff = (
            (MU_h.real**2 + MU_h.imag**2 + self.r_h.sum()) @ self.S2
            - 2 * (self.Y_conj_S @ MU_h).real
        ) / self.y.shape[0]
        return expectation_diff

    def loss(self):
        signal_term = (
            self.y.shape[0] * torch.log(2 * torch.pi * self.physical_model.sigma_w_2)
            + self.expectation_diff() / self.physical_model.sigma_w_2
        )

        PEG_mu_h = self.physical_model.lvecmul_PEG(self.mu_h)
        rir_term = (
            self.physical_model.Lh * torch.log(self.physical_model.sigma_epsilon_2)
            - self.physical_model.Lh
            * (self.physical_model.Lh - 1)
            * self.physical_model.a
            + PEG_mu_h @ PEG_mu_h / self.physical_model.sigma_epsilon_2
            + self.physical_model.trace_PEG_Rh(self.r_h)
            / self.physical_model.sigma_epsilon_2
            - torch.log(self.r_h).sum()
        )
        return signal_term + rir_term

    def stabilize_parameters(self):
        """Bring back parameters in a valid range."""
        self.physical_model.g.data = stabilize_filter(self.physical_model.g)
        self.r_h.data = self.r_h.data.clamp(min=ALMOST_ZERO)
        self.physical_model.a.data = self.physical_model.a.data.clamp(0)

    def correct_gradients(self):
        """
        Force the gradient on p[0] and g[0] to be zero so p[0] and g[0] stay equal to 1.
        """
        self.physical_model.p.grad[0] = 0
        self.physical_model.g.grad[0] = 0

    def fit_autodiff(self, n_steps: int, log_freq: int = 50, lr: float = 1e-3):
        if self.start_new:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            first_step = 0
        else:
            first_step = int(self.optimizer.state_dict()["state"][0]["step"])
        for k in tqdm(range(first_step, n_steps + 1)):
            # Keep the parameters in a valid range
            self.stabilize_parameters()
            # Plot params to Tensorboard
            if not k % log_freq:
                self.log_step(k)
            # Classical gradient descent step
            self.optimizer.zero_grad()
            loss = self.loss()
            self.last_loss = loss.item()
            loss.backward()
            # Keep g[0] and p[0] equal to 1
            self.correct_gradients()
            self.optimizer.step()

    def log_step(self, step: int):
        self.logger.add_scalar("Main/Loss", self.last_loss, step)
        self.logger.add_scalar("Main/a", self.physical_model.a.item(), step)
        self.logger.add_scalar(
            "Sigma/epsilon^2", self.physical_model.sigma_epsilon_2.item(), step
        )
        self.logger.add_scalar("Sigma/w^2", self.physical_model.sigma_w_2.item(), step)
        self.logger.add_figure(
            "Filters",
            filter_fig(
                self.physical_model.g,
                self.physical_model.g0_inv,
                self.physical_model.a,
                self.physical_model.p,
            ),
            step,
        )
        self.logger.add_figure(
            "RIR",
            rir_fig(self.mu_h, self.r_h, self.sr, self.reference_h),
            step,
        )

        # Generate a RIR from the model parameters
        with torch.no_grad():
            h_model = self.physical_model.lvecmul_PEG_inv(
                self.epsilon_sample * self.physical_model.sigma_epsilon_2**0.5
            )
        self.logger.add_figure(
            "Spectrograms",
            rir_spectrograms(self.mu_h, h_model, ref_rir=self.reference_h, sr=self.sr),
            step,
        )

        # Metrics
        if self.reference_h is not None:
            metrics = compare(self.reference_h, self.mu_h, self.sr)
        else:
            metrics = evaluate(self.mu_h, self.sr)
        for key, val in metrics.items():
            self.logger.add_scalar("Metrics/" + key, val, step)
        self.save()

    def save(
        self,
        original_signals: dict[str, Tensor] = None,
        config_file: Path = None,
    ):
        if original_signals is not None:
            # Copy the audios and the config file to the experiment directory
            for name, signal in original_signals.items():
                filename = self.full_exp_path / f"{name}.wav"
                wav_tensor = signal.unsqueeze(0)
                save(uri=filename, src=wav_tensor, sample_rate=self.sr)
        if config_file is not None:
            # Copy the config file to the experiment directory
            filename = self.full_exp_path / constants.SELF_CONFIG_NAME
            filename.write_text(config_file.read_text())

        # Save the model parameters
        torch.save(
            self.state_dict(),
            self.full_exp_path / constants.MODEL_STATE_DICT_NAME,
        )

        # Save the estimated RIR as a wav file
        filename = self.full_exp_path / constants.ESTIMATED_RIR_NAME
        estimated_rir_wav = self.mu_h.data.unsqueeze(0).cpu()
        save(uri=filename, src=estimated_rir_wav, sample_rate=self.sr)

        # Save the optimizer state
        if hasattr(self, "optimizer"):
            torch.save(
                self.optimizer.state_dict(),
                self.full_exp_path / constants.OPTIMIZER_STATE_DICT_NAME,
            )

    def load(self, last_exp_path: Path):
        """
        Continue the experiment from a previous state.
        This is meant to be used with the exact same inputs
        and only continue after a crash or for precision.
        """

        last_exp_path = Path(last_exp_path)

        # Check the compatibility of the two experiments
        last_y_path = last_exp_path / f"{constants.NOISY_REVERBERANT_SIGNAL_KEY}.wav"
        assert last_y_path.exists(), "Some files are missing in the last experiment"
        last_state_path = last_exp_path / constants.MODEL_STATE_DICT_NAME
        assert last_state_path.exists(), "Some files are missing in the last experiment"
        last_opti_path = last_exp_path / constants.OPTIMIZER_STATE_DICT_NAME
        assert last_opti_path.exists(), "Some files are missing in the last experiment"

        # This ensures Lh, source, noise, SNR and ref_h are the same
        last_y, last_sr = load(last_y_path)
        last_y = last_y.squeeze().to(self.y)
        assert last_sr == self.sr, "Sample rates are different"
        assert last_y.shape == self.y.shape, f"Signals lengths are different."
        assert torch.allclose(last_y, self.y, atol=1e-3), "Signals are different"

        # Load the model parameters
        last_state: dict[str, Tensor] = torch.load(last_state_path, weights_only=True)
        for name, p in self.named_parameters():
            assert name in last_state, f"{name} not in last state {last_state.keys()}"
            assert p.shape == last_state[name].shape, f"{name} shape mismatch"
        current_device = self.mu_h.device
        self.load_state_dict(last_state)
        self.to(current_device)

        # Load the optimizer state
        opti_state = torch.load(last_opti_path, weights_only=True)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer.load_state_dict(opti_state)

        self.last_loss = self.loss().item()
        self.start_new = False

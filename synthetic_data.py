from physical_model import PhysicalRIRModel
from pytorch_lightning import LightningDataModule
from signal_processing_utils import stabilize_filter
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch


def create_AR_coefs(L: int, sigma: float, device: torch.device) -> torch.Tensor:
    ar_coefs = torch.randn(L, device=device) * sigma
    ar_coefs[0] = 1.0
    return stabilize_filter(ar_coefs)


class SyntheticRIRSet(Dataset):
    def __init__(self, n_samples: int, Lh: int, device: torch.device = "cpu"):
        super().__init__()
        self.n_samples = n_samples
        # Max length for reasonable compute time
        self.max_Lh = Lh
        # Minimum 512 for metrics computation
        self.min_Lh = 600

        self.Lg = 20
        self.Lp = 10

        self.rir_model = PhysicalRIRModel(
            Lg=self.Lg,
            Lp=self.Lp,
            Lh=self.max_Lh,
            force_zeros=False,
        )
        self.device = device
        self.rir_model = self.rir_model.to(self.device)

        # For memeory efficiency, we will sample from lists of parameters
        self.sqr_n_samples = int(self.n_samples**0.5)
        self.g_parameters = [
            create_AR_coefs(self.Lg, 1e-2, self.device)
            for _ in range(self.sqr_n_samples)
        ]
        self.p_parameters = [
            create_AR_coefs(self.Lp, 3e-4, self.device)
            for _ in range(self.sqr_n_samples)
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        i = (index // self.sqr_n_samples) % self.sqr_n_samples
        j = index % self.sqr_n_samples

        a_scale = 0.05
        self.rir_model.g.data = self.g_parameters[i]
        self.rir_model.a.data = torch.rand(1, device=self.device) * a_scale + 1e-3
        self.rir_model.p.data = self.p_parameters[j]

        sigma_epsilon = torch.rand(1, device=self.device) * 0.5 + 0.5

        Lh = (
            int((1 - self.rir_model.a / a_scale) * (self.max_Lh - self.min_Lh))
            + self.min_Lh
        )
        epsilon = torch.randn(Lh, device=self.device) * sigma_epsilon
        with torch.no_grad():
            h = self.rir_model.lvecmul_PEG_inv(epsilon)

        power = h**2
        if power[: Lh // 2].mean() < power[Lh // 2 :].mean():
            # Power is growing <=> unstable p - a combination
            # Try again
            return self.__getitem__(index)

        delay = torch.randint(0, (self.max_Lh - Lh) // 10, (1,))
        h = torch.concat(
            [
                torch.zeros(delay, device=self.device),
                h,
                torch.zeros(self.max_Lh - Lh - delay, device=self.device),
            ]
        )

        parameters = {
            "g": self.rir_model.g.data,
            "a": self.rir_model.a.data,
            "p": self.rir_model.p.data,
            "epsilon": torch.concat(
                [
                    torch.zeros(delay, device=self.device),
                    epsilon,
                    torch.zeros(self.max_Lh - Lh - delay, device=self.device),
                ]
            ),
        }
        return h, parameters


class SyntheticRIRLightning(LightningDataModule):
    def __init__(self, size: int, Lh: int, batch_size: int):
        super().__init__()
        self.dataset = SyntheticRIRSet(size, Lh)
        self.batch_size = batch_size

    def params_to_latent(self, params: dict[str, Tensor]) -> Tensor:
        return torch.cat(
            [
                params["g"],
                params["a"],
                params["p"],
                params["epsilon"],
            ]
        )

    def collate(self, batch):
        # Batch composition : list of (h, parameters) tuples
        hs = batch[0][0]
        params = self.params_to_latent(batch[0][1])

        for h, p in batch[1:]:
            hs = torch.vstack((hs, h))
            params = torch.vstack((params, self.params_to_latent(p)))
        return hs, params

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=self.collate,
        )

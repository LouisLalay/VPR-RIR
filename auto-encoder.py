from matplotlib import pyplot as plt
from pathlib import Path
from pytorch_lightning import LightningModule, Trainer, seed_everything
from synthetic_data import SyntheticRIRLightning, SyntheticRIRSet
from torch import Tensor
from torch.nn import Module, Linear, ReLU
from tqdm import tqdm
from typing import Callable
from yaml import safe_load
import torch


class Encoder(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_layer = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.hidden_layer = Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        return x


class RIREncoder(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RIREncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, output_dim)
        self.loss: Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        h, params = batch
        latent_space = self.forward(h)
        truth = params[:, : latent_space.shape[1]]
        loss_value = self.loss(truth, latent_space)
        self.log("train_loss", loss_value.item())
        return loss_value


def main(n_epochs: int = 1):
    config_file = Path("config.yaml")
    config = safe_load(config_file.read_text())
    device = config["model"]["device"]

    input_dim = config["model"]["Lh"]
    L_eps = input_dim // 4
    hidden_dim = 2 * input_dim
    output_dim = config["model"]["Lg"] + config["model"]["Lp"] + 1 + L_eps

    model = RIREncoder(input_dim, hidden_dim, output_dim)
    model.to(device)

    data = SyntheticRIRLightning(4096, input_dim, batch_size=32)

    trainer = Trainer(max_epochs=n_epochs, accelerator=device)
    trainer.fit(model, data)

    return model


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    seed_everything(42)
    main(200)

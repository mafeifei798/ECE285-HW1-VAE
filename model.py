"""Convolutional VAE for 64 x 64 grayscale chest X-rays."""

import math

import torch
from torch import nn


def _groups(channels):
    return math.gcd(8, channels)


class ResidualBlock(nn.Module):
    """Pre-activation 3 x 3 residual block; the residual branch starts at zero."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.GroupNorm(_groups(in_channels), in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.shortcut = (nn.Identity() if in_channels == out_channels
                         else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=256, base_channels=32, observation_std=0.01):
        super().__init__()
        self.image_size = 64
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.observation_std = observation_std
        self.prior = "standard_normal"

        channels = [base_channels * 2 ** i for i in range(4)]
        self._encoded_channels = channels[-1]
        flattened = channels[-1] * 4 * 4
        self.encoder = self._residual_encoder(channels)
        self.fc_mu = nn.Linear(flattened, latent_dim)
        self.fc_logvar = nn.Linear(flattened, latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
        self.decoder_input = nn.Linear(latent_dim, flattened)
        self.decoder = self._residual_decoder(channels)

    @staticmethod
    def _residual_encoder(channels):
        """Residual blocks at each resolution, then a strided 4 x 4 convolution."""
        encoder = [nn.Conv2d(1, channels[0], 3, 1, 1)]
        previous = channels[0]
        for index, channel in enumerate(channels):
            encoder.append(ResidualBlock(previous, channel))
            previous = channel
            following = channels[min(index + 1, len(channels) - 1)]
            encoder.append(nn.Conv2d(previous, following, 4, 2, 1))
            previous = following
        encoder.extend([nn.GroupNorm(_groups(previous), previous), nn.SiLU()])
        return nn.Sequential(*encoder)

    @staticmethod
    def _residual_decoder(channels):
        """Residual blocks with nearest upsampling and 3 x 3 convolutions."""
        decoder = []
        previous = channels[-1]
        for channel in reversed(channels):
            decoder.append(ResidualBlock(previous, previous))
            decoder.extend([nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(previous, channel, 3, 1, 1)])
            previous = channel
        decoder.append(ResidualBlock(previous, previous))
        decoder.extend([nn.GroupNorm(_groups(previous), previous), nn.SiLU(),
                        nn.Conv2d(previous, 1, 3, 1, 1)])
        return nn.Sequential(*decoder)

    @property
    def model_config(self):
        return {"image_size": 64, "latent_dim": self.latent_dim,
                "base_channels": self.base_channels, "observation_std": self.observation_std,
                "likelihood": "gaussian", "prior": "standard_normal",
                "arch": "resnet", "res_blocks": 1}

    def encode(self, x):
        features = self.encoder(x).flatten(1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu, logvar, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * epsilon

    def decode(self, z):
        features = self.decoder_input(z).reshape(-1, self._encoded_channels, 4, 4)
        return self.decoder(features)

    def pixel_mean(self, logits):
        return torch.sigmoid(logits)

    def forward(self, x, epsilon=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, epsilon)
        return self.decode(z), mu, logvar


def elbo_loss(logits, target, mu, logvar, observation_std=0.01):
    """Negative ELBO: sum over pixels and latent dimensions, mean over images."""
    mean = torch.sigmoid(logits)
    pixel_nll = ((target - mean).square() / (2 * observation_std ** 2)
                 + math.log(observation_std) + 0.5 * math.log(2 * math.pi))
    reconstruction = pixel_nll.flatten(1).sum(1).mean()

    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return {"loss": reconstruction + kl, "reconstruction": reconstruction, "kl": kl}


def load_vae_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint["model_config"]
    model = ConvVAE(config["latent_dim"], config["base_channels"], config["observation_std"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(map_location).eval(), checkpoint

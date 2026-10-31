"""Convolutional VAE for 64 x 64 grayscale chest X-rays."""

import math

import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, image_size=64, latent_dim=64, base_channels=16,
                 likelihood="bernoulli", observation_std=0.05,
                 prior="standard_normal", arch="plain", res_blocks=1):
        super().__init__()
        if image_size not in (32, 64, 128, 256):
            raise ValueError("image_size must be 32, 64, 128, or 256")
        if arch not in ("plain", "resnet"):
            raise ValueError("arch must be plain or resnet")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.likelihood = likelihood
        self.observation_std = observation_std
        self.prior = prior
        self.arch = arch
        self.res_blocks = res_blocks
        fitted = prior == "fitted_gaussian"
        self.register_buffer("prior_mean", torch.zeros(latent_dim) if fitted else None)
        self.register_buffer("prior_scale_tril", torch.eye(latent_dim) if fitted else None)

        depth = int(math.log2(image_size)) - 2
        channels = [min(base_channels * 2 ** i, base_channels * 8) for i in range(depth)]
        self._encoded_channels = channels[-1]
        flattened = channels[-1] * 4 * 4
        residual = arch == "resnet"
        self.encoder = (self._residual_encoder(channels, res_blocks) if residual
                        else self._plain_encoder(channels))
        self.fc_mu = nn.Linear(flattened, latent_dim)
        self.fc_logvar = nn.Linear(flattened, latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
        self.decoder_input = nn.Linear(latent_dim, flattened)
        self.decoder = (self._residual_decoder(channels, res_blocks) if residual
                        else self._plain_decoder(channels))

    @staticmethod
    def _plain_encoder(channels):
        encoder = []
        previous = 1
        for channel in channels:
            encoder.extend([
                nn.Conv2d(previous, channel, 4, 2, 1),
                nn.GroupNorm(_groups(channel), channel),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            previous = channel
        return nn.Sequential(*encoder)

    @staticmethod
    def _plain_decoder(channels):
        decoder = []
        previous = channels[-1]
        for channel in reversed(channels[:-1]):
            decoder.extend([
                nn.ConvTranspose2d(previous, channel, 4, 2, 1),
                nn.GroupNorm(_groups(channel), channel),
                nn.ReLU(inplace=True),
            ])
            previous = channel
        decoder.append(nn.ConvTranspose2d(previous, 1, 4, 2, 1))
        return nn.Sequential(*decoder)

    @staticmethod
    def _residual_encoder(channels, blocks):
        """Residual blocks at each resolution, then a strided 4 x 4 convolution."""
        encoder = [nn.Conv2d(1, channels[0], 3, 1, 1)]
        previous = channels[0]
        for index, channel in enumerate(channels):
            for _ in range(blocks):
                encoder.append(ResidualBlock(previous, channel))
                previous = channel
            following = channels[min(index + 1, len(channels) - 1)]
            encoder.append(nn.Conv2d(previous, following, 4, 2, 1))
            previous = following
        encoder.extend([nn.GroupNorm(_groups(previous), previous), nn.SiLU()])
        return nn.Sequential(*encoder)

    @staticmethod
    def _residual_decoder(channels, blocks):
        """Residual blocks with nearest upsampling and 3 x 3 convolutions."""
        decoder = []
        previous = channels[-1]
        for channel in reversed(channels):
            decoder.extend(ResidualBlock(previous, previous) for _ in range(blocks))
            decoder.extend([nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(previous, channel, 3, 1, 1)])
            previous = channel
        decoder.extend(ResidualBlock(previous, previous) for _ in range(blocks))
        decoder.extend([nn.GroupNorm(_groups(previous), previous), nn.SiLU(),
                        nn.Conv2d(previous, 1, 3, 1, 1)])
        return nn.Sequential(*decoder)

    @property
    def model_config(self):
        return {key: getattr(self, key) for key in (
            "image_size", "latent_dim", "base_channels", "likelihood",
            "observation_std", "prior", "arch", "res_blocks")}

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
        if self.likelihood == "continuous_bernoulli":
            return torch.distributions.ContinuousBernoulli(
                logits=logits, validate_args=False).mean
        return torch.sigmoid(logits)

    def latent_from_base(self, epsilon):
        if self.prior == "standard_normal":
            return epsilon
        return epsilon @ self.prior_scale_tril.T + self.prior_mean

    def forward(self, x, epsilon=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, epsilon)
        return self.decode(z), mu, logvar


def elbo_loss(logits, target, mu, logvar, likelihood="bernoulli",
              observation_std=0.05, prior_mean=None, prior_scale_tril=None):
    """Negative ELBO: sum over pixels and latent dimensions, mean over images."""
    if likelihood == "bernoulli":
        pixel_nll = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    elif likelihood == "continuous_bernoulli":
        pixel_nll = -torch.distributions.ContinuousBernoulli(
            logits=logits, validate_args=False).log_prob(target)
    elif likelihood == "gaussian":
        mean = torch.sigmoid(logits)
        pixel_nll = ((target - mean).square() / (2 * observation_std ** 2)
                     + math.log(observation_std) + 0.5 * math.log(2 * math.pi))
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    reconstruction = pixel_nll.flatten(1).sum(1).mean()

    if prior_mean is None:
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    else:
        dimension = mu.shape[1]
        identity = torch.eye(dimension, device=mu.device, dtype=mu.dtype)
        inverse_scale = torch.linalg.solve_triangular(prior_scale_tril, identity, upper=False)
        precision_diagonal = inverse_scale.square().sum(0)
        centered = torch.linalg.solve_triangular(
            prior_scale_tril, (mu - prior_mean).T, upper=False).T
        logdet = 2 * prior_scale_tril.diagonal().log().sum()
        kl = 0.5 * (logvar.exp().mul(precision_diagonal).sum(1)
                    + centered.square().sum(1) - dimension + logdet - logvar.sum(1)).mean()
    return {"loss": reconstruction + kl, "reconstruction": reconstruction, "kl": kl}


def load_vae_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = ConvVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(map_location).eval(), checkpoint

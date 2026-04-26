from dataset import create_elevation_dataloader
import torch
from torch import nn


def _make_conv_block(in_features, out_features, kernel_size, dilation=1):
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        ),
        nn.InstanceNorm2d(out_features),
        nn.LeakyReLU(0.2),
    )


def _make_downsample_block(in_features, out_features, is_first=False):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.Identity() if is_first else nn.InstanceNorm2d(out_features),
        nn.LeakyReLU(0.2),
    )


def _make_upscale_block(in_features, out_features):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_features),
        nn.LeakyReLU(0.2),
    )


class DilatedResidualBlock(nn.Module):
    def __init__(self, features, dilation):
        super().__init__()
        self._cnn = nn.Sequential(
            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
            ),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.InstanceNorm2d(features),
        )
        self._activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self._activation(x + self._cnn(x))


class Generator(nn.Module):
    def __init__(self, image_width):
        super().__init__()
        self._stem_cnn = _make_conv_block(3, 32, kernel_size=3)
        self._downsample_1_cnn = _make_downsample_block(32, 64, is_first=True)
        self._downsample_2_cnn = _make_downsample_block(64, 128)
        self._downsample_3_cnn = _make_downsample_block(128, 256)
        self._middle_cnn = nn.Sequential(
            DilatedResidualBlock(256, 2),
            DilatedResidualBlock(256, 4),
            DilatedResidualBlock(256, 8),
            _make_conv_block(256, 256, kernel_size=3, dilation=1),
        )
        self._upscale_1_cnn = _make_upscale_block(256, 128)
        self._merge_1_cnn = _make_conv_block(256, 128, kernel_size=3)
        self._upscale_2_cnn = _make_upscale_block(128, 64)
        self._merge_2_cnn = _make_conv_block(128, 64, kernel_size=3)
        self._upscale_3_cnn = _make_upscale_block(64, 32)
        self._merge_3_cnn = _make_conv_block(64, 32, kernel_size=3)
        self._out_cnn = nn.Sequential(
            _make_conv_block(32, 32, kernel_size=3, dilation=1),
            _make_conv_block(32, 32, kernel_size=3, dilation=1),
            _make_conv_block(32, 32, kernel_size=3, dilation=1),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Tanh(),
        )

    def _inpaint(self, image, mask, target_image):
        return image * (1 - mask) + target_image * mask

    def forward(self, image, mask):
        z = torch.randn_like(image)
        x = torch.cat([image.unsqueeze(1), mask.unsqueeze(1), z.unsqueeze(1)], dim=1)
        skip_0 = self._stem_cnn(x)
        skip_1 = self._downsample_1_cnn(skip_0)
        skip_2 = self._downsample_2_cnn(skip_1)
        x = self._downsample_3_cnn(skip_2)
        x = self._middle_cnn(x)
        x = self._upscale_1_cnn(x)
        x = torch.cat([x, skip_2], dim=1)
        x = self._merge_1_cnn(x)
        x = self._upscale_2_cnn(x)
        x = torch.cat([x, skip_1], dim=1)
        x = self._merge_2_cnn(x)
        x = self._upscale_3_cnn(x)
        x = torch.cat([x, skip_0], dim=1)
        x = self._merge_3_cnn(x)
        x = self._out_cnn(x)
        output = x.squeeze(1)
        return output


class Discriminator(nn.Module):
    def __init__(self, image_width):
        super().__init__()
        self._cnn = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, image):
        image = image.unsqueeze(1)
        return self._cnn(image)


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataloader = create_elevation_dataloader(batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    batch = batch.squeeze(1)

    print("device:", device)
    generator = Generator(image_width=batch.shape[-1])
    discriminator = Discriminator(image_width=batch.shape[-1])
    generator.to(device)
    discriminator.to(device)
    batch = batch.to(device)

    mask = torch.ones_like(batch)
    mask[:, 24:40, 24:40] = 0
    target_image = batch.clone()
    target_image[:, 24:40, 24:40] = 0

    generated = generator(target_image, mask)
    discriminator_real = discriminator(batch)
    discriminator_fake = discriminator(generated)

    print("dataset_size:", len(dataset))
    print("batch_shape:", batch.shape)
    print("generated_shape:", generated.shape)
    print("discriminator_real_shape:", discriminator_real.shape)
    print("discriminator_fake_shape:", discriminator_fake.shape)
    print("generator_parameters:", _count_parameters(generator))
    print("discriminator_parameters:", _count_parameters(discriminator))
    print("g_d_ratio:", _count_parameters(generator) / _count_parameters(discriminator))


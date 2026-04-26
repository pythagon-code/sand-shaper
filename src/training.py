from pathlib import Path
import numpy as np
from PIL import Image
import torch
import tqdm
from torch import nn
from torch import optim
from dataset import create_elevation_dataloader
from elevations import copy_exports_images_to_elevations
from gan import Discriminator
from gan import Generator

image_width = 64
border_width = 16

masks = [torch.zeros(image_width, image_width) for _ in range(12)]

masks[1][:border_width] = 1
masks[2][-border_width:] = 1
masks[3][:, :border_width] = 1
masks[4][:, -border_width:] = 1

masks[5][:border_width] = 1
masks[5][-border_width:] = 1
masks[6][:, :border_width] = 1
masks[6][:, -border_width:] = 1

masks[7][:border_width] = 1
masks[7][:, :border_width] = 1
masks[8][:border_width] = 1
masks[8][:, -border_width:] = 1
masks[9][-border_width:] = 1
masks[9][:, :border_width] = 1
masks[10][-border_width:] = 1
masks[10][:, -border_width:] = 1

masks[11][:border_width] = 1
masks[11][-border_width:] = 1
masks[11][:, :border_width] = 1
masks[11][:, -border_width:] = 1


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().numpy()
    array_uint8 = (((array + 1.0) * 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode = "L")


def _tensor_to_overlay_image(real: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    gray = ((real.detach().cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    hole = (1.0 - mask.detach().cpu().numpy()).clip(0.0, 1.0)
    alpha = 0.5
    tint_r, tint_g, tint_b = 0.95, 0.22, 0.2
    r = gray * (1.0 - alpha * hole) + tint_r * alpha * hole
    g = gray * (1.0 - alpha * hole) + tint_g * alpha * hole
    b = gray * (1.0 - alpha * hole) + tint_b * alpha * hole
    rgb = (np.stack([r, g, b], axis=-1).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode = "RGB")


def _conditioning_to_overlay_image(composite: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    gray = ((composite.detach().cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    hole = (1.0 - mask.detach().cpu().numpy()).clip(0.0, 1.0)
    alpha = 0.45
    tint_r, tint_g, tint_b = 0.15, 0.45, 0.95
    r = gray * (1.0 - alpha * hole) + tint_r * alpha * hole
    g = gray * (1.0 - alpha * hole) + tint_g * alpha * hole
    b = gray * (1.0 - alpha * hole) + tint_b * alpha * hole
    rgb = (np.stack([r, g, b], axis=-1).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode = "RGB")


def train(epochs=10, batch_size=32, lr=2e-4, num_workers=0):
    copy_exports_images_to_elevations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    _, dataloader = create_elevation_dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers)

    generator = Generator(image_width=image_width)
    discriminator = Discriminator(image_width=image_width)
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    criterion_adv = nn.MSELoss()
    criterion_mse = nn.MSELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    mask_bank = torch.stack(masks).to(device)
    image_dir = Path("data/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_dir.glob("*.png"):
        image_path.unlink()
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    update_count = 0

    for epoch in tqdm.trange(epochs, desc="Training"):
        g_running = 0.0
        d_running = 0.0

        for real_batch in tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            real_batch = real_batch.squeeze(1).to(device)
            current_batch_size = real_batch.shape[0]

            mask_indices = torch.randint(0, mask_bank.shape[0], (current_batch_size,), device=device)
            batch_mask = mask_bank[mask_indices]
            input_image = real_batch * batch_mask
            border_pixels_per_sample = batch_mask.view(current_batch_size, -1).sum(dim=1)
            no_border_samples = border_pixels_per_sample == 0
            if no_border_samples.any():
                random_indices = torch.randint(0, current_batch_size, (current_batch_size,), device=device)
                random_tiles = real_batch[random_indices]
                input_image[no_border_samples] = random_tiles[no_border_samples]

            fake_batch = generator(input_image, batch_mask)

            optimizer_d.zero_grad()
            pred_real = discriminator(real_batch)
            pred_fake = discriminator(fake_batch.detach())
            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)
            d_loss_real = criterion_adv(pred_real, real_labels)
            d_loss_fake = criterion_adv(pred_fake, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizer_d.step()

            for _ in range(2):
                optimizer_g.zero_grad()
                fake_batch = generator(input_image, batch_mask)
                pred_fake_for_g = discriminator(fake_batch)
                adv_labels = torch.ones_like(pred_fake_for_g)
                g_loss_adv = criterion_adv(pred_fake_for_g, adv_labels)
                border_pixels = batch_mask.sum().clamp_min(1.0)
                g_loss_mse = criterion_mse(fake_batch * (1 - batch_mask), real_batch * (1 - batch_mask))
                g_loss_mse = (g_loss_mse * batch_mask.numel() / border_pixels) * 10.0
                g_loss = g_loss_adv + g_loss_mse
                g_loss.backward()
                optimizer_g.step()
                g_running += g_loss.item()

            d_running += d_loss.item()
            update_count += 1

            if update_count % 10 == 0:
                sample_real = real_batch[0]
                sample_fake = fake_batch[0]
                sample_mask = batch_mask[0]
                sample_input = input_image[0]
                _tensor_to_image(sample_real).save(image_dir / f"update_{update_count:06d}_real.png")
                _conditioning_to_overlay_image(sample_input, sample_mask).save(
                    image_dir / f"update_{update_count:06d}_overlay.png",
                )
                _tensor_to_image(sample_fake).save(image_dir / f"update_{update_count:06d}_fake.png")
                print(
                    f"\nUpdate={update_count} epoch={epoch + 1}/{epochs} "
                    f"g_loss={g_loss.item():.6f} d_loss={d_loss.item():.6f}"
                )
            if update_count % 100 == 0:
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "epoch": epoch + 1,
                        "update": update_count,
                    },
                    models_dir / f"{update_count:06d}.pt",
                )

        g_epoch = g_running / len(dataloader)
        d_epoch = d_running / len(dataloader)
        print(f"\nepoch={epoch + 1}/{epochs} g_loss={g_epoch:.6f} d_loss={d_epoch:.6f}")

    return generator, discriminator


if __name__ == "__main__":
    train()

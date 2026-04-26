from collections import deque
from pathlib import Path
import numpy as np
import torch
from gan import Generator
from PIL import Image


def _tile_coords(x0: int, y0: int, size: int, world: int) -> tuple[np.ndarray, np.ndarray]:
    ys = (np.arange(size)[:, None] + y0) % world
    xs = (np.arange(size)[None, :] + x0) % world
    return ys, xs


def _build_alpha(mask: np.ndarray, seam_width: int) -> np.ndarray:
    alpha = np.ones_like(mask, dtype=np.float32)
    known = mask > 0.5
    unknown = ~known
    if seam_width <= 0:
        alpha[known] = 0.0
        return alpha
    if not unknown.any():
        alpha[:] = 0.0
        return alpha
    unknown_positions = np.argwhere(unknown)
    known_positions = np.argwhere(known)
    alpha[unknown] = 1.0
    if known_positions.size == 0:
        return alpha
    for y, x in known_positions:
        distances = np.abs(unknown_positions[:, 0] - y) + np.abs(unknown_positions[:, 1] - x)
        nearest = int(distances.min())
        if nearest >= seam_width:
            alpha[y, x] = 0.0
        else:
            alpha[y, x] = float(seam_width - nearest + 1) / float(seam_width + 1)
    return alpha


def _load_generator(project_root: Path, checkpoint_name: str, image_width: int, device: torch.device) -> Generator:
    checkpoint_path = project_root / "exports" / "models" / checkpoint_name
    generator = Generator(image_width = image_width).to(device)
    checkpoint = torch.load(checkpoint_path, map_location = device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def _random_seed_patch(project_root: Path, tile_size: int, rng: np.random.Generator) -> np.ndarray:
    elevation_dir = project_root / "assets" / "data" / "elevations"
    tile_paths = sorted(elevation_dir.glob("*.png"))
    if not tile_paths:
        return rng.uniform(-1.0, 1.0, size=(tile_size, tile_size)).astype(np.float32)
    tile_path = tile_paths[int(rng.integers(0, len(tile_paths)))]
    tile_array = np.array(Image.open(tile_path).convert("L"), dtype=np.float32)
    tile_array = (tile_array / 255.0) * 2.0 - 1.0
    h, w = tile_array.shape
    if h < tile_size or w < tile_size:
        pad_y = max(0, tile_size - h)
        pad_x = max(0, tile_size - w)
        tile_array = np.pad(tile_array, ((0, pad_y), (0, pad_x)), mode="edge")
        h, w = tile_array.shape
    y0 = int(rng.integers(0, h - tile_size + 1))
    x0 = int(rng.integers(0, w - tile_size + 1))
    return tile_array[y0:y0 + tile_size, x0:x0 + tile_size].astype(np.float32)


def _connected_random_order(steps_x: list[int], steps_y: list[int], rng: np.random.Generator) -> list[tuple[int, int]]:
    grid_h = len(steps_y)
    grid_w = len(steps_x)
    start_y = int(rng.integers(0, grid_h))
    start_x = int(rng.integers(0, grid_w))
    order: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
    visited: set[tuple[int, int]] = {(start_y, start_x)}
    while queue:
        gy, gx = queue.popleft()
        order.append((steps_x[gx], steps_y[gy]))
        neighbors = [
            ((gy - 1) % grid_h, gx),
            ((gy + 1) % grid_h, gx),
            (gy, (gx - 1) % grid_w),
            (gy, (gx + 1) % grid_w),
        ]
        rng.shuffle(neighbors)
        for ny, nx in neighbors:
            if (ny, nx) not in visited:
                visited.add((ny, nx))
                queue.append((ny, nx))
    return order


def _upscale_with_averages(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    upscaled = np.empty((height * 2 - 1, width * 2 - 1), dtype=np.float32)
    upscaled[0::2, 0::2] = grid
    upscaled[0::2, 1::2] = 0.5 * (grid[:, :-1] + grid[:, 1:])
    upscaled[1::2, 0::2] = 0.5 * (grid[:-1, :] + grid[1:, :])
    upscaled[1::2, 1::2] = 0.25 * (grid[:-1, :-1] + grid[:-1, 1:] + grid[1:, :-1] + grid[1:, 1:])
    return upscaled


def generate_world_terrain(
    project_root: Path,
    generated_tile_name: str = "generated_terrain",
    checkpoint_name: str = "model.pt",
    tile_size: int = 64,
    world_size: int = 64,
    stride: int = 32,
    blend_width: int = 8,
    random_seed: int | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(random_seed)
    generator = _load_generator(project_root, checkpoint_name, tile_size, device)
    generated_tile_path = project_root / "assets" / "data" / "terrains" / f"{generated_tile_name}.png"
    generated_tile_path.parent.mkdir(parents = True, exist_ok = True)
    canvas = np.full((world_size, world_size), np.nan, dtype=np.float32)
    steps_x = list(range(0, world_size, stride))
    steps_y = list(range(0, world_size, stride))
    march_order = _connected_random_order(steps_x, steps_y, rng)
    start_x, start_y = march_order[0]
    start_ys, start_xs = _tile_coords(start_x, start_y, tile_size, world_size)
    canvas[start_ys, start_xs] = _random_seed_patch(project_root, tile_size, rng)
    for x0, y0 in march_order:
        ys, xs = _tile_coords(x0, y0, tile_size, world_size)
        existing = canvas[ys, xs]
        mask = np.isfinite(existing).astype(np.float32)
        image = np.where(np.isfinite(existing), existing, 0.0).astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            generated = generator(image_tensor, mask_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
        alpha = _build_alpha(mask, blend_width)
        blended = image * (1.0 - alpha) + generated * alpha
        canvas[ys, xs] = blended
    canvas = np.nan_to_num(canvas, nan = 0.0)
    canvas = _upscale_with_averages(canvas)
    canvas_uint8 = (((canvas + 1.0) * 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(canvas_uint8, mode = "L").save(generated_tile_path)
    return generated_tile_path

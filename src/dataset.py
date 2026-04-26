from pathlib import Path
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class ElevationDataset(Dataset):
    def __init__(self, root_dir="assets/data/elevations"):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.root_dir = project_root / self.root_dir

        self.image_paths = sorted([path for path in self.root_dir.glob("*.png")])
        self.tile_width = 64
        if not self.image_paths:
            raise ValueError(f"No PNG elevation tiles in {self.root_dir}")
        with Image.open(self.image_paths[0]) as first_image:
            ref_width, ref_height = first_image.size
        if ref_width != ref_height or ref_width % self.tile_width != 0:
            raise ValueError(
                f"{self.image_paths[0]} has invalid size {ref_width}x{ref_height}, "
                f"expected a square with side length divisible by {self.tile_width}"
            )
        self.tiles_per_side = ref_width // self.tile_width
        expected_width = ref_width
        self.index = []

        for image_path in tqdm.tqdm(self.image_paths, desc="Loading elevation dataset"):
            with Image.open(image_path) as image:
                width, height = image.size
            if width != expected_width or height != expected_width:
                raise ValueError(
                    f"{image_path} has invalid size {width}x{height}, "
                    f"expected {expected_width}x{expected_width}"
                )
            for y in range(self.tiles_per_side):
                for x in range(self.tiles_per_side):
                    self.index.append((image_path, x, y))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        image_path, tile_x, tile_y = self.index[idx]
        left = tile_x * self.tile_width
        upper = tile_y * self.tile_width
        right = left + self.tile_width
        lower = upper + self.tile_width

        with Image.open(image_path) as image:
            tile = image.crop((left, upper, right, lower))
            tile = tile.convert("L")
            tile_array = np.array(tile, dtype=np.float32)

        tile_array = (tile_array / 127.5) - 1.0

        tile_tensor = torch.from_numpy(tile_array).unsqueeze(0)
        return tile_tensor

def create_elevation_dataloader(batch_size=32, shuffle=True, num_workers=0, root_dir="assets/data/elevations"):
    dataset = ElevationDataset(root_dir=root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader

if __name__ == "__main__":
    dataset, dataloader = create_elevation_dataloader()
    print(len(dataset))
    print(len(dataloader))
    project_root = Path(__file__).resolve().parent.parent
    examples_dir = project_root / "data" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_count = min(128, len(dataset))
    for i in range(example_count):
        tile_tensor = dataset[i]
        tile_array = tile_tensor.squeeze(0).numpy()
        tile_uint8 = (((tile_array + 1.0) * 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(tile_uint8, mode="L").save(examples_dir / f"example_{i:03d}.png")
    print(f"saved {example_count} examples to {examples_dir}")

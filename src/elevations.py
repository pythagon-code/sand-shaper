import numpy as np
import os
import shutil
import tqdm
import zipfile
from pathlib import Path
from PIL import Image

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def copy_exports_images_to_elevations(
    exports_images_dir: str | Path | None = None,
    output_folder: str | Path | None = None,
) -> None:
    root = _project_root()
    if exports_images_dir is None:
        exports_images_dir = root / "exports" / "images"
    else:
        exports_images_dir = Path(exports_images_dir)
    if output_folder is None:
        output_folder = root / "assets" / "data" / "elevations"
    else:
        output_folder = Path(output_folder)
    if not exports_images_dir.is_dir():
        return
    output_folder.mkdir(parents=True, exist_ok=True)
    for path in sorted(exports_images_dir.glob("*.png")):
        shutil.copy2(path, output_folder / path.name)

def extract_elevations(source_dir="exports/elevations", output_folder="data/elevations"):
    if os.path.exists(os.path.join(output_folder, "done.txt")):
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    zip_files = [f for f in os.listdir(source_dir) if f.endswith('.zip')]

    for zip_name in zip_files:
        zip_path = os.path.join(source_dir, zip_name)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm.tqdm(zip_ref.namelist(), f"Extracting {zip_name}"):
                if member.lower().endswith('.hgt'):
                    filename = os.path.basename(member)
                    if not filename:
                        continue
                        
                    target_path = os.path.join(output_folder, filename)
                    
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

    with open(os.path.join(output_folder, "done.txt"), "w") as f:
        f.write("Done!")

def load_hgt_tile(file_path, output_folder="assets/data/elevations", skip=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = np.fromfile(file_path, dtype='>i2')
    size = int(len(data)**0.5)
    grid = data.reshape((size, size))
    grid = grid[:1024, :1024].astype(np.float32)
    downsampled_size = grid.shape[0] // skip
    grid = grid[:downsampled_size * skip, :downsampled_size * skip]
    grid = grid.reshape(downsampled_size, skip, downsampled_size, skip).mean(axis=(1, 3))

    min_value = grid.min()
    max_value = grid.max()
    if max_value == min_value:
        grid = np.zeros_like(grid, dtype=np.uint8)
    else:
        grid = ((grid - min_value) / (max_value - min_value) * 256).astype(np.uint8)

    output_filename = os.path.basename(file_path).replace(".hgt", ".png")
    output_path = os.path.join(output_folder, output_filename)
    img = Image.fromarray(grid, mode="L")
    img.save(output_path)
    
    return output_filename.rstrip(".png")

def load_all_hgt_tiles(source_dir="data/elevations", output_folder="assets/data/elevations"):
    if os.path.exists(os.path.join(output_folder, "done.txt")):
        return

    for file in tqdm.tqdm(os.listdir(source_dir), "Loading HGT tiles"):
        if file.endswith('.hgt'):
            load_hgt_tile(os.path.join(source_dir, file), output_folder)
    
    with open(os.path.join(output_folder, "done.txt"), "w") as f:
        f.write("Done!")

if __name__ == "__main__":
    copy_exports_images_to_elevations()
    print("Extracting elevations...")
    extract_elevations()
    print("Loading HGT tiles...")
    load_all_hgt_tiles()


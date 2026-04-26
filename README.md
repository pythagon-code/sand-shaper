# Sand Sifter

Training data in this project is designed around **[Viewfinder Panoramas](https://www.viewfinderpanoramas.org/dem3.html#hgt)** (Jonathan de Ferranti)—specifically the **15 arc-second** HGT tiles offered there. **You only need that dataset if you want to train your own model.** To **test or run** the app with the bundled **`model.pt`** and sample elevation PNGs (e.g. under **`exports/images/`**), you do not need to download DEM zips. If you do train, use that 15″ source so dimensions match the extract and conversion steps; **other DEM sources often use different grid sizes and may not work** without changes elsewhere in the pipeline.

**Sand Sifter** is a **3D continuous terrain generation engine**: it builds heightmap-based worlds that can **wrap or extend across boundaries** instead of stopping at a hard edge—useful for games and tools that need toroidal or seamless large maps.

The system trains a **generator** on **real Earth elevation** tiles, then at runtime **marches** over a grid: each step conditions on partially known neighbors (masked regions), **inpaints** missing height using the learned model, and **blends** overlaps so tiles meet without harsh seams. You can change the **terrain resolution** (grid size) in the interactive app; results are written as **PNG heightmaps** (grayscale = elevation) under `assets/data/terrains/` for reuse in other engines or pipelines.

---

## Setup

Use **Python 3.10 or newer**.

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/windows/) and enable **Add python.exe to PATH** in the installer.
2. Open **PowerShell** or **Command Prompt**, then go to the project folder (replace the path with yours):

   ```powershell
   cd C:\path\to\sand-sifter
   ```

3. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

4. Activate it:

   - **PowerShell:** `.\.venv\Scripts\Activate.ps1`  
     If execution policy blocks this, run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
   - **Command Prompt:** `.venv\Scripts\activate.bat`

5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Elevation data (training and generation)

**Training from raw DEMs:** Download **15 arc-second** HGT archives as **`.zip` files** from **[Viewfinder Panoramas — DEM3 / HGT](https://www.viewfinderpanoramas.org/dem3.html#hgt)**. **Any tile from that 15″ dataset should work.** Put **only** those zips into **`exports/elevations/`** (not loose `.hgt` files in that folder—the pipeline expects zip archives there). You only need **as much data as you want to download**—a subset of tiles is enough to train.

**HGT pipeline** (after the zips are in place)

1. Confirm **`exports/elevations/`** contains the **`.zip`** downloads (one zip per tile area, as provided by the site).
2. From the **`src`** directory, run:

   ```bash
   cd src
   python elevations.py
   ```

   This extracts `.hgt` files and writes grayscale PNG tiles under **`assets/data/elevations`**.

**Using PNGs in `exports/images/`**

If you already have **grayscale elevation PNGs** produced by this pipeline (square images whose side length is divisible by **64**), you can keep copies in **`exports/images/`**. When you start **training** or the **main program**, the project **copies** those PNGs into **`assets/data/elevations`**. That path is enough for **trying the viewer** without downloading zips, as long as **`model.pt`** is present.

---

## Generator checkpoint (`exports/models/`)

The 3D app loads the generator from **`exports/models/model.pt`**.

1. Obtain a trained checkpoint (your own training run or a file shared with you).
2. Copy it into **`exports/models/`** (create the folder if needed).
3. If the file is not already named **`model.pt`**, rename it to **`model.pt`** so the application can load it.

**Git and large files:** **`exports/models/`** can be committed when the checkpoint is small enough. GitHub rejects files **larger than 100 MB**; larger checkpoints should be shared via release assets, cloud storage, or [Git LFS](https://git-lfs.com/).

---

## Training

From the **`src`** directory (so imports like `from dataset import …` resolve):

```bash
cd src
python training.py
```

Training reads elevation PNGs from **`assets/data/elevations`** (after `elevations.py` and/or the `exports/images` copy step). It writes diagnostic images under **`data/images/`** and periodic checkpoints under **`data/models/`**. To use a new run in the viewer, copy the chosen **`.pt`** into **`exports/models/`** as **`model.pt`** (replacing the existing file if you are updating the default checkpoint).

---

## Running Sand Sifter (interactive viewer)

The main program is the **Ursina** 3D scene: terrain generation, expand/combined view, and grid-size slider.

```bash
cd src
python main.py
```

Ensure **`exports/models/model.pt`** exists and that **`assets/data/elevations`** is populated. Generated terrain PNGs are saved under **`assets/data/terrains/`**.

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── assets/                 # Static assets (e.g. window icon)
│   └── ocean.ico
├── exports/
│   ├── images/             # Optional: small set of elevation PNGs (can be versioned)
│   ├── models/             # Generator checkpoint: model.pt
│   └── elevations/         # Optional: HGT zip downloads (typically not committed; large)
├── src/
│   ├── main.py             # Interactive 3D viewer (Ursina)
│   ├── training.py         # GAN training loop
│   ├── terrain.py          # Tile marching, blending, heightmap export
│   ├── gan.py              # Generator / discriminator
│   ├── dataset.py          # Elevation PNG tiles → training batches
│   ├── elevations.py       # HGT extract / PNG conversion; syncs exports/images → assets
│   └── gui.py              # In-app controls
├── data/                   # Created at runtime (training previews, checkpoints, etc.); gitignored
├── assets/data/            # Created at runtime (elevations, generated terrains); gitignored
└── temp/                   # Gitignored scratch
```

Paths like `data/` and `assets/data/` are populated when you train, process elevations, or run the app. Only the layout above is guaranteed in the repository; generated folders may appear after first use.

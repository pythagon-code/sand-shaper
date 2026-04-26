from pathlib import Path
from PIL import Image
from elevations import copy_exports_images_to_elevations
from gui import create_expand_button
from gui import create_grid_size_slider
from terrain import generate_world_terrain
from ursina import *

project_root = Path(__file__).resolve().parent.parent
copy_exports_images_to_elevations()
checkpoint_name = "model.pt"
grid_size = 64
generation_index = 0
current_generated_tile_name = ""
current_combined_tile_name = ""

app = Ursina(title="Sand Sifter", icon=(project_root / "assets" / "ocean.ico").resolve())
application.asset_folder = project_root / "assets"
terrain: Entity | None = None
terrain_combined: Entity | None = None
ball: Entity | None = None

is_expanded = False


def _save_combined_tile(source_tile_path: Path, combined_tile_path: Path) -> None:
    source_image = Image.open(source_tile_path).convert("L")
    combined_image = Image.new("L", (source_image.width * 3, source_image.height * 3))
    for y in range(3):
        for x in range(3):
            combined_image.paste(source_image, (source_image.width * x, source_image.height * y))
    combined_image.save(combined_tile_path)


def _regenerate_terrain(world_size: int) -> None:
    global current_combined_tile_name
    global current_generated_tile_name
    global generation_index
    generation_index += 1
    generated_tile_name = f"generated_terrain_{world_size}_{generation_index:04d}"
    combined_tile_name = f"generated_terrain_3x3_{world_size}_{generation_index:04d}"
    combined_tile_path = project_root / "assets" / "data" / "terrains" / f"{combined_tile_name}.png"
    generated_tile_path = generate_world_terrain(
        project_root = project_root,
        generated_tile_name = generated_tile_name,
        checkpoint_name = checkpoint_name,
        world_size = world_size,
    )
    _save_combined_tile(generated_tile_path, combined_tile_path)
    current_generated_tile_name = generated_tile_name
    current_combined_tile_name = combined_tile_name
    if terrain is not None and terrain_combined is not None:
        terrain.model = Terrain(f"data/terrains/{generated_tile_name}")
        terrain.collider = "mesh"
        terrain.scale = (world_size, 20, world_size)
        terrain_combined.model = Terrain(f"data/terrains/{combined_tile_name}")
        terrain_combined.collider = "mesh"
        terrain_combined.scale = (world_size * 3, 20, world_size * 3)


def _toggle_expand() -> None:
    global is_expanded
    is_expanded = not is_expanded
    terrain.enabled = not is_expanded
    terrain_combined.enabled = is_expanded
    expand_button.text = "Reduce" if is_expanded else "Expand"


def _on_slider_change() -> None:
    global grid_size
    next_grid_size = int(grid_size_slider.value)
    if next_grid_size == grid_size:
        return
    grid_size = next_grid_size
    _regenerate_terrain(grid_size)


def _active_terrain() -> Entity | None:
    if terrain is None or terrain_combined is None:
        return None
    return terrain_combined if is_expanded else terrain


def _spawn_ball() -> None:
    global ball
    ball = Entity(
        model = "sphere",
        color = color.azure,
        collider = "sphere",
        position = (0, 50, 0),
        scale = 16,
    )


def update() -> None:
    return


_regenerate_terrain(grid_size)
terrain = Entity(
    model = Terrain(f"data/terrains/{current_generated_tile_name}"),
    scale = (grid_size, 15, grid_size),
    texture = "rainbow",
    collider = "mesh",
)
terrain_combined = Entity(
    model = Terrain(f"data/terrains/{current_combined_tile_name}"),
    position = (0, 0, 0),
    scale = (grid_size * 3, 15, grid_size * 3),
    texture = "rainbow",
    collider = "mesh",
)
terrain_combined.enabled = False

EditorCamera()
Sky()
_spawn_ball()
expand_button = create_expand_button(_toggle_expand)
grid_size_slider = create_grid_size_slider(_on_slider_change)

app.run()
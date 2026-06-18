import os
import pathlib

from ocsmesh import Geom, Hfun, MeshDriver, Raster


current_dir = pathlib.Path(__file__).parent.absolute()
tile_dir = current_dir.parent / "redsea_tiles"
output_dir = str(current_dir)

os.makedirs(output_dir, exist_ok=True)


def describe_channels(raster: Raster, level: float, width: float, target_size: float, expansion_rate: float) -> None:
    channels = raster.get_channels(level=level, width=width, tolerance=None)
    geom_type = getattr(channels, "geom_type", type(channels).__name__)
    is_empty = getattr(channels, "is_empty", None)
    print(f"  get_channels: type={geom_type} empty={is_empty}")

    hfun = Hfun([raster], hmin=100, hmax=8000)
    hfun.execution_mode = "serial"

    try:
        hfun.add_channel(
            level=level,
            width=width,
            target_size=target_size,
            expansion_rate=expansion_rate,
        )
        geom = Geom([raster], zmax=0)
        driver = MeshDriver(geom, hfun, engine_name="gmsh")
        driver.run()
        print("  add_channel: ok")
    except Exception as exc:
        print(f"  add_channel: {type(exc).__name__}: {exc}")


def main() -> None:
    tile_paths = sorted(tile_dir.glob("redsea_tile_*.tif"))
    print(f"Testing {len(tile_paths)} tiles from: {tile_dir}")

    for tile_path in tile_paths:
        print(f"\n{tile_path.name}")
        raster = Raster(tile_path)
        describe_channels(
            raster=raster,
            level=0,
            width=2000,
            target_size=300,
            expansion_rate=0.001,
        )


if __name__ == "__main__":
    main()
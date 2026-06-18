import os
import pathlib
from shapely.geometry import box
from ocsmesh import Raster, Geom, Hfun, Mesh, MeshDriver
# Set up paths relative to this script
current_dir = pathlib.Path(__file__).parent.absolute()
tile_dir = "../redsea_tiles"
output_dir = str(current_dir)

os.makedirs(output_dir, exist_ok=True)

# 2. Load the 4 tile
tile_rasters_hfun = [
    Raster(os.path.join(tile_dir, f"redsea_tile_{i}.tif"))
    for i in range(1, 5)
]

tile_rasters_geom = [
    Raster(os.path.join(tile_dir, f"redsea_tile_{i}.tif"))
    for i in range(1, 5)
]


print(f"Loaded {len(tile_rasters_hfun)} tile rasters for testing")
geom = Geom(tile_rasters_geom, zmax=100)
# Patch box (adjust min/max lon/lat for your Red Sea extent)
patch_box = box(32.5, 28.0, 33.5, 29.0)



# 3. Serial — one raster with add_patch
hfun_serial = Hfun(tile_rasters_hfun, hmin=100, hmax=8000)
hfun_serial.execution_mode = 'serial'
hfun_serial.add_patch(shape=patch_box, target_size=3000, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_serial, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "patch_serial.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


# 4. Parallel — 4 tile rasters with add_patch
hfun_parallel = Hfun(tile_rasters_hfun, hmin=100, hmax=8000, nprocs=4)
hfun_parallel.execution_mode = 'parallel'
hfun_parallel.add_patch(shape=patch_box, target_size=3000, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_parallel, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "patch_parallel.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


print("\nPatch test mesh files generated successfully in:", output_dir)

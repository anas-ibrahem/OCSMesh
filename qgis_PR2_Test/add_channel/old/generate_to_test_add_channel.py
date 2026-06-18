import os
import pathlib
from ocsmesh import Raster, Geom, Hfun, Mesh, MeshDriver

# Set up paths relative to this script
current_dir = pathlib.Path(__file__).parent.absolute()
raster_path = "./redsea.tif"
tile_dir = "./redsea_tiles"
output_dir = str(current_dir)

os.makedirs(output_dir, exist_ok=True)

# 1. Base Geometry
raster_for_geom = Raster(raster_path)

# 2. Load the 4 tile rasters
tile_rasters = [
    Raster(os.path.join(tile_dir, f"redsea_tile_{i}.tif"))
    for i in range(1, 5)
]
print(f"Loaded {len(tile_rasters)} tile rasters for testing")
geom = Geom(tile_rasters, zmax=0)


# 3. No Refinements (single raster baseline)
raster_for_hfun = Raster(raster_path)
hfun = Hfun(raster_for_hfun, hmin=100, hmax=8000)
driver = MeshDriver(geom, hfun, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "channel_no_refinement.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


# 4. Serial — one raster with add_channel
hfun_serial = Hfun([raster_for_hfun], hmin=100, hmax=8000)
hfun_serial.execution_mode = 'serial'
hfun_serial.add_channel(level=0, width=2000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_serial, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "channel_serial_one_raster.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


# 5. Parallel — 4 tile rasters with add_channel
hfun_parallel = Hfun(tile_rasters, hmin=100, hmax=8000, nprocs=4)
hfun_parallel.execution_mode = 'parallel'
hfun_parallel.add_channel(level=0, width=2000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_parallel, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "channel_parallel.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


print("\nChannel test mesh files generated successfully in:", output_dir)

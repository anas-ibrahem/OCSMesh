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
# 

# 2. Load the 4 tile rasters
tile_rasters_hfun = [
    Raster(os.path.join(tile_dir, f"redsea_tile_{i}.tif"))
    for i in range(1, 5)
]

tile_rasters_geom = [
    Raster(os.path.join(tile_dir, f"redsea_tile_{i}.tif"))
    for i in range(1, 5)
]

# raster_for_geom = Raster(tile_rasters)
geom = Geom(tile_rasters_geom, zmax=0)

print(f"Loaded {len(tile_rasters_hfun)} tile rasters for testing")


# 3. Serial — one raster with add_channel

hfun_serial = Hfun(tile_rasters_hfun, hmin=500, hmax=8000)
hfun_serial.execution_mode = 'serial'
hfun_serial.add_channel(level=0, width=3000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_serial, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "channel_serial_one_raster.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


# 4. Parallel — 4 tile rasters with add_channel

hfun_parallel = Hfun(tile_rasters_hfun, hmin=500, hmax=8000, nprocs=4)
hfun_parallel.execution_mode = 'parallel'
hfun_parallel.add_channel(level=0, width=3000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_parallel, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "channel_parallel.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


# Serial one raster file

raster_for_hfun = Raster(raster_path)
geom = Geom([raster_for_hfun], zmax=0)

hfun_serial = Hfun(raster_for_hfun, hmin=500, hmax=8000)
hfun_serial.execution_mode = 'serial'
hfun_serial.add_channel(level=0, width=3000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_serial, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "serial_one_raster.2dm")
mesh.write(out_file, format='2dm', overwrite=True)

print("\nChannel test mesh files generated successfully in:", output_dir)

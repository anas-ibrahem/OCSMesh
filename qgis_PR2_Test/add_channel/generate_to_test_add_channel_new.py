import os
import pathlib
from ocsmesh import Raster, Geom, Hfun, Mesh, MeshDriver
import time
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


# 4. Serial
hfun_serial = Hfun(tile_rasters_hfun, hmin=100, hmax=8000)
hfun_serial.execution_mode = 'serial'
hfun_serial.add_channel(level=0, width=100000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_serial, engine_name='gmsh')
time_start = time.time()
mesh = driver.run()
time_end = time.time()
print(f"Serial meshing completed in {time_end - time_start:.2f} seconds")
out_file = os.path.join(output_dir, "channel_serial.2dm")
mesh.write(out_file, format='2dm', overwrite=True)

# 5. Parallel
hfun_parallel = Hfun(tile_rasters_hfun, hmin=100, hmax=8000, nprocs=4)
hfun_parallel.execution_mode = 'parallel'
hfun_parallel.add_channel(level=0, width=100000, target_size=300, expansion_rate=0.001)
driver = MeshDriver(geom, hfun_parallel, engine_name='gmsh')
time_start = time.time()
mesh = driver.run()
time_end = time.time()
print(f"Parallel meshing completed in {time_end - time_start:.2f} seconds")
out_file = os.path.join(output_dir, "channel_parallel.2dm") 
mesh.write(out_file, format='2dm', overwrite=True)



print("\nChannel test mesh files generated successfully in:", output_dir)

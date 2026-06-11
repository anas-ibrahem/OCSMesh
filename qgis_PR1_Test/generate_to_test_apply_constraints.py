import os
import pathlib
from ocsmesh import Raster, Geom, Hfun, Mesh, MeshDriver
import time

# Set up paths relative to this script
current_dir = pathlib.Path(__file__).parent.absolute()
raster_path = "./redsea.tif"
tile_dir = "./redsea_tiles"
output_dir = str(current_dir)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 1. Base Geometry (use the full raster for domain boundary)
raster_for_geom = Raster(raster_path)
geom = Geom(raster_for_geom, zmax=0)

# 2. Load the 4 tile rasters for multi-raster parallel testing
tile_rasters = [
    Raster(os.path.join(tile_dir, f"redsea_tile_{i}.tif"))
    for i in range(1, 5)
]
print(f"Loaded {len(tile_rasters)} tile rasters for testing")


# 3. Mesh with No Constraints (single raster baseline)
raster_for_hfun = Raster(raster_path)
hfun = Hfun(raster_for_hfun, hmin=100, hmax=8000)
driver = MeshDriver(geom, hfun, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "mesh_no_constraints.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


################### DISCARDED ###################
# # 4. Mesh with Serial Constraint Application (4 tile rasters, serial mode)
# hfun_serial = Hfun(tile_rasters, hmin=100, hmax=8000)
# hfun_serial.execution_mode = 'serial'
# hfun_serial.add_topo_bound_constraint(value=3500, upper_bound=0, value_type='max')
# driver = MeshDriver(geom, hfun_serial, engine_name='gmsh')
# mesh = driver.run()
# out_file = os.path.join(output_dir, "serial.2dm")
# mesh.write(out_file, format='2dm', overwrite=True)

#4.2 Serial but one raster tile (raster_for_hfun)
hfun_serial_one = Hfun(raster_for_hfun, hmin=100, hmax=8000)
hfun_serial_one.execution_mode = 'serial'
hfun_serial_one.add_topo_bound_constraint(value=3500, upper_bound=0, value_type='max')
driver = MeshDriver(geom, hfun_serial_one, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "serial_one_raster.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


# 5. Mesh with Parallel Constraint Application (4 tile rasters, parallel mode)
hfun_parallel = Hfun(tile_rasters, hmin=100, hmax=8000, nprocs=4)
hfun_parallel.execution_mode = 'parallel'
hfun_parallel.add_topo_bound_constraint(value=3500, upper_bound=0, value_type='max')
driver = MeshDriver(geom, hfun_parallel, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "parallel.2dm")
mesh.write(out_file, format='2dm', overwrite=True)


print("\nMesh files generated successfully in:", output_dir)

"""
Fast method reference — for visual comparison only, not for serial/parallel testing.
'fast' uses a lower-resolution merged raster approach; it does NOT support
execution_mode='parallel' for constraints.
"""
import os
import pathlib
from shapely.geometry import box
from ocsmesh import Raster, Geom, Hfun, Mesh, MeshDriver

# Set up paths relative to this script
current_dir = pathlib.Path(__file__).parent.absolute()
raster_path = "../redsea.tif"
output_dir = str(current_dir)

os.makedirs(output_dir, exist_ok=True)

# Base Geometry
raster_for_geom = Raster(raster_path)
geom = Geom(raster_for_geom, zmax=100)
hfun_rast_list = [Raster(raster_path)]
patch_box = box(32.5, 28.0, 33.5, 29.0)


# 1. Fast — no refinements
hfun = Hfun(hfun_rast_list, base_shape_crs=geom.crs, hmin=100, hmax=8000, method='fast')
driver = MeshDriver(geom, hfun, engine_name='gmsh')
mesh = driver.run()
out_file = os.path.join(output_dir, "fast_no_refinement.2dm")
mesh.write(out_file, format='2dm', overwrite=True)



print("\nFast reference mesh files generated successfully in:", output_dir)

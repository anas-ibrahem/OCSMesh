import os
import tempfile
import urllib.request
import shutil
from pathlib import Path

from rasterio.enums import Resampling

from ocsmesh.raster import Raster


# Find a better way!
tif_url = (
    'https://chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n40x75_w073x75_2015v1.tif'
)

# Save permanently in a local `test_data` directory rather than /tmp
test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')
os.makedirs(test_data_dir, exist_ok=True)
TEST_FILE = os.path.join(test_data_dir, 'test_dem.tif')

if not Path(TEST_FILE).exists():
    import ssl
    # Bypass Windows certificate store parsing bug in Python < 3.11
    ssl_context = ssl._create_unverified_context()
    
    tmpfd, tmppath = tempfile.mkstemp()
    
    # Pass the context using urlopen since urlretrieve doesn't accept context directly in older Pythons easily,
    # or just use a custom opener. We will use urlopen to read and write to the file.
    with urllib.request.urlopen(tif_url, context=ssl_context) as response, open(tmppath, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
    os.close(tmpfd)
    r = Raster(tmppath)
    r.resampling_method = Resampling.average
    r.resample(scaling_factor=0.01)
    r.save(TEST_FILE)

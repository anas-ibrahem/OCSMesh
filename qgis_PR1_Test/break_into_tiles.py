import rasterio
from rasterio.windows import Window
import numpy as np
import os

src_path = './redsea.tif'
out_dir = './redsea_tiles'
os.makedirs(out_dir, exist_ok=True)

with rasterio.open(src_path) as src:
    w = src.width   # 528
    h = src.height  # 960
    
    # Split into 4 tiles: 2 rows x 2 cols
    half_w = w // 2
    half_h = h // 2
    
    tiles = {
        'redsea_tile_1.tif': Window(0, 0, half_w, half_h),         # top-left
        'redsea_tile_2.tif': Window(half_w, 0, w - half_w, half_h),  # top-right
        'redsea_tile_3.tif': Window(0, half_h, half_w, h - half_h),  # bottom-left
        'redsea_tile_4.tif': Window(half_w, half_h, w - half_w, h - half_h),  # bottom-right
    }
    
    for name, window in tiles.items():
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update({
            'width': window.width,
            'height': window.height,
            'transform': transform,
        })
        
        out_path = os.path.join(out_dir, name)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(src.read(window=window))
        
        print(f'Created {name}: {window.width}x{window.height}')

print('Done! Tiles saved to:', out_dir)
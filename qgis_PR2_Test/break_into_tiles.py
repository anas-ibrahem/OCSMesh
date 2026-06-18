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
    


    overlap = 0.2  # 20% overlap
    tile_width = int(w / 2 * (1 + overlap))  # width of each tile with overlap
    tile_height = int(h / 2 * (1 + overlap)) # height of each tile with overlap

    tiles = {
        'redsea_tile_1.tif': Window(0, 0, tile_width, tile_height),  # top-left
        'redsea_tile_2.tif': Window(w - tile_width, 0, tile_width, tile_height),  # top-right
        'redsea_tile_3.tif': Window(0, h - tile_height, tile_width, tile_height),  # bottom-left
        'redsea_tile_4.tif' : Window(w - tile_width, h - tile_height, tile_width, tile_height),  # bottom-right
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
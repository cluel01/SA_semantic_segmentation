import math
import numpy as np
from rasterio.windows import Window

def window_generator(width,height,patch_size,step_size,start_end=None):   
   
    nx = math.ceil(width / step_size)
    ny = math.ceil(height / step_size)
    total = nx * ny
    
    if start_end is None:
        y = 0
        x = 0
        area = (0,total)
    else:
        y = start_end[0] // nx 
        x = start_end[0] % nx
        area = start_end

    for _ in range(*area):
        h = y * step_size
        w = x * step_size
        win = Window(w,h,*patch_size)
        yield(win)
        if (x != nx-1) and (y != ny-1):
            x += 1
        elif (x == nx-1) and (y != ny-1):
            x = 0
            y += 1
        elif (x != nx-1) and (y == ny-1):
            x += 1
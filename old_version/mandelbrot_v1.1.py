''' Version Table
v1.0 First workable version with concurrent parallel
v1.1 Using numba jit on the core complex iteration
'''

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import numba as nb



def mandelbrot(center, radius, size, iter=int(1e4), num_workers=1, max_iter=None):
    # Basic iteration: 10000
    iter = 10000 if iter is None else int(iter)
    max_iter = iter if max_iter is None else int(max_iter)
    # Support square mode (e.g., size=1024) or rectangle mode (e.g., size=[1024, 768])
    if type(size) is not list: size = [size, size]
    size = [s+1 if s%2 else s for s in size]
    half_size = [(s-1)//2 for s in size]
    # Set resolution
    res = radius*2/min(size)
    
    # One dimensional (width*height) complex
    C = np.zeros(size[0]*size[1], dtype=np.csingle)
    # Two dimensional (height, width) recording iteration of convergence for every pixel
    I = np.zeros((size[1], size[0]), dtype=np.uint)
    # Two dimensional (width*height, 2) holding position of I in C
    pos = np.zeros((size[0]*size[1], 2), dtype=np.uint)
    
    # Init values
    cnt = 0
    for r in range(size[0]):
        for i in range(size[1]):
            C[cnt] = center+(r-half_size[0])*res+(half_size[1]-i)*res*1j
            pos[cnt, :] = [i,r]
            cnt+=1
    
    # Split C, I, pos into (as possibly) equal portion along width for parallel processing
    split_size = size[0]//num_workers
    split_points = [_*split_size for _ in range(num_workers)]+[size[0]]
    results = []
    progress = []
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        for i in range(len(split_points)-1):
            results.append(exe.submit(_mandelbrot,
                C[size[1]*split_points[i]:size[1]*split_points[i+1]],
                pos[size[1]*split_points[i]:size[1]*split_points[i+1], :],
                I[:, split_points[i]:split_points[i+1]], 
                split_points[i], iter,
                progress))
        for iter_cnt in tqdm(range(0, iter, 1), ncols=100, ascii=True):
            while len([None for p in progress if p==iter_cnt]) < num_workers: time.sleep(0.5)
            while len(progress) and progress[0]<=iter_cnt: progress.pop(0)
        for i, r in enumerate(results):
            r.result()
    
    # Turn iteration of convergence into opencv image (height, width, 3)
    I = np.log10(I+1)
    print(I.max())
    I = np.clip(I.astype(np.float)/np.log10(max_iter+1), 0, 1)
    # Color map: Icy (White -> Cyan(w-r) -> Blue(cy-g) -> Black)
    return np.stack([
        np.clip(I*765, 0, 255).astype(np.uint8),
        (np.clip(I*765, 255, 510)-255).astype(np.uint8),
        (np.clip(I*765, 510, 765)-510).astype(np.uint8)
         ], axis=-1)



def _mandelbrot(C, pos, I, split_from, iter, progress):
    # Moving values initialized
    Z = np.zeros_like(C)
    # Since C is only a splited part, pos must be updated for the split I
    pos[:,1]-=split_from
    ''' to-do: show the progress of the slowest thread only '''
    for iter_cnt in range(0, iter, 1): #tqdm(range(iter), ncols=100, ascii=True) if is_print else 
        # Update values according to the rules of Mandelbrot set
        Z = _iterate(Z, C)
        progress.append(iter_cnt)
        # Criterion of divergence
        diverged = np.abs(Z)>2
        
        # Newly diverged points are found:
        # "Draw" I with the number of iteration
        for pos_to_draw in pos[diverged, :]: I[pos_to_draw[0], pos_to_draw[1]] = iter_cnt+1
        # Remove diverged points from Z, C, and pos, for faster future calculation
        Z = Z[~diverged]    #= np.delete(Z, diverged, axis=0)
        C = C[~diverged]    #np.delete(C, diverged, axis=0)
        pos = pos[~diverged, :] #np.delete(pos, diverged, axis=0)
    return I
    
    
    
@nb.jit
def _iterate(Z, C):
    #for i in range(10): Z = np.square(Z)+C
    return np.square(Z)+C
    
    
if __name__=="__main__":
    import cv2, os
    ## Now
    #c = -0.48109+0.614645j
    #r = 5e-6
    ## Start
    c, r = -0.5+0j, 1
    # iter = int(1e4)
    # img = mandelbrot(c, r, [1920, 1080], iter, num_workers=6)
    # cv2.imwrite("{}_{}_ice.png".format(str(c)[1:-1], str(r)), img)
    ## Find new site
    import time
    start = time.time()
    #c, r = 0.2501+0j, 1e-4
    img = mandelbrot(c, r, [1280, 720], iter=5e3, num_workers=6)
    cv2.imwrite("journey.png", img)
    print("Take {:.5f} sec".format(time.time()-start))
        
        
    
    
''' Version Table
v1.2    Using parallel on all parallelable matrix operation
v1.2.1  Continuous Calculation, cuda
v1.2.2  Thread for jit and cuda
'''
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time, os
import numba as nb
import cupy as cp


def mandelbrot(center=None, radius=None, size=None, iter=int(1e4), max_iter=None, cuda=False, ckpt=None):
    ''' to-do: checkpoint file mode '''
    # Basic iteration: 10000
    iter = 10000 if iter is None else int(iter)
    max_iter = iter if max_iter is None else int(max_iter)
    
    # Checkpoint found. Load variables.
    if ckpt is not None and os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            try: C, I, pos, Z, last_iter = [np.load(f) for _ in range(5)]
            except:
                print("This checkpoint may be damaged.")
                return None
        
    # No checkpoint, basic info is needed to start a new job.
    else:
        assert all([var is not None for var in [center, radius, size]]), "center, radius, and size are necessary if no checkpoint is provided."
            
        # Init necessary variables
        # Support square mode (e.g., size=1024) or rectangle mode (e.g., size=[1024, 768])
        if type(size) is not list: size = [size, size]
        size = [s+1 if s%2 else s for s in size]
        half_size = [(s-1)//2 for s in size]
        # Set resolution
        res = radius*2/min(size)
        # Starting point of iteration
        last_iter = 0
        
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
        Z = np.zeros_like(C)
    


    # Prepare for cuda
    if cuda: Z, C, I, pos = [cp.asarray(mat) for mat in [Z, C, I, pos]]
    
    # Core iteration
    if ckpt is not None: print(f"Doing {last_iter+1} to {iter} iteration.")
    for iter_cnt in tqdm(range(last_iter, iter), ncols=100, ascii=True):
    
        # Update values according to the rules of Mandelbrot set
        if cuda: Z, C, pos = _iterate_cuda(Z, C, I, pos, iter_cnt)
        else: Z, C, pos = _iterate(Z, C, I, pos, iter_cnt)
    
    # Move cuda object back to cpu
    if cuda:
        C, I, pos, Z = [cp.asnumpy(mat) for mat in [C, I, pos, Z]]
        
    # Save checkpoint
    if ckpt is not None:
        with open(ckpt, 'wb') as f:
            for mat in [C, I, pos, Z, iter]: np.save(f, mat)
            
    # Turn iteration of convergence into opencv image (height, width, 3)
    I = np.log10(I+1)
    I = np.clip(I.astype(np.float)/np.log10(max_iter+1), 0, 1)
    
    # Color map: Icy (White -> Cyan(w-r) -> Blue(cy-g) -> Black)
    I *= 765
    return np.stack([
        np.clip(I, 0, 255).astype(np.uint8),
        (np.clip(I, 255, 510)-255).astype(np.uint8),
        (np.clip(I, 510, 765)-510).astype(np.uint8)
         ], axis=-1)
    # Color map: Ice and Flame (Blue -> Cyan -> White -> Gray -> Red -> Yellow -> White)



@nb.jit(nopython=True)
def _iterate(Z, C, I, pos, iter_cnt):
    # One iteration
    Z = np.square(Z)+C
    
    # Criterion of divergence
    diverged = np.abs(Z)>2
    if not diverged.any(): return Z, C, pos
    
    # Newly diverged points are found:
    # "Draw" I with the number of iteration
    for target in pos[diverged, :]: I[target[0], target[1]] = iter_cnt+1
    
    # Remove diverged points from Z, C, and pos, for faster future calculation
    Z = Z[~diverged]
    C = C[~diverged]
    pos = pos[~diverged, :]
    return Z, C, pos
    
    
    
def _iterate_cuda(Z, C, I, pos, iter_cnt):
    # One iteration
    Z = cp.square(Z)+C
    
    # Criterion of divergence
    diverged = cp.abs(Z)>2
    if not diverged.any(): return Z, C, pos
    
    # Newly diverged points are found:
    # "Draw" I with the number of iteration
    for target in pos[diverged, :]: I[target[0], target[1]] = iter_cnt+1
    
    # Remove diverged points from Z, C, and pos, for faster future calculation
    Z = Z[~diverged]
    C = C[~diverged]
    pos = pos[~diverged, :]
    return Z, C, pos
    
    
if __name__=="__main__":
    import cv2, os
    ## Redo
    redo_dict = {
        "o":[-0.48109+0.61465j, 2e-5],
        # "p":[-0.48109+0.61465j, 1e-5]
        # "a":[-0.5+0j,   1],
        # "b":[-0.5+0.5j, 0.5], 
        # "c":[-0.5+0.6j, 0.2], 
        # "d":[-0.5+0.6j, 0.1], 
        # "e":[-0.5+0.6j, 0.05],
        # "f":[-0.48+0.62j,   0.02],
        # "g":[-0.48+0.62j,   0.01],
        # "h":[-0.481+0.615j, 5e-3],
        # "i":[-0.481+0.615j, 2e-3], 
        # "j":[-0.481+0.615j, 1e-3], 
        # "k":[-0.4811+0.6146j, 5e-4], 
        # "l":[-0.4811+0.6146j, 2e-4],
        # "m":[-0.4811+0.6146j, 1e-4],
        # "n":[-0.48109+0.61465j, 5e-5]
    }
    
    ## Start
    #c, r = -0.5+0j, 1
    # iter = int(1e4)
    # img = mandelbrot(c, r, [1920, 1080], iter, num_workers=6)
    # cv2.imwrite("{}_{}_ice.png".format(str(c)[1:-1], str(r)), img)
    ## Find new site
    #c, r = 0.2501+0j, 1e-4
    for name, (c, r) in redo_dict.items():
        img = mandelbrot(-0.48109+0.61465j, 1e-5, [1024, 1024], iter=5e5, cuda=False, ckpt=name+"_rerun")
        cv2.imwrite("{}{}_{}_ice.png".format(name, str(c)[1:-1], r), img)
        
        
    
    
''' Version Table
v1.2    Using parallel on all parallelable matrix operation
v1.2.1  Continuous Calculation, cuda
v1.2.2  Thread for jit and cuda
'''
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
from time import time, sleep
import numba as nb
import cupy as cp


def mandelbrot(center=None, radius=None, size=None, ckpt=None,
               iter=int(1e4), max_iter=None,
               cuda=False, max_workers=None):
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
        # Determine data type to be used
        if res > 1e-7: dtype = np.csingle
        elif res > 1e-15: dtype = np.cdouble
        else: dtype = np.clongdouble
        
        # One dimensional (width*height) complex
        C = np.zeros(size[0]*size[1], dtype=dtype)
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
    if ckpt is not None:
        assert last_iter < iter, "New iteration target is smaller than that of previous checkpoint."
        print(f"Doing {last_iter+1} to {iter} iteration.")
    # Single thread condition
    if max_workers is None or max_workers == 1:
        start = time()
        for iter_cnt in tqdm(range(last_iter, iter), ncols=100, ascii=True):
        
            # Update values according to the rules of Mandelbrot set
            if cuda: Z, C, pos = _iterate_cuda(Z, C, I, pos, iter_cnt)
            else: Z, C, pos = _iterate(Z, C, I, pos, iter_cnt)
        print("Done in {:.2f}s".format(time()-start))
            
    # Multithread condition
    else:
        if cuda: raise NotImplementedError
        
        # Progress report
        progress = [0]*max_workers
        results = []
        # Split
        split = list(range(0, size[0], size[0]//max_workers))
        if len(split) < max_workers+1: split += [size[0]]
        else: split[-1] = size[0]
        # Start
        start = time()
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, (_s, _e) in enumerate(zip(split[:-1], split[1:])):
                results.append( exe.submit( _looping,
                # exe.submit( _looping,
                    Z[_s*size[1]:_e*size[1]], C[_s*size[1]:_e*size[1]],
                    I, pos[_s*size[1]:_e*size[1], :],
                    #I[:, _s:_e], pos[_s*size[1]:_e*size[1], :]-np.asarray([0, _s]),
                    last_iter, iter, progress, i ) )
            for i in tqdm(range(last_iter, iter), ncols=100, ascii=True):
                while min(progress) <= i:
                    sleep(1)
                    pass
            results = [r.result() for r in results]
        print("Done in {:.2f}s".format(time()-start))
        [Z, C, pos] = [np.concatenate([r[i] for r in results], axis=0) for i in range(3)]
        
        
    # Move cuda object back to cpu
    if cuda: C, I, pos, Z = [cp.asnumpy(mat) for mat in [C, I, pos, Z]]
        
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
    # Color map: Ice and Flame (Black -> Blue -> Cyan -> White -> Black -> Red -> Yellow -> White)



def _looping(Z, C, I, pos, last_iter, iter, progress, i):
    for iter_cnt in range(last_iter, iter):
        Z, C, pos = _iterate(Z, C, I, pos, iter_cnt)
        progress[i] = iter_cnt+1
    return Z, C, pos

@nb.jit(nopython=True, nogil=True)
def _iterate(Z, C, I, pos, iter_cnt):
    # One iteration
    Z = np.square(Z) + C
    # Z = _core(Z, C)
    
    # Criterion of divergence
    diverged = np.abs(Z)>2
    # diverged = absZ > 2
    if not diverged.any(): return Z, C, pos
    
    # Newly diverged points are found:
    # "Draw" I with the number of iteration
    for target in pos[diverged, :]: I[target[0], target[1]] = iter_cnt+1
    
    # Remove diverged points from Z, C, and pos, for faster future calculation
    Z = Z[~diverged]
    C = C[~diverged]
    pos = pos[~diverged, :]
    return Z, C, pos
    
    
# @nb.jit(nopython=True)
# def _core(Z, C):
    # Z = np.square(Z)+C
    # return Z
    
# @nb.jit(nopython=True)
# def _judge(Z):
    # return np.abs(Z)
    
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
    name = 'test'
    ## Redo
    redo_dict = {
        
        # "w": [-0.48108935+0.6146492j, 5e-8],
        # "x": [-0.48108935+0.6146492j, 2e-8],
        # "y": [-0.48108935+0.6146492j, 1e-8],
        "z": [-0.48108935+0.6146492j, 5e-9],
        # "w": [-0.48108935+0.6146492j, 5e-8],
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
        # "n":[-0.48109+0.61465j, 5e-5],
        # "o":[-0.48109+0.61465j, 2e-5],
        # "p":[-0.48109+0.61465j, 1e-5],
        # "q":[-0.48109+0.61465j, 5e-6],
        # "r": [-0.48109+0.61465j, 2e-6],
        # "s": [-0.4810894+0.6146492j, 1e-6],
        # "t": [-0.4810894+0.6146492j, 5e-7],
        # "u": [-0.4810894+0.6146492j, 2e-7],
        # "v": [-0.4810894+0.6146492j, 1e-7],
    }
    
    ## Start
    c, r = -0.5+0j, 1
    # iter = int(1e4)
    # img = mandelbrot(c, r, [1920, 1080], iter, num_workers=6)
    # cv2.imwrite("{}_{}_ice.png".format(str(c)[1:-1], str(r)), img)
    ## Find new site
    #c, r = 0.2501+0j, 1e-4
    #start = time.time()
    img = mandelbrot(c, r, [1024, 1024], iter=1e4, max_workers=6)
    #print("Done in {:.2f}s".format(time.time()-start))
    cv2.imwrite("{}{}_{}_ice.png".format(name, str(c)[1:-1], r), img)
    # for name, (c, r) in redo_dict.items():
        # img = mandelbrot(c, r, [1024, 1024], iter=1e6, max_workers=6, ckpt=name+".ckpt")
        # cv2.imwrite("{}{}_{}_ice.png".format(name, str(c)[1:-1], r), img)
        
        
    
    
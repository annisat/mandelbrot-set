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
from collections import namedtuple
Section = namedtuple("Section", "up, down, high")


def mandelbrot(center=None, radius=None, size=None, iter=int(1e4), max_iter=None,
               ckpt=None, cuda=False, max_workers=None):
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
        if res > 1e-6: dtype = np.csingle
        elif res > 1e-14: dtype = np.cdouble
        else: dtype = np.clongdouble
        
        # One dimensional (width*height) complex
        C = np.zeros(size[0]*size[1], dtype=dtype) #if not cuda else np.zeros((size[1], size[0]), dtype=dtype)
        # Two dimensional (height, width) recording iteration of convergence for every pixel
        I = np.zeros((size[1], size[0]), dtype=np.uint)
        # Two dimensional (width*height, 2) holding position of I in C
        pos = np.zeros((size[0]*size[1], 2), dtype=np.uint)
    
        # Init values
        cnt = 0
        for r in range(size[0]):
            for i in range(size[1]):
                if False: C[i, r] = center+(r-half_size[0])*res+(half_size[1]-i)*res*1j
                else: C[cnt] = center+(r-half_size[0])*res+(half_size[1]-i)*res*1j
                pos[cnt, :] = [i,r]
                cnt+=1
        Z = np.zeros_like(C)
    


    # Prepare for cuda
    if cuda:
        Z, C, I, pos = [cp.asarray(mat) for mat in [Z, C, I, pos]]   #, pos
        # print(Z.nbytes, C.nbytes, I.nbytes, pos.nbytes)
        # print((Z.nbytes+C.nbytes+I.nbytes+pos.nbytes)/1024**2)
        # mp = cp.get_default_memory_pool()
    
    # Core iteration
    if ckpt is not None:
        if last_iter >= iter:
            print(f"WARNING: New iteration target {iter} is not larger than that of previous checkpoint {last_iter}. Last result will be return as a proper result.")
        else: print(f"Doing {last_iter+1} to {iter} iteration.")
        
    start = time()
    
    # Single thread condition
    if max_workers is None or max_workers == 1:
        if False:#cuda:
            for iter_cnt in tqdm(range(last_iter, iter), ncols=100, ascii=True):
                Z, C, I = _iterate_cuda_whole(Z, C, I, iter_cnt)
            
        else:
            _iter_func = _iterate_cuda if cuda else _iterate  #_iterate_cuda
            for iter_cnt in tqdm(range(last_iter, iter), ncols=100, ascii=True):
                
                # Update values according to the rules of Mandelbrot set
                Z, C, pos = _iter_func(Z, C, I, pos, iter_cnt)
                #print(mp.used_bytes()/1024**2)
            
        
            
    # Multithread condition
    else:
        if cuda: raise NotImplementedError("Ini kblaq msqun mtzyuwaw balay qu CPU multithreading ru cuda.")
        
        # Progress report
        progress = [0]*max_workers
        results = []
        # Split
        split = list(range(0, size[0], size[0]//max_workers))
        if len(split) < max_workers+1: split += [size[0]]
        else: split[-1] = size[0]
        # Start
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
                    sleep(0.4)
                    pass
            results = [r.result() for r in results]
        [Z, C, pos] = [np.concatenate([r[i] for r in results], axis=0) for i in range(3)]
        
    print("Done in {:.2f}s".format(time()-start))   
    
    # Move cuda object back to cpu
    if cuda: C, I, pos, Z = [cp.asnumpy(mat) for mat in [C, I, pos, Z]]
        
    # Save checkpoint
    if ckpt is not None:
        with open(ckpt, 'wb') as f:
            for mat in [C, I, pos, Z, max(iter, last_iter)]: np.save(f, mat)
            
    # Turn iteration of convergence into opencv image (height, width, 3)
    I = np.log10(I+1)
    I = np.clip(I.astype(np.float)/np.log10(max_iter+1), 0, 1)
    
    # Color map: Icy (White -> Cyan(w-r) -> Blue(cy-g) -> Black)
    # split = [255,255,255]
    # I *= sum(split)
    # sec = split2sec(split, I)
    
    # blue = sec[0].up + sec[1].high + sec[2].high
    # green = sec[1].up + sec[2].high
    # red = sec[2].up
    
    # Color map: Ice and Flame (Black -> Blue -> Cyan -> White -> Red -> Yellow -> White -> Cyan)
    split = [255,255,255,255,255,255,765]   #,255,255
    I *= sum(split)
    sec = split2sec(split, I)
    
    blue  = sec[0].up   + sec[1].high + sec[2].high + sec[3].down               + sec[5].up +\
            sec[6].high #+ sec[7].high + sec[8].high
    green =               sec[1].up   + sec[2].high + sec[3].down + sec[4].up   + sec[5].high +\
            sec[6].high #+ sec[7].up   + sec[8].high
    red   =                             sec[2].up   + sec[3].high + sec[4].high + sec[5].high +\
            sec[6].down #              + sec[8].up
    
    return np.stack([ blue.astype(np.uint8),
                      green.astype(np.uint8),
                      red.astype(np.uint8) ], axis=-1)


def _looping(Z, C, I, pos, last_iter, iter, progress, i):
    for iter_cnt in range(last_iter, iter):
        Z, C, pos = _iterate(Z, C, I, pos, iter_cnt)
        progress[i] = iter_cnt+1
        if not Z.size:
            progress[i] = iter
            break
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
    
def _iterate_cuda_whole(Z, C, I, iter_cnt):
    # One iteration
    Z = cp.square(Z)+C
    
    # Criterion of divergence
    diverged = cp.abs(Z)>2
    Z[diverged] = 0
    C[diverged] = 0
    I[diverged] = iter_cnt+1
    
    return Z, C, I
    
def split2sec(split, I):
    sec = []
    sec_s = 0
    for interval in split:
        sec_e = sec_s + interval
        mask = np.where((sec_s<I if sec_s else sec_s<=I) & (I<=sec_e), 1, 0)
        sec.append(Section._make([
            mask*(I-sec_s)*255/interval, mask*(sec_e-I)*255/interval, mask*255]))
        sec_s = sec_e
    return sec
    
if __name__=="__main__":
    import cv2, os
    name = 'test'
    ## Redo
    #ctr = -0.14-0.65j
    ctr = -0.13975-0.64965j
    redo_dict = {
        "0a":[ -0.310-0.017j, 1],
        # "0b":[ ctr+0.2-0.8j,   5e-1],
        # "0c":[ ctr+0.2-0.8j, 2e-1],
        # "0d":[ ctr+0.3-0.65j, 1e-1],
        # "0e":[ ctr+0.3-0.65j, 5e-2],
        # "0f":[ ctr+0.34-0.65j, 2e-2],
        # "0g":[ ctr+0.36-0.65j, 1e-2],
        # "0h":[ ctr+0.36-0.65j, 5e-3],
        # "0i":[ ctr+0.36-0.65j, 2e-3],
        # "0j":[ ctr+0.36-0.65j, 1e-3],
        # "0k":[ ctr+1e-3*(0+0.5j), 5e-4],
        # "0l":[ ctr+1e-3*(0.2+0.4j), 2e-4],
        # "0m":[ ctr+1e-3*(0.2+0.4j), 1e-4],
        # "0n":[ ctr+1e-3*(0.25+0.35j), 5e-5],
        # "0o":[ ctr+1e-3*(0.25+0.35j), 2e-5],
        # "0p":[ ctr+1e-3*(0.25+0.35j), 1e-5],
        # "0q":[ ctr+1e-3*(0.25+0.35j), 5e-6],
        # "0r":[ ctr+1e-3*(0.25+0.35j), 2e-6],
        # "0s":[ ctr+1e-3*(0.25+0.35j), 1e-6],
        # "0t":[ctr, 5e-7],
        # "0u":[ctr, 2e-7],
        # "0v":[ctr, 1e-7],
        # "0w":[ctr, 5e-8],
        # "0x":[ctr, 2e-8],
        # "0y":[ctr, 1e-8],
        # "0z":[ctr+1e-9*(-4+0j), 5e-9],
        # "1a":[ctr+1e-9*(-4+0j), 2e-9],
        # "1b":[ctr+1e-9*(-4+0j), 1e-9],
        # "1c":[ctr+1e-9*(-4+0j), 5e-10],
        # "1d":[ctr+1e-9*(-4+0j), 2e-10],
        # "1e":[ctr+1e-9*(-4+0j), 1e-10],
    }
    
    ## Start
    # c, r = -0.5+0j, 1
    # iter = int(1e4)
    # img = mandelbrot(c, r, [1920, 1080], iter, cuda=True)   #, max_workers=6
    # cv2.imwrite("base.png".format(str(c)[1:-1], str(r)), img)
    # Find new site
    # c, r = -0.13975-0.64965j, 5e-06
    # img = mandelbrot(c, r, [1920, 1080], iter=1e5, max_workers=6, max_iter=1e7)
    # cv2.imwrite("wallpaper.png".format(name, str(c)[1:-1], r), img)
    for name, (c, r) in redo_dict.items():
        img = mandelbrot(c, r, [1024, 1024], iter=1e4, max_workers=6, max_iter=1e7)   # , ckpt=name+".ckpt", cuda=True
        cv2.imwrite("{}{}_{}.png".format(name, str(c)[1:-1], r), img)
        
        
    
    
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



def mandelbrot(center, res, size, iter=int(1e4), num_workers=1):
    # Basic iteration: 10000
    if iter is None: iter=int(1e4)
    # Support square mode (e.g., size=1024) or rectangle mode (e.g., size=[1024, 768])
    if type(size) is not list: size = [size, size]
    size = [s+1 if s%2 else s for s in size]
    half_size = [(s-1)//2 for s in size]
    
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
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        for i in range(len(split_points)-1):
            results.append(exe.submit(_mandelbrot,
                C[size[1]*split_points[i]:size[1]*split_points[i+1]],
                pos[size[1]*split_points[i]:size[1]*split_points[i+1], :],
                I[:, split_points[i]:split_points[i+1]], 
                split_points[i], iter, is_print=(not i)))
        for i, r in enumerate(results):
            r.result()
    
    # Turn iteration of convergence into opencv image (height, width, 3)
    I = np.log10(I+1)
    print(I.max())
    I = I.astype(np.float)/I.max()
    # Color map: Icy (White -> Cyan(w-r) -> Blue(cy-g) -> Black)
    return np.stack([
        np.clip(I*765, 0, 255).astype(np.uint8),
        (np.clip(I*765, 255, 510)-255).astype(np.uint8),
        (np.clip(I*765, 510, 765)-510).astype(np.uint8)
         ], axis=-1)
        
def _mandelbrot(C, pos, I, split_from, iter, is_print):
    Z = np.zeros_like(C)
    pos[:,1]-=split_from
    for iter_cnt in tqdm(range(iter)) if is_print else range(iter):
        Z = np.square(Z)+C
        diverged = np.abs(Z)>2
        if not any(diverged): continue
        for pos_to_draw in pos[diverged, :]: I[pos_to_draw[0], pos_to_draw[1]] = iter_cnt+1
        Z = np.delete(Z, diverged, axis=0)
        C = np.delete(C, diverged, axis=0)
        pos = np.delete(pos, diverged, axis=0)
    return I
    
if __name__=="__main__":
    import cv2
    c = -0.481+0.615j
    r = 2e-3
    ## Now
    #c = -0.48109+0.614645j
    #r = 5e-6
    ## Start
    #c = 0.5+0j
    #r = 1
    iter = int(1e4)
    img = mandelbrot(c, r/500, [1920, 1080], iter, num_workers=6)
    cv2.imwrite("{}_{}_ice.png".format(str(c)[1:-1], str(r)), img)
    
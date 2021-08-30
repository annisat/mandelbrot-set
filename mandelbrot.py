import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



def mandelbrot(center, res, size, iter=int(1e4), num_workers=1):
    if iter is None: iter=int(1e4)
    if size%2: size+=1
    half_size = (size-1)//2
    
    C = np.zeros(size*size, dtype=np.csingle)
    I = np.zeros((size, size), dtype=np.uint)
    pos = np.zeros((size*size, 2), dtype=np.uint)
    cnt = 0
    for r in range(size):
        for i in range(size):
            C[cnt] = center+(r-half_size)*res+(half_size-i)*res*1j
            pos[cnt, :] = [i,r]
            cnt+=1
    
    split_size = size//num_workers
    split_points = [_*split_size for _ in range(num_workers)]+[size]
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        for i in range(len(split_points)-1):
            results.append(exe.submit(_mandelbrot,
                C[size*split_points[i]:size*split_points[i+1]],
                pos[size*split_points[i]:size*split_points[i+1], :],
                I[:, split_points[i]:split_points[i+1]], 
                split_points[i], iter, is_print=(not i)))
        for i, r in enumerate(results):
            r.result()
            
    I = np.log10(I+1)
    print(I.max())
    I = I.astype(np.float)/I.max()
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
    c = -0.48109+0.614645j
    r = 5e-6
    #c = 0.5+0j
    #r = 1
    iter = int(1e7)
    img = mandelbrot(c, r/500, 1024, iter, num_workers=6)
    cv2.imwrite("{}_{}_ice.png".format(str(c)[1:-1], str(r)), img)
    
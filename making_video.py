import cv2, os
from pathlib import Path
from tqdm import tqdm



### Fit resolution function
def fit_resolution(img, width, height):
    if img.shape[1]!=width or img.shape[0]!=height:
        if width/height > img.shape[1]/img.shape[0]:
            # Magnify ratio
            dH = round((img.shape[0] - img.shape[1]*height/width)//2)
            img = img[dH:-dH, :, :]
        elif width/height < img.shape[1]/img.shape[0]:
            # Shrink ratio
            dW = round((img.shape[1] - img.shape[0]*width/height)//2)
            img = img[:, dW:-dW, :]
        # Resize
        img = cv2.resize(img, (width, height))
    return img



### Basic setting
width, height = 1280, 720
fps = 40
pref_path = Path("D://data/md_ckpt")
folder_list = ['cDp1','cAp1','cAp2','cBp2','cBp3','cCp3','cCp4','cDp4']

def make_section(writer, fd, into=True, end_frame=0):
    fn_list = [fn for fn in fd.iterdir() if fn.suffix in ['.jpg', '.png']]
    if not into: fn_list = fn_list[::-1]
    for fn in tqdm(fn_list, ncols=100):
        img = cv2.imread(str(fn))
        img = fit_resolution(img, width, height)
        writer.write(img)
    for i in range(end_frame): writer.write(img)


### Making pretty video
writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for fd_i, fd_o in zip(folder_list[::2], folder_list[1::2]):
    make_section(writer, pref_path/fd_i, end_frame=int(fps*2))
    make_section(writer, pref_path/fd_o, into=False)
    
writer.release()



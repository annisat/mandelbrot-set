import cv2
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



### Source images
source_path = Path('animation')
width, height = 1280, 720
path_list = list(source_path.iterdir())#[::3]



### Making pretty video
writer = cv2.VideoWriter('mandelreise.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
for image in tqdm(path_list):
    if not image.suffix == '.jpg': continue
    img = cv2.imread(str(image))
    img = fit_resolution(img, width, height)
    writer.write(img)
for image in tqdm(path_list[::-1]):
    if not image.suffix == '.jpg': continue
    img = cv2.imread(str(image))
    img = fit_resolution(img, width, height)
    writer.write(img)
    
writer.release()



from mandelbrot import mandelbrot



if __name__=="__main__":
    import cv2, os
    from math import log
    
    # Const during this mission
    x0, y0, r0, xn, yn, rn = -0.5, 0, 1, -0.48108935, 0.6146492, 1e-8
    rx, ry = x0-xn, y0-yn
    STEP = 1200
    output_size = [1280, 720]
    frame_size = [int(l*5) for l in output_size]
    
    
    # Base image
    asp_r = output_size[0]/output_size[1]
    base_x, base_y, base_r = x0, y0, r0
    base_left, base_top = base_x - base_r*asp_r, base_y + base_r
    base_img = None
    #mandelbrot(base_x+base_y*1j, base_r, [1920, 1080], 
    #           iter=int(log(1/base_r,10)+1)*10000, num_workers=6, max_iter=50000)
    # f = open('stat.csv', 'w')
    
    for i in range(0, STEP+1):
        print(f"Doing image number {i}")
    
        # Radius and center
        r = r0*10**(log(rn/r0, 10)*i/STEP)
        x = xn+rx*r*(1-i/STEP)
        y = yn+ry*r*(1-i/STEP)
        
        # Check if this is resizable
        left, right, top, btm = x-r*asp_r, x+r*asp_r, y+r, y-r  # in coordinate
        top, btm = [int((base_top-p)/base_r*(min(frame_size)/2)) for p in [top, btm]]
        left, right = [int((p-base_left)/base_r*(min(frame_size)/2)) for p in [left, right]]
        resizable, action = True, "resize"
        
        # No previous base_img
        if base_img is None:
            print("No base image to extrapolate from.")
            resizable = False
        # Any boundary exceed the original boundary
        if resizable and top<0 or left<0 or btm>frame_size[1]-1 or right>frame_size[0]-1:
            print("Some boundaries exceed those of the base image.")
            action = "produce for excession"
            resizable = False
        # Not enough resolution
        if resizable and ( right-left < output_size[0]+1 or btm-top < output_size[1]+1 ):
            print("Not enough resolution.")
            action = "produce for low resolution"
            resizable = False
        
        # Resize if resizable
        if resizable:
            print("This image can be extrapolated from the base image.")
            img = base_img[top:btm, left:right, :]
        else:
            base_img = mandelbrot(x+y*1j, r, frame_size, max_workers=6,
                                  iter=1e4*(r**(-0.25)), max_iter=1e6)
            base_x, base_y, base_r = x, y, r
            base_left, base_top = base_x - base_r*asp_r, base_y + base_r
            img = base_img
        img = cv2.resize(img, output_size)
        cv2.imwrite('animation/{:0>5d}.jpg'.format(i), img)
        # f.write(','.join([str(x),str(y),str(r),str(top),str(btm),str(left),str(right),action])+'\n')
    # f.close()
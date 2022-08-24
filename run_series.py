from mandelbrot import mandelbrot

def one_series(u_path, p_name, cuda, thread, FROM, ITER, base_param):
    x0, y0, xn, yn, r0, rn = base_param + [1, 1e-10]
    rx, ry = x0-xn, y0-yn
    STEP = 1200
    output_size = [1280, 720]
    
    if p_name not in os.listdir(u_path): os.mkdir(u_path+"/"+p_name)
    
    # Base image
    lbda = 0
    asp_r = output_size[0]/output_size[1]
    base_x, base_y, base_r = x0, y0, r0*(2-lbda**.5)
    base_left, base_top = base_x - base_r*asp_r, base_y + base_r
    base_img = None
    base_frame_size = [int(l*(2+lbda**.5)) for l in output_size]   #(2+lbda**.5)
    
    for i in range(FROM, STEP+1):  #STEP+1
        print(f"Doing image number {i} of {p_name}"+(" on GPU" if cuda else ""))
        
        # Radius and center
        lbda = i/STEP
        r = r0*10**(log(rn/r0, 10)*lbda**2)
        x = xn+rx*r*(1-lbda)
        y = yn+ry*r*(1-lbda)
        
        
        
        # Check if this is resizable
        resizable, action = True, "resize"
        
        # No previous base_img
        if base_img is None:
            print("No base image to extrapolate from.")
            resizable = False
            
        else:
            left, right, top, btm = x-r*asp_r, x+r*asp_r, y+r, y-r  # in coordinate
            top, btm = [int((base_top-p)/base_r*(min(frame_size)/2)) for p in [top, btm]]
            left, right = [int((p-base_left)/base_r*(min(frame_size)/2)) for p in [left, right]]
            # print(left, right, top , btm, base_frame_size)
            # Any boundary exceed the original boundary
            if top<0 or left<0 or btm>frame_size[1]-1 or right>frame_size[0]-1:
                print("Some boundaries exceed those of the base image.")
                resizable = False
                
            # Not enough resolution
            if right-left < output_size[0] or btm-top < output_size[1]:
                print("Not enough resolution.")
                resizable = False
        
        # Resize if resizable
        if resizable:
            print("This image can be extrapolated from the base image.")
            img = base_img[top:btm, left:right, :]
        else:
            # Check if this config is workable
            left, right, top, btm = x-r*asp_r, x+r*asp_r, y+r, y-r  # in coordinate
            base_x, base_y, base_r = x, y, r*(2-lbda**.5)
            base_left, base_top = base_x - base_r*asp_r, base_y + base_r
            frame_size = [int(l*(2+lbda**.5)) for l in output_size]    #
            top, btm = [int((base_top-p)/base_r*(min(frame_size)/2)) for p in [top, btm]]
            left, right = [int((p-base_left)/base_r*(min(frame_size)/2)) for p in [left, right]]
            # print(left, right, top, btm)
            # print(frame_size, output_size)
            if top<0 or left<0 or btm>frame_size[1] or right>frame_size[0]:
                print("This image is still out of bound in this setting.")
                break
            if right-left < output_size[0] or btm-top < output_size[1]:
                print("This image still lacks resolution in this setting.")
                break
            
            # Do the actual work
            ckpt_path = os.path.join(u_path, p_name, "series_"+str(i))
            base_img = mandelbrot(base_x+base_y*1j, base_r, frame_size,
                                  max_workers=thread, cuda=cuda,
                                  iter=ITER, max_iter=1e6,
                                  ckpt=ckpt_path) #, , cuda=True,
            img = base_img[top:btm, left:right, :]
            
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(u_path, p_name, '{:0>5d}.png'.format(i)), img)

    print(f"Done {p_name}")


if __name__=="__main__":
    import cv2, os
    from math import log
    
    # Const during this mission
    # center4-1: -0.31, -0.017
    # point1: -0.4810893486, 0.6146491937
    # center1-2: -0.883, 0.094
    # point2: -1.2840059947, -0.4275550867
    # center2-3: -0.452, -0.119
    # point3: 0.380656, 0.18888
    # center3-4: 0.12, -0.23
    # point4: -0.139750004, -0.64965
    
    u_path = "D://data/md_ckpt"
    '''
    	2k	5k	10k	20k	50k	    .1M	    .2M	    .5M		1M
    p1	0	310	410	540	730	    800	    830	    1020    1060    done
    p2	0	700	960 980 1010    1040    1040    1150    
    p3	0	400 470 530 570     done
    p4	0	320 390 420 460     590     660     690
    '''
    param_set = {
        'cDp1': [-0.31, -0.017,  -0.4810893486, 0.6146491937],  #| 826, 2e5
        'cAp1': [-0.883, 0.094,  -0.4810893486, 0.6146491937],  #|  
        'cAp2': [-0.883, 0.094,  -1.2840059947, -0.4275550867], #|  
        'cBp2': [-0.452, -0.119, -1.2840059947, -0.4275550867], #|  
        'cBp3': [-0.452, -0.119,  0.380656, 0.18888],           #|  
        'cCp3': [ 0.12, -0.23,    0.380656, 0.18888],           #|  
        'cCp4': [ 0.12, -0.23,   -0.139750004, -0.64965],       #|  
        'cDp4': [-0.31, -0.017,  -0.139750004, -0.64965]        #|  
        
    }
    prog_set = {
        ###'cDp1': [1060, 1e6], 'cAp1': [1060, 1e6],
        ###'cBp3': [570, 5e4], 'cCp3': [570, 5e4],
        ###'cAp2': [1150, 5e5], 'cBp2': [1150, 5e5],
        ###'cCp4': [690, 5e5], 'cDp4': [690, 5e5]
        #DONE# 
        
    }
    p_name = "cDp1"
    # one_series(u_path, p_name, True, 1, 792, 1e5, param_set[p_name])
    # Repaint
    #print(prog_set)
    for k, v in param_set.items():
        if k not in prog_set: continue
        # Find the last checkpoint
        target_num = [int(fn[7:]) for fn in os.listdir(u_path+"/"+k) if fn[:7]=='series_']
        target_num = max([n for n in target_num if n < prog_set[k][0]])
        one_series(u_path, k, True, 1, target_num, prog_set[k][1], v)
    
    
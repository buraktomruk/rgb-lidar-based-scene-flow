import numpy as np
import cv2 as cv
from PIL import Image


  
# # Extracts Optical Flow From Sequential Images
    
frame1 = cv.imread(r'./nuscenes_first.jpg')
frame2 = cv.imread(r'./nuscenes_second.jpg')


def resize_images(frame1, frame2):
    if frame1.shape[1] <  frame2.shape[1]:   
        dim = (frame1.shape[1], frame1.shape[0])
    else:
        dim = (frame2.shape[1], frame2.shape[0])
    
    frame1 = cv.resize(frame1, dim, interpolation = cv.INTER_AREA) 
    frame2 = cv.resize(frame2, dim, interpolation = cv.INTER_AREA) 
        
    return (frame1, frame2)

def extract_dense_optical_flow(frame1, frame2, save_file_idx=1):

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imwrite('optical_sequence'+str(save_file_idx)+'.jpg', rgb)

    cv.destroyAllWindows()
    
    return rgb

(frame1, frame2) = resize_images(frame1, frame2)


flow1 = extract_dense_optical_flow(frame1, frame2, 1)
flow2 = extract_dense_optical_flow(frame2, frame1, 2)

PIL_image1 = Image.fromarray(np.uint8(flow1)).convert('RGB')
PIL_image1.show()


PIL_image2 = Image.fromarray(np.uint8(flow2)).convert('RGB')
PIL_image2.show()


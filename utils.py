import cv2 
# import matplotlib.pyplot as plt

def border(filename):
    img = cv2.imread(filename)
    center = (img.shape[0]//2, img.shape[1]//2)
    dist = 20 
    start = (center[0]-dist,center[1]-dist)
    end = (center[0]+dist, center[1]+dist)
    color = (0,0,0)
    thickness = 4
    img = cv2.rectangle(img,start,end,color,thickness)
    cv2.imwrite(filename,img)
    return 0




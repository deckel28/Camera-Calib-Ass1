import cv2
import numpy as np

points = np.genfromtxt('data/recovered_coords.csv', delimiter=',')
print(points)
image = cv2.imread('data/images/camera.jpg', 1)

font = cv2.FONT_HERSHEY_SIMPLEX 

def mark_recovered(image,x,y):
    ix = round(x)
    fx = round(x,3)
    iy = round(y)
    fy = round(y,3)
    image[(iy-2):(iy+2),(ix-2):(ix+2)] = [255,255,0]
    cv2.putText(image, ' (' + str(fx) + ',' + str(fy)+ ')', (ix,iy), font, 1, (255, 255, 0), 2)
    return image

# points= [[ 990.07978675,332.87128297],
#         [1002.96060133,595.34702437],
#         [ 908.96984317,754.82548362],
#         [1542.03249886,666.91737521],
#         [1764.06504406,453.08499945],
#         [1653.89241247,306.95338747]]

for i in range(len(points)):
    image = mark_recovered(image,points[i][0],points[i][1])


cv2.imshow('image', image) 
cv2.waitKey(0) 

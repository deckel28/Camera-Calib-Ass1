import cv2
import numpy as np

coords = np.empty((0,2))

# Function to display the coordinates of the points clicked on the image 
def click_event(event, x, y, flags, params): 
	global coords
	# checking for left mouse clicks 
	if event == cv2.EVENT_LBUTTONDOWN: 

		# displaying the coordinates on the Shell 
		print(x, ' ', y)
		coords = np.append(coords, np.array([[x,y]]), axis=0) 

		# displaying the coordinates on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2) 
		cv2.imshow('image', img) 

# reading the image 
img = cv2.imread('data/images/camera.jpg', 1) 

# displaying the image 
cv2.imshow('image', img) 

# setting mouse hadler for the image and calling the click_event() function 
cv2.setMouseCallback('image', click_event) 
cv2.waitKey(0)

# saving the image 
cv2.imwrite('coords_marked.png', img) 

np.savetxt("image_coords.csv", coords, delimiter=",", fmt='%i')



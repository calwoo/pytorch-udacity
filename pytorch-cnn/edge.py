import cv2
import numpy as np 
# image plotting
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg



img = mpimg.imread("imgs/nyc.jpg")
plt.imshow(img)
plt.show()

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# sobel filter for directional edge detection
sobel_x = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]])
sobel_y = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]])

# cv2.filter2D(<img>,<bit-depth>,<filter>)
filtered = cv2.filter2D(gray, -1, sobel_y)

plt.imshow(filtered, cmap="gray")
plt.show()

edge = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]])

edge_filtered = cv2.filter2D(gray, -1, edge)
plt.imshow(edge_filtered, cmap="gray")
plt.show()
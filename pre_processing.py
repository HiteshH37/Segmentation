import cv2
from PIL import Image
import pytesseract
import numpy as np

def noiseRemoval(image):
    kernel = np.ones((1,1),np.uint8)
    image = cv2.dilate(image,kernel,iterations=1)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image,kernel,iterations=1)
    image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    image = cv2.medianBlur(image,3)
    return (image)

def thinFont(image):
    kernel = np.ones((1,1),np.uint8)
    image = cv2.bitwise_not(image)
    image = cv2.erode(image,kernel,iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thickFont(image):
    kernel = np.ones((2,2),np.uint8)
    image = cv2.bitwise_not(image)
    image = cv2.dilate(image,kernel,iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def removeBorder(image):
    countours,hiearchy = cv2.findContours(image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cntsorted = sorted(countours, key = lambda x: cv2.contourArea(x))
    cnt = cntsorted[-1]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = image[y:y+h , x:x+w]
    return crop

file = "1.jpg" 
image = cv2.imread(file)

# cv2.imshow("This is image",image)
# cv2.waitKey(0)

## inversion
# inv_image = cv2.bitwise_not(image)
# cv2.imshow("Inverted",inv_image)
# cv2.waitKey(0)

##Binarize
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray",gray_image)
# cv2.waitKey(0)

thres , bw_image = cv2.threshold(gray_image, 188, 255 ,cv2.THRESH_BINARY)
cv2.imshow("BW image",bw_image)
cv2.waitKey(0)

## noise removal
nn_image = noiseRemoval(bw_image)
cv2.imshow("No noise image",nn_image)
cv2.waitKey(0)

#thinning

thin_image = thinFont(nn_image)
cv2.imshow("thin image",thin_image)
cv2.waitKey(0)

##thick

thick_image = thickFont(bw_image)
cv2.imshow("thick image",thick_image)
# cv2.imwrite("output.jpg",thick_image)
cv2.waitKey(0)
 
# borderremover

nb_image = removeBorder(nn_image)
cv2.imshow("No border", nb_image) 


#save which image u want
cv2.imwrite("preprocessed.png",bw_image)
cv2.waitKey(0)
import numpy as np
import cv2
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import re
import pytesseract

def printImage(image):
    cv2.imshow("my image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def skewImage1(image):
	# image = image[:,250:w-100]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = cv2.bitwise_not(image)
	# threshold the image, setting all foreground pixels to
	# 255 and all background pixels to 0
	gray = cv2.bitwise_not(gray)
	thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# kernel = np.ones((5,5), np.uint8)
	# thresh = cv2.dilate(thresh, kernel, iterations=5)
	# printImage(thresh)
	# are greater than zero, then use these coordinates to
	# compute a rotated bounding box that contains all
	# coordinates
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	# the `cv2.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)
	
	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle

	# rotate the image to deskew it
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	cv2.imwrite("skew_corrected.png",rotated)
	return rotated
	
def getInput():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image",required = True, help = "path to image") # -i để cho viết tắt trước khi truyền tham số còn không thì
    args = vars(ap.parse_args()) 
    return args["image"]

def skewImage2(image):
	img = im.open("skew_corrected.png")
	# convert to binary
	wd, ht = img.size
	pix = np.array(img.convert('1').getdata(), np.uint8)
	bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)


	def find_score(arr, angle):
		data = inter.rotate(arr, angle, reshape=False, order=0)
		hist = np.sum(data, axis=1)
		score = np.sum((hist[1:] - hist[:-1]) ** 2)
		return hist, score


	delta = 1
	limit = 5
	angles = np.arange(-limit, limit+delta, delta)
	scores = []
	for angle in angles:
		hist, score = find_score(bin_img, angle)
		scores.append(score)

	best_score = max(scores)
	best_angle = angles[scores.index(best_score)]

	# correct skew
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return rotated

def skewImage3(image):
	newdata=pytesseract.image_to_osd(image)
	angle =  re.search('(?<=Rotate: )\d+', newdata).group(0)
	angle = int(angle)
	if angle==0:
		return image,angle
	return rotationImage(image,angle),angle
import imutils
def rotationImage(img,angle):
    (h,w) = img.shape[:2]
    center = (w//2,h//2)
    scale = 1.0
    if angle==90 or angle==270:
        if w>h: 
            img= cv2.copyMakeBorder(img.copy(),(w-h)//2,(w-h)//2,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        elif w<h: 
            img= cv2.copyMakeBorder(img.copy(),0,0,(h-w)//2,(h-w)//2,cv2.BORDER_CONSTANT,value=[0,0,0])
        center = (img.shape[1]//2,img.shape[1]//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    temp =  cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if angle==90 or angle==270:
	    temp = cropImage(temp,h,w)
    return temp

def cropImage(img,h,w):
    (x1,y1,x2,y2)=  (0,0,0,0)
    if w>h: # anh nam ngang
        (x1,y1,x2,y2) = ((w-h)//2,0,(w+h)//2,w)
    elif w<h: # anh nguoc
        (x1,y1,x2,y2) = ((h-w)//2,0,(w+h)//2,h)
    img = img[y1:y2,x1:x2]
    return img

def skewImage(image):
	if image.shape[1] < 900:
		image = imutils.resize(image,width=900)
	image,_ = skewImage3(image)
	image = skewImage1(image)
	image = skewImage2(image)
	image = skewImage1(image)
	image,_ = skewImage3(image)
	return image
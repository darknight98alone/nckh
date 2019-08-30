from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os
import matplotlib
import argparse
import imutils
from imutils import contours
import pytesseract
import skew
import PdfToImages
import TxtToDocx
from os import path
class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:  
            gray_img = self.src_img
        elif len(self.src_img.shape) ==3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        print(gray_img.shape)

        scale_percent = 50 # percent of original size
        width = int(gray_img.shape[1] * scale_percent / 100)
        height = int(gray_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)

        thresh_img = cv2.adaptiveThreshold(~resized,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        cv2.imwrite('thresh_img.jpg',thresh_img)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1]/scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1)) # 形态学因子
        h_erode_img = cv2.erode(h_img,h_structure,1)
        h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)
        cv2.imwrite('h_dilate_image.jpg',h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)
        cv2.imwrite('v_dilate_img.jpg',v_dilate_img)
        mask_img = h_dilate_img+v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
        cv2.imwrite("mask.jpg",mask_img)
        cv2.imwrite("joints.jpg",joints_img)
        boxes = []
        h_dilate_img_autofill = self.autofillimg_horizon(h_dilate_img,v_dilate_img)
        mask_img_temp = h_dilate_img_autofill+v_dilate_img
        v_dilate_img_autofill = self.autofillimg_vertical(h_dilate_img,v_dilate_img)
        h_dilate_img_autofill = self.remove_single_horizon(h_dilate_img_autofill,v_dilate_img_autofill)
        v_dilate_img_autofill = self.autofillimg_vertical_2nd(h_dilate_img_autofill,v_dilate_img_autofill)
        mask_img_autofill = h_dilate_img_autofill+v_dilate_img_autofill
        cv2.imwrite("mask_autofill.jpg",mask_img_autofill)
        joints_img_autofill = cv2.bitwise_and(h_dilate_img_autofill,v_dilate_img_autofill)
        cv2.imwrite('joints_img_autofill.jpg',joints_img_autofill)
        return mask_img,joints_img,mask_img_autofill,joints_img_autofill
    def autofillimg_horizon(self,_h_dilate_img,_v_dilate_img):
        height,width = _h_dilate_img.shape
        #autofill horizon
        array = _h_dilate_img.copy()
        for i in range(0,height):
            for j in range(0,width):
                if _h_dilate_img[i,j]!=0 and _h_dilate_img[i,j-1]==0 and _h_dilate_img[i,j-10]!=0 and j>10:
                    point=-1
                    for k in range(j,0,-1):
                        for l in range(i,0,-1):
                            if _v_dilate_img[l,k]!=0:
                                point = k
                                break
                        for l in range(i,height):
                            if _v_dilate_img[l,k]!=0 and l>point:
                                point = k
                                break
                        if point!=-1: break
                    if point!=-1:
                        for l in range(j,point-2,-1):
                            array[i,l]=255
                if _h_dilate_img[i,j]==0 and _h_dilate_img[i,j-1]!=0 and _h_dilate_img[i,j+10] and j>0 and j<width-10:
                    point=-1
                    for k in range(j,width):
                        for l in range(i,0,-1):
                            if _v_dilate_img[l,k]!=0:
                                point = k
                                break
                        for l in range(i,height):
                            if _v_dilate_img[l,k]!=0 and l>point:
                                point = k
                                break
                        if point!=-1: break
                    if point!=-1:
                        for l in range(j,point+2):
                            array[i,l]=255
                        j=point
        return array
    def autofillimg_vertical(self,_h_dilate_img,_v_dilate_img):
        height,width = _h_dilate_img.shape
        #autofill horizon
        array = _v_dilate_img.copy()

        for i in range(0,width):
            for j in range(0,height):
                if _v_dilate_img[j,i]!=0 and _v_dilate_img[j-1,i]==0 and j>0:
                    point=-1
                    for k in range(j,0,-1):
                        for l in range(i,0,-1):
                            if _h_dilate_img[k,l]!=0:
                                point = k
                                break
                        for l in range(i,width):
                            if _h_dilate_img[k,l]!=0 and k>point:
                                point = k
                                break
                        if point!=-1: #and _h_dilate_img[point,i-5]!=0 and _h_dilate_img[point,i+5]!=0 and i>5 and i<width-5 : 
                            break
                    if point!=-1:
                        for l in range(j,point-2,-1):
                            array[l,i]=255
                if _v_dilate_img[j,i]==0 and _v_dilate_img[j-1,i]!=0 and i>0:
                    point=-1
                    for k in range(j,height):
                        for l in range(i,0,-1):
                            if _h_dilate_img[k,l]!=0:
                                point = k
                                break
                        for l in range(i,width):
                            if _h_dilate_img[k,l]!=0 and k>point:
                                point = k
                                break
                        if point!=-1: break
                    if point!=-1:
                        for l in range(j,point+2):
                            array[l,i]=255
                        j=point
        return array
    def autofillimg_vertical_2nd(self,_h_dilate_img,_v_dilate_img):
        height,width = _h_dilate_img.shape
        #autofill horizon
        array = _v_dilate_img.copy()

        for i in range(0,width):
            for j in range(0,height):
                if _v_dilate_img[j,i]!=0 and _v_dilate_img[j-1,i]==0 and j>0:
                    point=-1
                    for k in range(j,0,-1):
                        for l in range(i,0,-1):
                            if _h_dilate_img[k,l]!=0:
                                point = k
                                break
                        for l in range(i,width):
                            if _h_dilate_img[k,l]!=0 and k>point:
                                point = k
                                break
                        if point!=-1 and _h_dilate_img[point,i-5]!=0 and _h_dilate_img[point,i+5]!=0 and i>5 and i<width-5 : 
                            break
                    if point!=-1:
                        for l in range(j,point-2,-1):
                            array[l,i]=255
                if _v_dilate_img[j,i]==0 and _v_dilate_img[j-1,i]!=0 and i>0:
                    point=-1
                    for k in range(j,height):
                        for l in range(i,0,-1):
                            if _h_dilate_img[k,l]!=0:
                                point = k
                                break
                        for l in range(i,width):
                            if _h_dilate_img[k,l]!=0 and k>point:
                                point = k
                                break
                        if point!=-1: break
                    if point!=-1:
                        for l in range(j,point+2):
                            array[l,i]=255
                        j=point
        return array
    def remove_single_horizon(self,_h_dilate_img,_v_dilate_img):
        height,width = _h_dilate_img.shape
        array = _h_dilate_img.copy()
        for i in range(0,height):
            point = True
            for j in range(0,width):
                if _h_dilate_img[i,j]!=0 and _v_dilate_img[i,j]!=0:
                    point = False
                    break
            if point:
                for j in range(0,width):
                    array[i,j]=0
        return array

# get image coordinate
def get_boxes_coordinate(image):
    image = cv2.resize(image,(361,500))
def printImage(image):
    cv2.imshow("my image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getInput():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image",required = True, help = "path to image") # -i để cho viết tắt trước khi truyền tham số còn không thì
    ap.add_argument("-m","--mask",required = True, help = "path to image")
    args = vars(ap.parse_args()) 
    return args["image"],args["mask"]

def getTableCoordinate(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    (h1,w1) = image.shape
    blured = cv2.GaussianBlur(image,(11,11),0)
    canImage = cv2.Canny(blured,100,250)
    newimage = np.zeros_like(image)
    if imutils.is_cv2() or imutils.is_cv4():
        (conts,_)= cv2.findContours(canImage.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    elif imutils.is_cv3():
        (_,conts,_)= cv2.findContours(canImage.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    conts = contours.sort_contours(conts)[0]
    listBigBoxPoint = []
    listBigBox = []
    listPoint = []
    listResult = []
    for i in range (len(conts)):
        (x, y, w, h) = cv2.boundingRect(conts[i])
        if w>10 and h>10 and w < 0.7*w1:
            if (x,y) not in listPoint:
                for j in range(3):
                    listPoint.append((x+j,y+j))
                listResult.append((x,y,w,h))
                cv2.rectangle(newimage,(x,y),(x+w,y+h),255,1)
        if w>10 and h>10 and w>0.7*w1:
            if (x,y) not in listBigBoxPoint:
                listBigBox.append((x,y,w,h))
                listBigBoxPoint.append((x,y))
    ## phuong phap xu li tam thoi
    return listResult,listBigBox

def appendListBigBox(listBigBox,img,listResult):
    result = []
    if len(listBigBox)>0:
        if len(listBigBox)==1:
            listBigBox = []
        else:
            listBigBox = listBigBox[1:]
    for pt in listResult:
        (x,y,w,h) = pt
        if len(listBigBox)>0:
            if y>listBigBox[0][1]:
                break
        tempImage = img[y:(y+h-1),x:(x+w-1)]
        tempImage = imutils.resize(tempImage,height=90)
        cv2.imwrite("temp.jpg",tempImage)
        result.append(pytesseract.image_to_string(Image.open('temp.jpg'), lang='vie'))
    
    return result,listBigBox

def process_par(image,output,listBigBox,listResult):
    if len(listBigBox)>0:
        listBigBox.sort(key=lambda x: x[1])
        print(listBigBox[0][1])
    results = []	
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # assign a rectangle kernel size
    kernel = np.ones((5,5), 'uint8')	
    par_img = cv2.dilate(thresh,kernel,iterations=5)
    if imutils.is_cv2() or imutils.is_cv4():
        (contours, hierarchy) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    elif imutils.is_cv3():
        (_,contours, hierarchy) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )
    for i,cnt in enumerate(sorted_contours):
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)
        crop = output[y:y+h, x:x+w]
        if len(listBigBox)>0:
            if y > listBigBox[0][1]:
                string,listBigBox = appendListBigBox(listBigBox,output,listResult)
                for st in string:
                    results.append(st)
        cv2.imwrite("temp.jpg",crop)
        results.append(pytesseract.image_to_string(Image.open('temp.jpg'), lang='vie'))
    return output,results


def writeToTxt(result,filePath):
    
    f = open(filePath + "/result.txt","a")
    for rs in result:
        f.write(rs+"\n")        
    f.close()
    TxtToDocx.txtToDocx(filePath+"/result.txt")

def readImageFileInFolder(inputPath):
    listName = []
    for r, d, f in os.walk(inputPath):
        for file in f:
            # if '.png' in file:
            print(r)
            print(file)
            if '.pdf' in file:
                PdfToImages.pdfToImage(r+"/"+file,r+"/")
            if str(file).lower().endswith(('.png', '.jpg', '.jpeg')):
                listName.append(os.path.join(r, file))
    return listName
    


if __name__=='__main__':
    inputPath,outputPath = getInput()
    if path.exists(inputPath) and path.exists(outputPath):
        if path.exists(outputPath+"/result.txt"):
            os.remove(outputPath + "/result.txt")
        for imageName in readImageFileInFolder(inputPath):
            # imageName = "d.jpg"
            img = cv2.imread(imageName)
            # img = skew.skewImage(img)
            
            mask,joint,mask_img,joint_img = detectTable(img).run()
            maskName = "mask.jpg"
            mask_img = cv2.imread(maskName)
            (h,w,d) = mask_img.shape
            mask_img = imutils.resize(mask_img,width=w*2,height=h*2)
            printImage(mask_img)
            listResult,listBigBox = getTableCoordinate(mask_img)
            img= cv2.resize(img,(mask_img.shape[1],mask_img.shape[0]))
            origin = img.copy()
            for pt in listBigBox:
                (x,y,w,h) = pt
                img[y:(y+h-1),x:(x+w-1)] = 255
            out,result = process_par(img,origin,listBigBox,listResult)
            printImage(out)
            for rs in result:
                print(rs + "\n")
            writeToTxt(result,outputPath)
    else:
        print("sai duong dan")
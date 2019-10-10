import functools
import os
import cv2
import imutils
import numpy as np

class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self,choose):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
        elif len(self.src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        # print(gray_img.shape)

        scale_percent = 50  # percent of original size
        width = int(gray_img.shape[1] * scale_percent / 100)
        height = int(gray_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)

        thresh_img = cv2.adaptiveThreshold(~resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # cv2.imwrite('thresh_img.jpg',thresh_img)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1] / scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))  # 形态学因子
        h_erode_img = cv2.erode(h_img, h_structure, 1)
        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)
        v_size = int(v_img.shape[0] / scale)
        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)
        ## mask chua auto fill
        mask_img = h_dilate_img + v_dilate_img
        if choose == 1:
            cv2.imwrite("mask.jpg", mask_img)
            return mask_img
        boxes = []
        h_dilate_img_autofill = self.autofillimg_horizon(h_dilate_img, v_dilate_img)
        v_dilate_img_autofill = self.autofillimg_vertical(h_dilate_img, v_dilate_img)
        h_dilate_img_autofill = self.remove_single_horizon(h_dilate_img_autofill, v_dilate_img_autofill)
        v_dilate_img_autofill = self.autofillimg_vertical_2nd(h_dilate_img_autofill, v_dilate_img_autofill)
        ## mask auto fill
        mask_img_autofill = h_dilate_img_autofill + v_dilate_img_autofill
        if choose == 2:
            cv2.imwrite("mask.jpg", mask_img_autofill)
            return mask_img_autofill

    def autofillimg_horizon(self, _h_dilate_img, _v_dilate_img):
        height, width = _h_dilate_img.shape
        # autofill horizon
        array = _h_dilate_img.copy()
        for i in range(0, height):
            for j in range(0, width):
                if _h_dilate_img[i, j] != 0 and _h_dilate_img[i, j - 1] == 0 and _h_dilate_img[
                    i, j - 10] != 0 and j > 10:
                    point = -1
                    for k in range(j, 0, -1):
                        for l in range(i, 0, -1):
                            if _v_dilate_img[l, k] != 0:
                                point = k
                                break
                        for l in range(i, height):
                            if _v_dilate_img[l, k] != 0 and l > point:
                                point = k
                                break
                        if point != -1: break
                    if point != -1:
                        for l in range(j, point - 2, -1):
                            array[i, l] = 255
                if _h_dilate_img[i, j] == 0 and _h_dilate_img[i, j - 1] != 0 and _h_dilate_img[
                    i, j + 10] and j > 0 and j < width - 10:
                    point = -1
                    for k in range(j, width):
                        for l in range(i, 0, -1):
                            if _v_dilate_img[l, k] != 0:
                                point = k
                                break
                        for l in range(i, height):
                            if _v_dilate_img[l, k] != 0 and l > point:
                                point = k
                                break
                        if point != -1: break
                    if point != -1:
                        for l in range(j, point + 2):
                            array[i, l] = 255
                        j = point
        return array

    def autofillimg_vertical(self, _h_dilate_img, _v_dilate_img):
        height, width = _h_dilate_img.shape
        # autofill horizon
        array = _v_dilate_img.copy()

        for i in range(0, width):
            for j in range(0, height):
                if _v_dilate_img[j, i] != 0 and _v_dilate_img[j - 1, i] == 0 and j > 0:
                    point = -1
                    for k in range(j, 0, -1):
                        for l in range(i, 0, -1):
                            if _h_dilate_img[k, l] != 0:
                                point = k
                                break
                        for l in range(i, width):
                            if _h_dilate_img[k, l] != 0 and k > point:
                                point = k
                                break
                        if point != -1:  # and _h_dilate_img[point,i-5]!=0 and _h_dilate_img[point,i+5]!=0 and i>5 and i<width-5 :
                            break
                    if point != -1:
                        for l in range(j, point - 2, -1):
                            array[l, i] = 255
                if _v_dilate_img[j, i] == 0 and _v_dilate_img[j - 1, i] != 0 and i > 0:
                    point = -1
                    for k in range(j, height):
                        for l in range(i, 0, -1):
                            if _h_dilate_img[k, l] != 0:
                                point = k
                                break
                        for l in range(i, width):
                            if _h_dilate_img[k, l] != 0 and k > point:
                                point = k
                                break
                        if point != -1: break
                    if point != -1:
                        for l in range(j, point + 2):
                            array[l, i] = 255
                        j = point
        return array

    def autofillimg_vertical_2nd(self, _h_dilate_img, _v_dilate_img):
        height, width = _h_dilate_img.shape
        # autofill horizon
        array = _v_dilate_img.copy()

        for i in range(0, width):
            for j in range(0, height):
                if _v_dilate_img[j, i] != 0 and _v_dilate_img[j - 1, i] == 0 and j > 0:
                    point = -1
                    for k in range(j, 0, -1):
                        for l in range(i, 0, -1):
                            if _h_dilate_img[k, l] != 0:
                                point = k
                                break
                        for l in range(i, width):
                            if _h_dilate_img[k, l] != 0 and k > point:
                                point = k
                                break
                        if point != -1 and _h_dilate_img[point, i - 5] != 0 and _h_dilate_img[
                            point, i + 5] != 0 and i > 5 and i < width - 5:
                            break
                    if point != -1:
                        for l in range(j, point - 2, -1):
                            array[l, i] = 255
                if _v_dilate_img[j, i] == 0 and _v_dilate_img[j - 1, i] != 0 and i > 0:
                    point = -1
                    for k in range(j, height):
                        for l in range(i, 0, -1):
                            if _h_dilate_img[k, l] != 0:
                                point = k
                                break
                        for l in range(i, width):
                            if _h_dilate_img[k, l] != 0 and k > point:
                                point = k
                                break
                        if point != -1: break
                    if point != -1:
                        for l in range(j, point + 2):
                            array[l, i] = 255
                        j = point
        return array

    def remove_single_horizon(self, _h_dilate_img, _v_dilate_img):
        height, width = _h_dilate_img.shape
        array = _h_dilate_img.copy()
        for i in range(0, height):
            point = True
            for j in range(0, width):
                if _h_dilate_img[i, j] != 0 and _v_dilate_img[i, j] != 0:
                    point = False
                    break
            if point:
                for j in range(0, width):
                    array[i, j] = 0
        return array
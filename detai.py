import argparse
import functools
import os

import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
from PyPDF2 import PdfFileReader
from docx import Document
from docx.shared import Inches
from docx.shared import Cm
from docx.shared import Mm
from docx.enum.table import WD_TABLE_ALIGNMENT
from imutils import contours
from pdf2image import convert_from_path

import skew


class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
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
        # cv2.imwrite('h_dilate_image.jpg',h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)
        # cv2.imwrite('v_dilate_img.jpg',v_dilate_img)
        mask_img = h_dilate_img + v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
        # plt.imsave("h_dilate.jpg",h_dilate_img)
        # plt.imsave('v_dilate.jpg',v_dilate_img)
        cv2.imwrite("mask.jpg", mask_img)
        # cv2.imwrite("joints.jpg",joints_img)
        # plt.imsave("joints.jpg",joints_img)
        boxes = []
        # h_dilate_img_autofill = self.autofillimg_horizon(h_dilate_img, v_dilate_img)
        # cv2.imwrite("h_dilate_autofill.jpg",h_dilate_img_autofill)
        # mask_img_temp = h_dilate_img_autofill + v_dilate_img
        # plt.imsave('mask_h_autofill.jpg',mask_img_temp)
        v_dilate_img_autofill = self.autofillimg_vertical(h_dilate_img, v_dilate_img)
        # cv2.imwrite('v_dilate_autofill.jpg',v_dilate_img_autofill)
        # h_dilate_img_autofill = self.remove_single_horizon(h_dilate_img_autofill, v_dilate_img_autofill)
        # v_dilate_img_autofill = self.autofillimg_vertical_2nd(h_dilate_img_autofill, v_dilate_img_autofill)
        # mask_img_autofill = h_dilate_img_autofill + v_dilate_img_autofill
        # plt.imsave("mask_autofill.jpg",mask_img_autofill)
        # cv2.imwrite("mask_autofill.jpg",mask_img_autofill)
        # plt.imsave('joints_img_autofill.jpg',joints_img_autofill)
        # joints_img_autofill = cv2.bitwise_and(h_dilate_img_autofill, v_dilate_img_autofill)
        # plt.imsave('joints_img_autofill.jpg',joints_img_autofill)
        # cv2.imwrite('joints_img_autofill.jpg',joints_img_autofill)
        # return mask_img, joints_img, mask_img_autofill, joints_img_autofill
        return mask_img, joints_img

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


# get image coordinate
def get_boxes_coordinate(image):
    image = cv2.resize(image, (361, 500))


def printImage(image):
    cv2.imshow("my image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getInput():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to image")  # -i để cho viết tắt trước khi truyền tham số còn không thì
    ap.add_argument("-o", "--outPath", required=True, help="output path")
    # ap.add_argument("-n","--outName",required = True, help = "name of docx")
    args = vars(ap.parse_args())
    return args["image"], args["outPath"]


def getTableCoordinate(image):
    """

    :param image:
    :return:
    listResult: x, y coordinates of layout 's bounding box
    listBigBox: x, y coordinates of table in image
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    (h1, w1) = image.shape
    blured = cv2.GaussianBlur(image, (11, 11), 0)
    canImage = cv2.Canny(blured, 100, 250)
    newimage = np.zeros_like(image)
    if imutils.is_cv2() or imutils.is_cv4():
        (conts, _) = cv2.findContours(canImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    elif imutils.is_cv3():
        (_, conts, _) = cv2.findContours(canImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    listBigBoxPoint = []
    listBigBox = []
    listPoint = []
    listResult = []
    if len(conts) > 0:
        conts = contours.sort_contours(conts)[0]
        # conts = sorted(conts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )
        for i in range(len(conts)):
            (x, y, w, h) = cv2.boundingRect(conts[i])
            if w > 10 and h > 10 and w < 0.7 * w1:
                if (x, y) not in listPoint:
                    for j in range(-5, 5, 1):
                        listPoint.append((x + j, y + j))
                    listResult.append((x, y, w, h))
                    cv2.rectangle(newimage, (x, y), (x + w, y + h), 255, 1)
                    # printImagenewimage)
            if w > 10 and h > 10 and w > 0.7 * w1:
                if (x, y) not in listBigBoxPoint:
                    listBigBox.append((x, y, w, h))
                    listBigBoxPoint.append((x, y))
    ## phuong phap xu li tam thoi
    return listResult, listBigBox


def appendListBigBox(listBigBox, img, listResult):
    result = []
    if len(listBigBox) > 0:
        if len(listBigBox) == 1:
            listBigBox = []
        else:
            listBigBox = listBigBox[1:]
    number_of_bbox = 1
    for pt in listResult:
        (x, y, w, h) = pt
        if len(listBigBox) > 0:
            if y > listBigBox[0][1]:
                break
        tempImage = img[y:(y + h - 1), x:(x + w - 1)]
        (h, w, d) = tempImage.shape
        tempImage = imutils.resize(tempImage, height=h * 2)
        # #printImagetempImage)
        cv2.imwrite("temp.jpg", tempImage)
        result.append((pytesseract.image_to_string(Image.open('temp.jpg'), lang='vie')
                       , x, y, w, h, number_of_bbox))
        number_of_bbox += 1
        # print(result[len(result)-1])
    return result, listBigBox


def compare_table(item1, item2):
    # return (item1[2]-item2[2])/10
    if (item1[2] - item2[2]) // 10 > 0:  # return 1 means swap
        return 1
    elif (item1[2] - item2[2]) // 10 < 0:
        return -1
    else:
        return item1[1] - item2[1]


def process_par(image, output, listBigBox, listResult):
    if len(listBigBox) > 0:
        listBigBox.sort(key=lambda x: x[1])
        # print(listBigBox[0][1])
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # assign a rectangle kernel size
    kernel = np.ones((5, 5), 'uint8')
    par_img = cv2.dilate(thresh, kernel, iterations=5)
    if imutils.is_cv2() or imutils.is_cv4():
        (contours, hierarchy) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif imutils.is_cv3():
        (_, contours, hierarchy) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        sorted_contours = sorted(contours,
                                 key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1])
        k = 1
        for i, cnt in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            crop = output[y:y + h, x:x + w]
            if len(listBigBox) > k-1:
                if y > listBigBox[0][1]:
                    # string_coordinate, listBigBox = appendListBigBox(listBigBox, output, listResult)
                    results.append(('', listBigBox[k-1][0], listBigBox[k-1][1], listBigBox[k-1][2],
                                    listBigBox[k-1][3], k))
                    k += 1
                    # string_coordinate.sort(key=lambda k: (k[2],k[1]))
                    # string_coordinate.sort(key=functools.cmp_to_key(compare_table))
                    # string_coordinate.sort(key=functools.cmp_to_key(compare_table))
                    # for st in string_coordinate:
                    #     results.append(st)
            cv2.imwrite("temp.jpg", crop)
            output_tesseract = pytesseract.image_to_string(Image.open('temp.jpg'),
                                                lang='vie')
            if output_tesseract == '':
                continue
            temp = (output_tesseract, x, y, w, h, 0)
            # print(i , " " , temp)
            # print("###########")

            results.append(temp)

            # results.append(pytesseract.image_to_string(Image.open('temp.jpg'),
            #                                            lang='vie'))
    return output, results


def writeToTxt(result, docName):
    if os.path.exists(docName) == False:
        dc = Document()
        dc.save(docName)
    # else:
    #     i =1
    #     while (True) :
    #         if os.path.exists(docName[:len(docName)-5]+str(i)+".docx")==False:
    #             docName = docName[:len(docName)-5]+str(i)+".docx"
    #             dc = Document()
    #             dc.save(docName)
    #             break
    #         i = i + 1
    document = Document(docName)
    for line in result:
        para = document.add_paragraph(line)
        paragraph_format = para.paragraph_format
        paragraph_format.space_before = 0
        paragraph_format.space_after = 0
        paragraph_format.line_spacing = 1
    document.save(docName)


# def readImageFileInFolder(inputPath,outputPath,outputName):
#     listName = []
#     for r, d, f in os.walk(inputPath):
#         for file in f:
#             # if '.png' in file:
#             #print(r)
#             #print(file)
#             if str(file).lower().endswith(".pdf"):
#                 pdf = PdfFileReader(open(inputPath+"/"+file,'rb'))
#                 maxPages = pdf.getNumPages()
#                 for page in range(1,maxPages,10) : 
#                     images_from_path = convert_from_path(inputPath +"/"+ file, dpi=200, first_page=page, last_page = min(page+10-1,maxPages))
#                     for image in images_from_path:
#                         image.save('temp.jpg',)
#                         img = cv2.imread("temp.jpg")
#                         # #printImageimg)
#                         handleFileToDocx(img,outputPath,outputName)
#             if str(file).lower().endswith(('.png', '.jpg', '.jpeg')):
#                 listName.append(os.path.join(r, file))
#     return listName

#


# param: bounding_boxs = [('',x,y,w,h,0), ('',x,y,w,h,1),...] 0 = bounding box of text, 1 = bounding box of table
def recover_docx(bounding_boxs, output_name, image):
    line_tables = []
    current_line = 0
    # current_bbox = 0
    i = 0
    bounding_boxs.sort(key=functools.cmp_to_key(compare_table))
    # bounding_boxs.sort(key=functools.cmp_to_key(compare_table))
    while i < len(bounding_boxs):
        if bounding_boxs[i][5] == 0:
            if bounding_boxs[i][0] == '':
                i += 1
                continue
            str_on_line = []
            str_on_line.append(bounding_boxs[i][0])
            j = i+1
            if j < len(bounding_boxs):
                mi = min(bounding_boxs[i][4] + bounding_boxs[i][2], bounding_boxs[j][4] + bounding_boxs[j][2])
                ma = max(bounding_boxs[i][2], bounding_boxs[j][2])
            # min - max >= 55
            while j < len(bounding_boxs) and (mi - ma >= 40):
                if bounding_boxs[j][0] == '':
                    j += 1
                    break
                str_on_line.append(bounding_boxs[j][0])
                j += 1
                if j < len(bounding_boxs):
                    mi = min(bounding_boxs[i][4] + bounding_boxs[i][2], bounding_boxs[j][4] + bounding_boxs[j][2])
                    ma = max(bounding_boxs[i][2], bounding_boxs[j][2])
            i = j
            if len(str_on_line) > 0:
                line_tables.append((str_on_line, 0))
        elif bounding_boxs[i][5] != 0:
            line_tables.append(((bounding_boxs[i][1],bounding_boxs[i][2],bounding_boxs[i][3],bounding_boxs[i][4]), 1))
            i += 1

    if os.path.exists(outputName) == False:
        dc = Document()
        dc.save(outputName)
    document = Document(outputName)
    for row in line_tables:
        if row[1] == 0:
            table = document.add_table(rows=1, cols=len(row[0]))
            i = 0
            row_cells = table.rows[0].cells
            for cell in row[0]:
                p = row_cells[i].add_paragraph(cell)
                p.alignment = WD_TABLE_ALIGNMENT.CENTER
                i += 1
        if row[1] == 1:
            cropped_table = image[row[0][1]:row[0][1]+row[0][3],
                            row[0][0]:row[0][0]+row[0][2]]
            # cv2.imshow('cropped_table', cropped_table)
            # cv2.waitKey(0)
            cv2.imwrite('cropped_table.jpg', cropped_table)
            table = document.add_table(rows=1, cols=1)
            row_cells = table.add_row().cells
            p = row_cells[0].add_paragraph()
            p.alignment = WD_TABLE_ALIGNMENT.CENTER
            r = p.add_run()
            img_width = image.shape[0]
            print(img_width)
            img_height = image.shape[1]
            print(img_height)
            print('row[0][3]', row[0][3])
            print('height', row[0][3]/img_height)
            # r.add_picture('cropped_table.jpg')
            document.add_picture('cropped_table.jpg',
                                 width=Inches(6.0),
                                 height=Inches(row[0][3]/img_height*5))

            i += 1

    document.save(outputName)

def handleFileToDocx(fileName, outputName):
    """
    :param fileName: name of image to be converted
    :param outputName: name of doc file to be saved
    :return:

    detect table and layout-analyzing
    """
    img = cv2.imread(fileName)
    img = skew.skewImage(img)
    mask, joint = detectTable(img).run()
    # mask, joint, mask_img, joint_img = detectTable(img).run()
    original_img = img.copy()
    # cv2.imwrite('debug.jpg', img)
    maskName = "mask.jpg"
    mask_img = cv2.imread(maskName)
    (h, w, d) = mask_img.shape
    mask_img = imutils.resize(mask_img, width=w * 2, height=h * 2)
    # #printImagemask_img)
    listResult, listBigBox = getTableCoordinate(mask_img)
    img = cv2.resize(img, (mask_img.shape[1], mask_img.shape[0]))
    origin = img.copy()
    for pt in listBigBox:
        (x, y, w, h) = pt
        img[y:(y + h - 1), x:(x + w - 1)] = 255
    out, result = process_par(img, origin, listBigBox, listResult)
    cv2.imwrite('debug.jpg', out)
    # stringResult = [ res[0] for res in result ]
    for i, res in enumerate(result):
        print(i, ' ', res)
        print('#############')
    #     # print(len(result))
    # stringResult = list(list(zip(*result))[0])
    # print("str result ", len(stringResult))
    # #printImageout)
    # for rs in result:
    #     #print(rs + "\n")
    # writeToTxt(stringResult, outputName)
    # error reading first page in result list
    recover_docx(result, outputName, original_img)


# def folderFileToDocx(inputPath,outputPath,outputName):
#     if path.exists(inputPath) and path.exists(outputPath):
#         if path.exists(outputPath+"/"+outputName):
#             os.remove(outputPath + "/"+outputName)
#         for imageName in readImageFileInFolder(inputPath,outputPath,outputName):
#             # imageName = "d.jpg"
#             img = cv2.imread(imageName)
#             handleFileToDocx(img,outputName)
#     else:
#         #print("sai duong dan")

def FileToDocx(inputFile, outputName):
    if inputFile.lower().endswith(".pdf"):
        pdf = PdfFileReader(open(inputFile, 'rb'))
        maxPages = pdf.getNumPages()
        for page in range(1, maxPages, 10):
            images_from_path = convert_from_path(inputFile, dpi=200, first_page=page,
                                                 last_page=min(page + 10 - 1, maxPages))
            for image in images_from_path:
                image.save('temp.jpg')
                handleFileToDocx("temp.jpg", outputName)
    elif str(inputFile).lower().endswith(('.png', '.jpg', '.jpeg')):
        handleFileToDocx(inputFile, outputName)


if __name__ == '__main__':
    # inputFile,outputName= getInput()
    inputPath = r"nckh\\14"
    for r, d, f in os.walk(inputPath):
        i = 0
        for file in f:
            inputFile = str(r) + "\\" + str(file)
            outputName = str(r) + r"\\bo_autofill.docx"
            FileToDocx(inputFile, outputName)
    print("hello end")
    # inputFile = "./image/1611703.PDF"
    # outputName = "newdoc.docx"
    # for

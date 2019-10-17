import argparse
import pytesseract
from PIL import Image
from PyPDF2 import PdfFileReader
from imutils import contours
import imutils
from pdf2image import convert_from_path
import cv2
import numpy as np

# get image coordinate
def get_boxes_coordinate(image):
    image = cv2.resize(image, (361, 500))

def printImage(image):
    cv2.imshow("my image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getInput():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputFile", required=True,
                    help="path to image")  # -i để cho viết tắt trước khi truyền tham số còn không thì
    # ap.add_argument("-n","--outName",required = True, help = "name of docx")
    args = vars(ap.parse_args())
    return args["inputFile"]


def getTableCoordinate(image):
    """

    :param image:
    :return:
    listResult: x, y coordinates of layout 's bounding box
    listBigBox: x, y coordinates of table in image
    """
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
                    # printImage(newimage)
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


def process_par(image, output, listBigBox):
    if len(listBigBox) > 0:
        listBigBox.sort(key=lambda x: x[1])
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
                    results.append(('', listBigBox[k-1][0], listBigBox[k-1][1], listBigBox[k-1][2],
                                    listBigBox[k-1][3], k))
                    k += 1
            cv2.imwrite("temp.jpg", crop)
            output_tesseract = pytesseract.image_to_string(Image.open('temp.jpg'),
                                                lang='vie')
            if output_tesseract == '':
                continue
            results.append(output_tesseract)
    return output, results

def retreiveTextFromTable(listResult,image):
    results = []
    for cnt in listResult:
        x, y, w, h = cnt
        crop = image[y:y + h, x:x + w]
        output_tesseract = pytesseract.image_to_string(crop,
                                                lang='vie')
        if output_tesseract == '':
                continue
        results.append(output_tesseract)
    return results                                        
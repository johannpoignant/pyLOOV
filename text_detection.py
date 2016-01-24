
import cv, cv2
from scipy import ndimage
import numpy as np


def sauvola(img):   
    mean = np.mean(img.ravel())
    if mean-127.5 < 0: mean = abs(mean-127.5)+127.5
    else:              mean = 255 - mean
    return 255-round(mean*(1+0.5*(np.std(img.ravel()) /127.5 - 1)), 0)

def find_connected_component(frame, mask, height, width, thresholdSobel, itConnectedCaractere, yMinSizeText, xMinSizeText):
    img = cv2.Sobel(frame, cv.CV_8U, 1, 0, ksize=1)
    img = cv2.convertScaleAbs(img)
    _, img = cv2.threshold(img, thresholdSobel, 255, cv.CV_THRESH_BINARY)

    # connect character
    kernel = np.array(((0, 0, 0), (1, 1, 1), (0, 0, 0)),np.uint8)
    img = cv2.dilate(img, kernel, iterations = itConnectedCaractere)
    img = cv2.erode(img, kernel, iterations = itConnectedCaractere)

    # delete horizontal bar
    cv2.rectangle(img, (0,0), (width, height), (0,0,0), 2, 4, 0)
    kernel = np.array(((0, 0, 0), (1, 1, 1), (0, 0, 0)),np.uint8)
    img = cv2.erode(img, kernel, iterations = int(round(float(xMinSizeText)/4.)))
    img = cv2.dilate(img, kernel, iterations = int(round(float(xMinSizeText)/4.)))

    # delete vertical bar
    cv2.rectangle(img, (0,0), (width, height), (0,0,0), 2, 4, 0)
    kernel = np.array(([1], [1], [1]),np.uint8)
    img = cv2.erode(img, kernel, iterations = int(round(yMinSizeText/4.0)))
    img = cv2.dilate(img, kernel, iterations = int(round(yMinSizeText/4.0)))

    # apply mask
    if mask: img = cv2.bitwise_and(img, mask)
    cv2.rectangle(img, (0,0), (width, height), (0,0,0), 2, 4, 0)

    # find contour
    contours, _ = cv2.findContours(img, cv.CV_RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return  contours  

def text_white_or_black(frame, x1, y1, x2, y2):
    #find if the text is write in black or in white
    blocksize = y2-y1
    if (blocksize%2 == 0): blocksize+=1
    
    # compute standard deviation if the text is write in white
    img1 = np.copy(frame[y1:y2, x1:x2])    
    img1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 2)
    cv2.rectangle(img1, (0,0), (x2-x1, y2-y1), 0, 1, 1, 0)

    label_cc1, nb_cc1 = ndimage.label(img1)   
    size_cc1 = np.bincount(label_cc1.ravel())[1:]
    
    # compute standard deviation if the text is write in black
    img2 = np.copy(frame[y1:y2, x1:x2])
    img2 = cv2.bitwise_not(img2)    
    img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 2)
    cv2.rectangle(img2, (0,0), (x2-x1, y2-y1), 255, 1, 1, 0)

    label_cc2, nb_cc2 = ndimage.label(img2)    
    size_cc2 = np.bincount(label_cc2.ravel())[1:]    

    if np.std(size_cc1) > np.std(size_cc2):  # reverse the image
        return False
    return True

def find_x_position(frame, x1, x2, y1, y2, text_white):
    img = np.copy(frame[y1:y2, x1:x2])    
    if not text_white: img = cv2.bitwise_not(img)
        
    _, img = cv2.threshold(img, sauvola(img), 255, cv.CV_THRESH_BINARY)

    kernel = np.array(([1], [1], [1]),np.uint8)
    img = cv2.dilate(img, kernel, iterations = x2-x1)
    img = img[0]
    findRow = False
    l, lTmp = [], []
    for i in range(len(img)):
        if img[i] == 0 and not findRow:
            findRow = True
        if img[i] == 0 and findRow:
            l+=lTmp
            lTmp = []
        if img[i] == 255 and findRow:
            lTmp.append(i+x1)

    if not l: return 0, 0
    return l[0], l[-1]

def find_y_position(frame, x1, x2, y1, y2, text_white):
    img = np.copy(frame[y1:y2, x1:x2])    
    if not text_white: img = cv2.bitwise_not(img)
    
    _, img = cv2.threshold(img, sauvola(img), 255, cv.CV_THRESH_BINARY)

    kernel = np.array(((0, 0, 0), (1, 1, 1), (0, 0, 0)),np.uint8)
    img = cv2.dilate(img, kernel, iterations = x2-x1)
    img = img[:,0]
    
    findLine = False
    start, end = 0, 0
    l = []
    for i in range(len(img)):
        if img[i] == 255:
            if not findLine : 
                start = i
                findLine = True
            else : end = i
        elif findLine: 
            l.append([end-start, start+y1, end+y1]) 
            findLine = False  
    if findLine: l.append([end-start, start+y1, i+y1]) 
    
    if not l: return 0, 0    
    return sorted(l, reverse = True)[0][1:]


def spatial_detection_LOOV(frame, mask, height, width, thresholdSobel, itConnectedCaractere, yMinSizeText, xMinSizeText, marginCoarseDetectionX, marginCoarseDetectionY, minRatioWidthHeight):
    newBoxes = []
    # for each connected component
    contours = find_connected_component(frame, mask, height, width, thresholdSobel, itConnectedCaractere, yMinSizeText, xMinSizeText)
    for c in contours:
        c = np.array(c)
        x1, y1 = c.min(axis=0)[0]
        x2, y2 = c.max(axis=0)[0]
        
        text_white = text_white_or_black(frame, x1, y1, x2, y2)
            
        # refine detection
        margeH = marginCoarseDetectionX * float(y2-y1)
        margeW = marginCoarseDetectionY * float(y2-y1)
        
        x1 = max(0, int(round(x1-margeH, 0)))
        x2 = min(width, int(round(x2+margeH)))
        y1 = max(0, int(round(y1-margeW)))
        y2 = min(height, int(round(y2+margeW)))
        
        y1, y2 = find_y_position(frame, x1, x2, y1, y2, text_white)
        if y2-y1 < yMinSizeText: continue
        
        x1, x2 = find_x_position(frame, x1, x2, y1, y2, text_white)
        if x2-x1 < xMinSizeText: continue

        if (x2-x1) < minRatioWidthHeight*(y2-y1): continue

        newBoxes.append([x1, y1, x2, y2])

    return newBoxes


def temporal_detection(newBoxes, Boxes, ImageQueue, frameID, maxGapBetween2frames):

    # check if boxes are finished
    for b in Boxes:
        if b.finish: continue
        b.finished(frameID, maxGapBetween2frames)

    # if new boxes correspond to existing boxes, add to them, else create a new one
    for newb in newBoxes:
        find=False
        for b in Boxes:
            if b.finish: continue
            if compare_boxes(newb, b):
                b.add_box(newb)
                b.fill_gap(ImageQueue)
                find=True
                break
        if not find:
            newb = box_image(newb)
            Boxes.append(newb)



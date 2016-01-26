import cv, cv2
import tesserpy
import numpy as np
from scipy import ndimage
import json

class box:
    """ list of box image of the same text
    lx1, ly1: top left corner
    lx2, ly2: bottom right corner
    meanx1, meany1, meanx2, meany2: mean value of the position
    lenghtInter: size after interpolation
    lframeId: frameId
    ldetected: is the box has been detected in the image (True or False)
    nbDetected: number of box that are detected
    finish: is the boxes is dead
    nbDetected: # detection
    lframe: list of images
    """

    def __init__(self, x1, y1, x2, y2, frameId):
        self.lx1 = [x1]
        self.ly1 = [y1]
        self.lx2 = [x2]
        self.ly2 = [y2]
        self.meanx1, self.meany1, self.meanx2, self.meany2 = float(x1), float(y1), float(x2), float(y2)
        self.lframeId = [frameId]
        self.finish = False
        self.nbDetected = 1
        self.lframe = [] 
        self.lenghtInter = 0

    def add_detected_box(self, x1, y1, x2, y2, frameId):
        self.lx1.append(x1)
        self.ly1.append(y1)
        self.lx2.append(x2)
        self.ly2.append(y2)
        self.lframeId.append(frameId)
        self.meanx1 = (self.meanx1 * self.nbDetected + float(x1)) / (self.nbDetected+1)
        self.meany1 = (self.meany1 * self.nbDetected + float(y1)) / (self.nbDetected+1)
        self.meanx2 = (self.meanx2 * self.nbDetected + float(x2)) / (self.nbDetected+1)
        self.meany2 = (self.meany2 * self.nbDetected + float(y2)) / (self.nbDetected+1)
        self.nbDetected += 1

    def OCR(self, freqReco, tess, resizeOCRImage):
        d = {}
        d["x"] = self.meanx1
        d["y"] = self.meany1
        d["w"] = self.meanx2-self.meanx1
        d["h"] = self.meany1-self.meany1

        # process OCR on mean images
        img = np.array(np.average(np.array(self.lframe), axis=0), dtype=np.uint8)
        img = cv2.resize(img, (self.lenghtInter, resizeOCRImage) , interpolation=cv2.INTER_CUBIC)
        tess.set_image(img)
        d["transcriptions"] = {}
        d["confidences"] = {}
        d["transcriptions"][str(self.lframeId[0])+"_to_"+str(self.lframeId[-1])] = tess.get_utf8_text().replace('\n', '')
        d["confidences"][str(self.lframeId[0])+"_to_"+str(self.lframeId[-1])] = [int(round(c.confidence, 0)) for c in tess.symbols()]

        # process OCR on intermediate images
        for i in range(int(len(self.lframe)/freqReco)):
            img = np.array(np.average(np.array(self.lframe[i*freqReco:(i+1)*freqReco]), axis=0), dtype=np.uint8)
            img = cv2.resize(img, (self.lenghtInter, resizeOCRImage) , interpolation=cv2.INTER_CUBIC)
            tess.set_image(img)
            d["transcriptions"][str(self.lframeId[i*freqReco])+"_to_"+str(self.lframeId[(i+1)*freqReco-1])] = tess.get_utf8_text().replace('\n', '')
            d["confidences"][str(self.lframeId[i*freqReco])+"_to_"+str(self.lframeId[(i+1)*freqReco-1])] = [int(round(c.confidence, 0)) for c in tess.symbols()]
            #transcriptions.append(tess.get_utf8_text().replace('\n', ''))
            #confidences.append([int(round(c.confidence, 0)) for c in tess.symbols()])

        if (i+1)*freqReco < len(self.lframe):
            img = np.array(np.average(np.array(self.lframe[(i+1)*freqReco:]), axis=0), dtype=np.uint8)
            img = cv2.resize(img, (self.lenghtInter, resizeOCRImage) , interpolation=cv2.INTER_CUBIC)
            tess.set_image(img)
            d["transcriptions"][str(self.lframeId[(i+1)*freqReco])+"_to_"+str(self.lframeId[-1])] = tess.get_utf8_text().replace('\n', '')
            d["confidences"][str(self.lframeId[(i+1)*freqReco])+"_to_"+str(self.lframeId[-1])] = [int(round(c.confidence, 0)) for c in tess.symbols()]
            #transcriptions.append(tess.get_utf8_text().replace('\n', ''))
            #confidences.append([int(round(c.confidence, 0)) for c in tess.symbols()])
        return d

    def __repr__(self):
        """print the last detection of the box"""
        return "last values: x1({}), y2({}), x2({}), y2({}), frameId({})".format(
                self.lx1[-1], self.ly1[-1], self.lx2[-1], self.ly2[-1], self.lframeId[-1])


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
    boxesDetected = []
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

        boxesDetected.append([x1, y1, x2, y2])

    return boxesDetected


def find_existing_boxes(x1, y1, x2, y2, imageQueue, frameId, boxes, tolBox, maxValDiffBetween2boxes):
    for b in boxes:
        if b.finish: continue

        # compare position
        x1Inter = max(x1, b.meanx1)
        y1Inter = max(y1, b.meany1)
        x2Inter = min(x2, b.meanx2)
        y2Inter = min(y2, b.meany2)

        if x1Inter>x2Inter or y1Inter>y2Inter : continue

        areaInter = (x2Inter-x1Inter)*(y2Inter-y1Inter)
        areaNewBox = (x2-x1)*(y2-y1)
        areab = (b.meanx2-b.meanx1)*(b.meany2-b.meany1)

        if areaInter/(areaNewBox+areab-areaInter) < tolBox: continue

        # compare image
        imgNewBox = imageQueue[0][y1Inter:y2Inter, x1Inter:x2Inter]

        imgb = imageQueue[frameId - b.lframeId[-1]][y1Inter:y2Inter, x1Inter:x2Inter]

        if np.mean(cv2.absdiff(imgNewBox, imgb).ravel()) > maxValDiffBetween2boxes: continue

        return b
    return False

def temporal_detection(boxesDetected, boxes, imageQueue, frameId, maxGapBetween2frames, tolBox, maxValDiffBetween2boxes):
    # if new boxes correspond to existing boxes, add to them, else create a new one
    newBoxes = []
    for x1, y1, x2, y2 in boxesDetected:
        boxFind = find_existing_boxes(x1, y1, x2, y2, imageQueue, frameId, boxes, tolBox, maxValDiffBetween2boxes)
        if boxFind:
            boxFind.add_detected_box(x1, y1, x2, y2, frameId)
        else:
            newBoxes.append([x1, y1, x2, y2])

    # for those neither existing boxes was found, create a newBoxes object
    for x1, y1, x2, y2 in newBoxes:
        newb = box(x1, y1, x2, y2, frameId)
        boxes.append(newb)

    # check if boxes are finished
    for b in boxes:
        if b.finish: continue
        if frameId - b.lframeId[-1] > maxGapBetween2frames: b.finish = True

    return boxes




#!/usr/bin/env python
# encoding: utf-8

"""
Video text extractor

Usage:
  pyLOOV.py <video> <output> [--verbose] [--mask=<path>] [--tessdata=<path>] [--lang=<abc>] [--ss=<frameID>] [--endpos=<frameID>] [--onlyTextDetection] [--thresholdSobel=<Thr>] [--itConnectedCaractere=<it>] [--yMinSizeText=<thr>] [--xMinSizeText=<thr>] [--minRatioWidthHeight=<thr>] [--marginCoarseDetectionY=<marge>] [--marginCoarseDetectionX=<marge>] [--tolBox=<thr>] [--minDurationBox=<thr>] [--maxGapBetween2frames=<thr>] [--maxValDiffBetween2boxes=<thr>] [--freqReco=<freq>] [--resizeOCRImage=<size>] [--authorizedCharacters=<list>]
  pyLOOV.py -h | --help
Options:
  -h --help                         Show this screen.
  --verbose                         print massages
  --mask=<path>                     image in black and white to process only a subpart of the image
  --tessdata=<path>                 path to tessdata, [default: /usr/local/share/tessdata/]
  --lang=<abc>                      language [default: eng]
  --ss=<frameID>                    start frame to process, [default: 0]
  --endpos=<frameID>                last frame to process, [default: -1]
  --onlyTextDetection               process only the text detection, [default: False]
  --thresholdSobel=<Thr>            value of the threshold sobel, [default: 103]
  --itConnectedCaractere=<it>       iteration number for the connection of the edges of characters after the sobel operator, calculation : iteration = 352(width of the image) * 0.03(itConnectedCaractere) = 11 iterations, [default: 0.03]
  --yMinSizeText=<thr>              minimum pixel height of a box, calculation : iteration = 288(height of the image) * 0.02(yMinSizeText) = 6, [default: 0.02]
  --xMinSizeText=<thr>              minimum pixel width of a box, calculation : iteration = 352(width of the image) * 0.05(xMinSizeText) = 18, [default: 0.05]
  --minRatioWidthHeight=<thr>       minimum ratio between height and width of a text box, [default: 2.275]
  --marginCoarseDetectionY=<marge>  margin arround the box after coarse detection, calculation : margin = 10(height of the image box) * 0.5(marginCoarseDetectionY) = 5 pixels, [default: 0.5]
  --marginCoarseDetectionX=<marge>  margin arround the box after coarse detection, calculation : margin = 10(height of the image) * 1(marginCoarseDetectionX) = 10 pixels, [default: 1.0]
  --tolBox=<thr>                    tolerance of the f-measure between to box on 2 consecutive frames, [default: 0.5]
  --minDurationBox=<thr>            min duration of a box in frame, [default: 19]
  --maxGapBetween2frames=<thr>      max space between to frames to continue detection of a box, [default: 14]
  --maxValDiffBetween2boxes=<thr>   max value in average color between two box compared, [default: 20.0]
  --freqReco=<freq>                 make text recognition every freqReco frames, [default: 10]
  --resizeOCRImage=<size>           height of the image box after bicubic interpolation, [default: 200]
  --authorizedCharacters=<list>     file with the list of autorized characters
"""

from docopt import docopt
import numpy as np
import cv, cv2
import os, sys
from collections import deque
import io, json

from text_detection import *

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    video                   = args['<video>']
    output                  = args['<output>']
    verbose                 = args['--verbose']
    applyMask = False
    mask = None
    if args['--mask']: 
        mask = cv2.imread(args['--mask'], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        applyMask = True

    tessdata                = args['--tessdata']
    lang                    = args['--lang']
    onlyTextDetection       = args['--onlyTextDetection']
    ss                      = max(1, int(args['--ss']))
    endpos                  = int(args['--endpos'])
    thresholdSobel          = int(args['--thresholdSobel'])
    itConnectedCaractere    = float(args['--itConnectedCaractere'])
    yMinSizeText            = float(args['--yMinSizeText'])
    xMinSizeText            = float(args['--xMinSizeText'])
    minRatioWidthHeight     = float(args['--minRatioWidthHeight'])
    marginCoarseDetectionY  = float(args['--marginCoarseDetectionY'])
    marginCoarseDetectionX  = float(args['--marginCoarseDetectionX'])
    tolBox                  = float(args['--tolBox'])
    minDurationBox          = int(args['--minDurationBox'])
    maxGapBetween2frames    = int(args['--maxGapBetween2frames'])
    maxValDiffBetween2boxes = float(args['--maxValDiffBetween2boxes'])
    freqReco                = int(args['--freqReco'])
    resizeOCRImage          = int(args['--resizeOCRImage'])
    authorizedCharacters    = args['--authorizedCharacters']  

    # check if the character recognition model is found
    if not onlyTextDetection:
        if not os.path.isfile(tessdata+"/"+lang+".traineddata"): 
            print tessdata+"/"+lang+".traineddata could not be found"
            sys.exit(1)
        tess = tesserpy.Tesseract(tessdata, language=lang)
        if authorizedCharacters: tess.tessedit_char_whitelist = open(authorizedCharacters).read()

    # open video
    capture = cv2.VideoCapture(video)
    lastFrameCapture = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

    # check if ss and endpos are inside the video
    if ss >= lastFrameCapture: 
        print "the video lengh ("+str(lastFrameCapture)+") is under ss position("+str(ss)+")"
        sys.exit(1)
    if endpos == -1: endpos = lastFrameCapture
    if endpos > lastFrameCapture: 
        print "the last frame to process ("+str(endpos)+") is higher than the video lengh ("+str(lastFrameCapture)+")"
        sys.exit(1)
    if ss > endpos: 
        print "first frame to process ("+str(lastFrameCapture)+") is higher than the last frame to process("+str(endpos)+")"
        sys.exit(1)

    # seek to the begin of the segment to process
    ret, frame = capture.read()                             # get next frame
    frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES)) 
    while (frameId<ss):
        ret, frame = capture.read()                             # get next frame
        frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES)) 

    # adapt parameters to the size of the image
    height, width, channels = frame.shape
    itConnectedCaractere = int(round(width*itConnectedCaractere, 0))
    yMinSizeText = int(round(height*yMinSizeText, 0))
    xMinSizeText = int(round(width*xMinSizeText, 0))

    # read video
    if verbose : print "processing of frames from", ss, "to", endpos
    boxes = []
    imageQueue = deque([], maxlen=maxValDiffBetween2boxes+1)
    while (frameId<endpos+1):
        if ret:
            # convert image to grau scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply mask to process on a subpart of the image
            if applyMask: frame = cv2.bitwise_and(frame, frame, mask = mask)
            # add the current image to the queue
            imageQueue.appendleft(frame)
            # detect boxes spatially
            boxesDetected = spatial_detection_LOOV(frame, applyMask, mask, height, width, thresholdSobel, itConnectedCaractere, yMinSizeText, xMinSizeText, marginCoarseDetectionX, marginCoarseDetectionY, minRatioWidthHeight)
            # Check if boxes have temporal consistency
            boxes = temporal_detection(boxesDetected, boxes, imageQueue, frameId, maxGapBetween2frames, tolBox, maxValDiffBetween2boxes)
        else:
            # if the current frame is not good, add the previous last good frame to the queue
            frame = np.copy(imageQueue[0])

        # if the queue is full, remove the oldest frame from it
        if len(imageQueue)>maxValDiffBetween2boxes+1:  imageQueue.pop()

        if verbose and frameId % 100 == 0: print "frameId:", frameId

        # get next frame
        ret, frame = capture.read() 
        frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))
    capture.release()

    #select only box with a good duration
    boxesFinal = []
    for b in boxes:
        if b.lframeId[-1] - b.lframeId[0] >=  minDurationBox:
            #enlarge a litlle be the box
            b.meanx1 = int(round(b.meanx1,0)) -  2
            b.meany1 = int(round(b.meany1,0)) -  2
            b.meanx2 = int(round(b.meanx2,0)) +  2
            b.meany2 = int(round(b.meany2,0)) +  2
            # calcul lengh of the resized image
            b.lenghtInter = int(round( resizeOCRImage*(b.meanx2-b.meanx1)/(b.meany2-b.meany1)))
            # fill gap of the list of frameId
            b.lframeId = range(b.lframeId[0], b.lframeId[-1]) 
            # and append the box to final list
            boxesFinal.append(b)
    if verbose: print "number of boxes", len(boxesFinal)

    # read the video a second time to get the ROI images corresponding to boxes
    capture = cv2.VideoCapture(video)
    frameId = -1
    # seek to the begin of the segment to process
    while (frameId<ss-1):
        ret, frame = capture.read()                             # get next frame
        frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES)) 
    
    # for each boxes, if the current frame is within, get the ROI image based on mean position spatial
    while (frameId<endpos):
        ret, frame = capture.read()                             # get next frame
        if not ret: continue
        frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        if applyMask: frame = cv2.bitwise_and(frame, frame, mask = mask)

        for b in boxesFinal:            
            if frameId >= b.lframeId[0] and frameId <= b.lframeId[-1]:
                img = np.copy(frame[b.meany1:b.meany2, b.meanx1:b.meanx2])
                #   apply a black rectangulaire arroud the ROI due to the enlargement done before
                cv2.rectangle(img, (0,0), (b.meanx2-b.meanx1, b.meany2-b.meany1), (0,0,0), 2, 4, 0)
                b.lframe.append(img)
    capture.release()

    # process transcription
    d = {}
    i=0    
    if not onlyTextDetection:
        for b in boxesFinal:
            d[i] = b.OCR(freqReco, tess, resizeOCRImage, i)
            i+=1
    # save the output in json format
    with io.open(output, 'w', encoding='utf-8') as f:
        #json.dump(d, f, sort_keys = True, indent = 4)
        f.write(unicode(json.dumps(d, ensure_ascii=False)))


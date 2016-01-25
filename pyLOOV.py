"""
Video text extractor

Usage:
  pyLOOV.py <video> <output>   [--mask=<path>] [--traindata=<path>] [--ss=<frameID>] [--endpos=<frameID>] [--onlyTextDetection] [--thresholdSobel=<Thr>] [--itConnectedCaractere=<it>] [--yMinSizeText=<thr>] [--xMinSizeText=<thr>] [--minRatioWidthHeight=<thr>] [--marginCoarseDetectionY=<marge>] [--marginCoarseDetectionX=<marge>] [--tolBox=<thr>] [--minDurationBox=<thr>] [--maxGapBetween2frames=<thr>] [--maxValDiffBetween2boxes=<thr>] [--freqReco=<freq>] [--resizeOCRImage=<size>] 
  pyLOOV.py -h | --help
Options:
  -h --help                         Show this screen.
  --mask=<path>                     image in black and white to process only a subpart of the image
  --traindata=<path>                path to tessdata, [default: /usr/local/share/tessdata/fra.traineddata]
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
"""


from docopt import docopt
import numpy as np
import cv, cv2
import os
from collections import deque

from text_detection import *

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    # parse arguments
    video                   = args['<video>']
    output                  = open(args['<output>'], 'w')
    if args['--mask']: mask = cv2.rimead(args['--mask'])
    else:              mask = False
    traineddata             = args['--traindata']
    onlyTextDetection       = args['--onlyTextDetection']
    ss                      = int(args['--ss'])
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

    # check if the character recognition model is found
    if not os.path.isfile(traineddata): raise ValueError(traineddata+" could not be found")

    # open video
    capture = cv2.VideoCapture(video)
    lastFrameCapture = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

    # check if ss and endpos are inside the video
    if ss >= lastFrameCapture: raise ValueError("the video lengh ("+str(lastFrameCapture)+") is under ss position("+str(ss)+")")
    if endpos == -1: endpos = lastFrameCapture
    if endpos > lastFrameCapture: raise ValueError("the last frame to process("+str(endpos)+") is higher than the video lengh ("+str(lastFrameCapture)+")")

    frameId = -1
    while (frameId<ss-1):
        ret, frame = capture.read()                             # get next frame
        frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES)) 

    # adapt parameters to the image size
    height, width, channels = frame.shape
    itConnectedCaractere = int(round(width*itConnectedCaractere, 0))
    yMinSizeText = int(round(height*yMinSizeText, 0))
    xMinSizeText = int(round(width*xMinSizeText, 0))

    # read video
    boxes = []
    imageQueue = deque([], maxlen=maxValDiffBetween2boxes+1)

    print "processing of frames from", ss, "to", endpos
    while (frameId<endpos):
        ret, frame = capture.read()                             # get next frame
        if ret:
            frameId = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imageQueue.appendleft(frame)
            boxesDetected = spatial_detection_LOOV(frame, mask, height, width, thresholdSobel, itConnectedCaractere, yMinSizeText, xMinSizeText, marginCoarseDetectionX, marginCoarseDetectionY, minRatioWidthHeight)
            boxes         = temporal_detection(boxesDetected, boxes, imageQueue, frameId, maxGapBetween2frames, tolBox, maxValDiffBetween2boxes)
            if len(imageQueue)>maxValDiffBetween2boxes+1:  imageQueue.pop()
        else:
            imageQueue.appendleft(frame)
            if len(imageQueue)>maxValDiffBetween2boxes+1: imageQueue.pop()            
    capture.release()

    boxesFinal = []
    #select only box with a good duration
    for b in boxes:
        if b.lframeId[-1] - b.lframeId[0] >=  minDurationBox:
            boxesFinal.append(b)

    # process transcription
    if not onlyTextDetection:
        for b in boxesFinal:
            b.compute_mean_images(freqReco)
            b.OCR()

    # save boxes
    for b in boxesFinal:
        b.save(output)
    output.close()

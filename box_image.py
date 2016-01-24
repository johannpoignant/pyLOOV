


class box_image:
    """ list of box image of the same text
    transcriptions: list of transcriptions obtained by the OCR
    x1, y1: top left corner
    x2, y2: bottom right corner
    frameId: frameId
    img: image of the boxes
    thr: threshold find to binarised the image
    nbCC: number of connected component in the image
    finish: is the boxes is dead
    """

    def __init__(self, newb):
        x, y, w, h, frameId, img, thr = newb
        self.transcriptions = []
        self.x1 = [x1]
        self.y1 = [y1]
        self.x2 = [x2]
        self.y2 = [y2]
        self.frameId = [frameId]
        self.image = [img]
        self.thr = [thr]
        self.nbCC = [nbCC]
        self.finish = False

    def add_box(newb):
        x, y, w, h, frameId, img, thr = newb
        self.x1.append(x1)
        self.y1.append(y1)
        self.x2.append(x2)
        self.y2.append(y2)
        self.frameId.append(frameId)
        self.image.append(img)
        self.thr.append(thr)
        self.nbCC.append(nbCC)

    def finished(currentFrameID, maxGapBetween2frames):
        if self.frameId[-1] + maxGapBetween2frames < currentFrameID:
            self.finish = True        

    def fill_gap(ImageQueue):
        # fill the gap between image
        pass

    def sort_image():
        # sort all list by frameId
        pass

    def compute_mean_images(freqReco):
        # compute mean image every freqReco frame
        pass

    def OCR():
        # process OCR on mean images
        pass

    def save(output):
        # write transcription on file
        pass

    def __repr__(self):
        """print the last detection of the box"""
        return "last values: x1({}), y2({}), x2({}), y2({}), frameId({}), thr({}), nbCC({})".format(
                self.x1[-1], self.y1[-1], self.x2[-1], self.y2[-1], self.frameId[-1], self.thr[-1], self.nbCC[-1])


def compare_boxes(newb, b):

    return False

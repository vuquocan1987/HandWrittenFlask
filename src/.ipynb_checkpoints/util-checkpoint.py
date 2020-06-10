import cv2
import numpy as np
import tensorflow as tf
import os
import datetime
import string
from .network.model import HTRModel
from .data.generator import DataGenerator,Tokenizer

from .data.preproc import normalization

def getParagraph(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((120,200), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, im2 = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(ctrs)
    boundRect = [None]*len(ctrs)
    for i, c in enumerate(ctrs):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    areaBoundRect = [rect[2]*rect[3] for rect in boundRect]
    index = np.argmax(areaBoundRect)
    max_box = boundRect[index]
    x,y,w,h = max_box
    return image[y:y+h,x:x+w]
def getLines(para):
    gray = cv2.cvtColor(para,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,200), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(ctrs)
    boundRect = [None]*len(ctrs)
    for i, c in enumerate(ctrs):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
#     return boundRect
    lines = []
    for bb in boundRect:
        x,y,w,h = bb
        line = gray[y:y+h,x:x+w]
        lines.append(line)
    return lines
def pre_process(lines,dsize):
    new_lines = []
    for line in lines:
        
        line = line.T
        print(line.shape)
        new_line = cv2.resize(line,dsize)
        print(new_line.shape)
        new_lines.append(new_line)
    return np.array(new_lines)
def creat_model():
    source = "iam"
    arch = "flor"
    epochs = 1000
    batch_size = 16

    # define paths
    source_path = os.path.join("./src", "data", f"{source}.hdf5")
    output_path = os.path.join("./src", "output", source, arch)
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
    print(os.path.isfile(target_path))
    print(os.path.abspath(target_path))
    os.makedirs(output_path, exist_ok=True)

    # define input size, number max of chars per line and list of valid chars
    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]
    print('pwd',os.getcwd())
    print("source:", source_path)
    print("output", output_path)
    print("target", target_path)
    print("charset:", charset_base)
    tokenizer = Tokenizer(charset_base)
    model = HTRModel(architecture=arch, input_size=input_size, vocab_size=tokenizer.vocab_size)
    model.compile(learning_rate=0.001)
    model.summary(output_path, "summary.txt")
# get default callbacks and load checkpoint weights file (HDF5) if exists
    model.load_checkpoint(target=target_path)
    callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)
    return tokenizer,model


class HandWrittingPredictor:
    def __init__(self):
        self.tokenizer, self.model = creat_model()
    def predict(self,img):
        para = getParagraph(img)
        lines = getLines(para)
        preprocessed_lines = pre_process(lines, (128, 1024))
        print(preprocessed_lines.shape)
        preprocessed_lines = normalization(preprocessed_lines)
        predicts, _ = self.model.predict(preprocessed_lines)
        predicts_ = [self.tokenizer.decode(x[0]) for x in predicts]
        return predicts_
if __name__ == "__main__":
    hw = HandWrittingPredictor()
    print(os.path.isfile('form.png'))
    print(os.getcwd())
    image = cv2.imread('./src/form.png')
    print(image)
    predicts = hw.predict(image)
    print(predicts)

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import glob

PATH = "/home/wilfred/Datasets/flower_photos"
cnt = 0

def imageDisp():

if __name__ == "__main__":

    dirs = os.listdir(PATH)
    print(dirs)
    for dirs_ in dirs:
        cnt += len(glob.glob(os.path.join(PATH,dirs_)+'/*.jpg'))

    roses = glob.glob(os.path.join(PATH,'roses')+'/*.jpg')
    img = PIL.Image.open(roses[1])
    img.show()

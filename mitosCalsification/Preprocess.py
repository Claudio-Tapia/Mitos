import os

import cv2
from PyQt5.QtCore import QDir

from common.Params import Params as P


def rotateImageAndSave(image, baseName, path):
    k = 1
    rows, cols, _ = image.shape
    while k <= 3:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90*k, 1)
        res = cv2.warpAffine(image, M, (cols, rows))
        name = path+baseName+'-rot-'+str(90*k)+'.png'
        cv2.imwrite(name, res)
        k += 1

def pre_process():
    #dirTrain1 = QDir('C:/Users/home/Desktop/mitos dataset/train/mitosis')
    dirTrain2 = QDir(P().saveCutMitosDir)
    imagesFilter = ['*.png', '*.tif', '*.bmp']

    #info1 = dirTrain1.entryInfoList(imagesFilter)
    info2 = dirTrain2.entryInfoList(imagesFilter)
    infoList = []
    #infoList.extend(info1)
    infoList.extend(info2)

    i = 0
    for fileInfo in infoList:
        i += 1
        imagePath = fileInfo.absoluteFilePath()
        basePath = fileInfo.absolutePath() +'/'
        basenameExt = os.path.basename(imagePath)
        baseName, _ = os.path.splitext(basenameExt)
        savePath = P().saveMitosisPreProcessed

        imbase = cv2.imread(imagePath)
        rows, cols, _ = imbase.shape
        imXMirror = cv2.flip(imbase, 1)

        cv2.imwrite(savePath + fileInfo.fileName(), imbase)
        cv2.imwrite(savePath+baseName+'-mirror.png', imXMirror)
        rotateImageAndSave(imbase, baseName, savePath)
        rotateImageAndSave(imXMirror, baseName+'-mirror', savePath)

        print('%d / %d'%(i, len(infoList)))


def rotate_image(image):
    k = 1
    rows, cols, _ = image.shape
    imlist = []
    while k <= 3:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 * k, 1)
        res = cv2.warpAffine(image, M, (cols, rows))
        imlist.append(res)
        k += 1

    return imlist

def preprocess_in_memory(im_list):
    new_images_list = []
    for im in im_list:
        im_x_flip = cv2.flip(im, 1)
        new_images_list.extend(rotate_image(im))
        new_images_list.extend(rotate_image(im_x_flip))
        new_images_list.append(im_x_flip)

    im_list.extend(new_images_list)

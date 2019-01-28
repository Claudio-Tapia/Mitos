from PyQt5.QtCore import QDir
import os
import cv2

dirTrain1 = QDir('C:/Users/home/Desktop/mitos dataset eval/cutted/mitosis')
dirTrain2 = QDir('C:/Users/home/Desktop/mitos dataset eval/cutted/noMitos')

imagesFilter = ['*.png', '*.tif', '*.bmp']
infoList1 = dirTrain1.entryInfoList(imagesFilter)
infoList2 = dirTrain2.entryInfoList(imagesFilter)

for e in infoList1:
    filePath = e.absoluteFilePath()
    basenameExt = os.path.basename(filePath)
    baseName, _ = os.path.splitext(basenameExt)

    im = cv2.imread(filePath)
    savePath = dirTrain1.absolutePath()+'/'+baseName+'.png'
    cv2.imwrite(savePath, im)

for e in infoList2:
    filePath = e.absoluteFilePath()
    basenameExt = os.path.basename(filePath)
    baseName, _ = os.path.splitext(basenameExt)

    im = cv2.imread(filePath)
    cv2.imwrite(dirTrain2.absolutePath()+'/' + baseName + '.png', im)

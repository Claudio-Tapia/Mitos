#import sys
#sys.path.append('C:/Users/PelaoT/Desktop/Practica/codigo')

import json

import cv2
import numpy as np

from common import utils


def debugAnotations(im, centroids, winsize = 63):
    im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        cx = c[0]
        cy = c[1]
        x = cx - (int(winsize / 2))
        y = cy - (int(winsize / 2))
        cv2.rectangle(im2, (x,y), (x+winsize, y+winsize), (0,0,255), 3)

    cv2.imwrite('holap.png', im2)

def generatePoint (splittedString):
    i = 0
    pointsList = []
    while i < len(splittedString):
        x = splittedString[i]
        y = splittedString[i+1]
        pointsList.append([x,y])
        i += 2

    return np.array(pointsList).astype(np.int32)

def paintAnotations(points):
    im = np.zeros((2084, 2084), np.uint8)
    for point in points:
        x = point[:,0]
        y = point[:,1]
        im[y,x] = 255

    #cv2.imwrite('holap.png', im)
    return im

def findCentroidsByBounRect(point):
    x,y,w,h = cv2.boundingRect(point)
    cx = int(x + (w/2))
    cy = int(y + (h/2))
    return cy, cx


baseDir = 'D:/Descargas/mitosis_evaluation_set_A/test/'
filter = ['*.csv']

fileList = utils.listFiles(baseDir, filter)
jsonDict = {}

i= 1
total = len(fileList)

for fileInfo in fileList:
    mitosRegion = []
    csvPath = fileInfo.absoluteFilePath()
    csvFile = open(csvPath)

    for line in csvFile:
        splitted = str.split(line, ',')
        point = generatePoint(splitted)
        center = findCentroidsByBounRect(point)
        pointDict = {"row" : center[0], "col" : center[1]}
        mitosRegion.append(pointDict)

    jsonDict[fileInfo.baseName()] = mitosRegion

    suffix = str(i) + "/" + str(total) + " Completed"
    utils.printProgressBar(i, total, 'Progress ', suffix, length=50, fill='=')
    i += 1

parsed = json.dumps(jsonDict, indent=4, sort_keys=True)

file = open('D:/Descargas/mitosis_evaluation_set_A/test/MitosEval.json', 'w')
file.write(parsed)
file.close()

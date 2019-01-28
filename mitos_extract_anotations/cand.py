import sys
sys.path.append('C:/Users/PelaoT/Desktop/Practica/codigo')

from mitos_extract_anotations import candidateSelection as cs
from common.utils import listFiles
import sys


filters = ['*.bmp', '*.png', '*.jpg']
if sys.platform == 'win32':
    file_list = listFiles('C:/Users/PelaoT/Desktop/Practica/dataset/eval/heStain/', filters)
    params = cs.Candidates_extractor_params(file_list)
    params.save_candidates_dir_path = 'C:/Users/PelaoT/Desktop/Practica/dataset/eval/no-mitosis/'
    params.save_mitosis_dir_path = 'C:/Users/PelaoT/Desktop/Practica/dataset/eval/mitosis/'
    params.candidates_json_save_path = 'C:/Users/PelaoT/Desktop/Practica/dataset/eval/test.json'
else:
    file_list = listFiles('/home/facosta/dataset/normalizado/testHeStain/', filters)
    params = cs.Candidates_extractor_params(file_list)
    params.save_candidates_dir_path = '/home/facosta/dataset/test/no-mitosis/'
    params.save_mitosis_dir_path = '/home/facosta/dataset/test/mitosis/'
    params.candidates_json_save_path = '/home/facosta/dataset/test//test.json'

extractor = cs.Candidates_extractor(params)
extractor.extract()

#
# import cv2
# import numpy as np
#
# def get_center(rectangle):
#     x,y,w,h = rectangle
#
#     cx = int(x + w / 2)
#     cy = int(y + h / 2)
#     return cx, cy
#
# im = cv2.imread('C:/Users/felipe/mitos dataset/normalizado/heStain/A00_01.bmp')
# b,g,r = cv2.split(im)
# b = b.astype(np.float64)
# g = g.astype(np.float64)
# r = r.astype(np.float64)
# br = (100 * b / (1 + r + g)) * (256 / (1 + b + r + g))
# br = br.astype(np.uint8)
#
#
# mean, std = cv2.meanStdDev(br)
# mean = mean[0][0]
# std = std[0][0]
#
# _, thresh = cv2.threshold(br, mean + std, 255, cv2.THRESH_BINARY)
# kernel = np.ones((7,7), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# # cv2.imwrite('C:/Users/felipe/b.png', thresh)
#
# _,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.imshow('hol',thresh)
# # cv2.waitKey()
#
# candidates = []
# i = 0
# while i < len(contours):
#     c = contours[i]
#     area = cv2.contourArea(c)
#     if area < 80 or area > 1800:
#         del contours[i]
#         continue
#
#     rectangle = cv2.boundingRect(c)
#     point = get_center(rectangle)
#     candidates.append(point)
#     cv2.circle(im, point, 20, (0,0,255), 2)
#     i += 1
#
# im2 = np.zeros(im.shape, np.uint8)
# cv2.drawContours(im2, contours, -1, (255,255,255), cv2.FILLED)
# cv2.imwrite('C:/Users/felipe/ab.png', im)

import sys
sys.path.append('C:/Users/PelaoT/Desktop/Practica/codigo')

import cv2
from common import utils
from common.Params import Params as P
from mitos_extract_anotations import ImCutter as cut
import json
import numpy as np


def join_path(path, add):
    if path[len(path) - 1] != '/':
        return path + '/' + add
    else:
        return path + add


def get_center(rectangle):
    x,y,w,h = rectangle

    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy


def morph_extract(im):
    im_copy = np.copy(im)
    im2 = np.copy(im)
    b,g,r = cv2.split(im_copy)
    b = b.astype(np.float64)
    g = g.astype(np.float64)
    r = r.astype(np.float64)
    br = (100 * b / (1 + r + g)) * (256 / (1 + b + r + g))
    br = br.astype(np.uint8)


    mean, std = cv2.meanStdDev(br)
    mean = mean[0][0]
    std = std[0][0]
    # std *= 0.9

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(br, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(br, mean + std, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('C:/Users/felipe/b.png', thresh)


    _,contours,_= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('hol',thresh)
    # cv2.waitKey()

    candidates = []
    i = 0
    while i < len(contours):
        c = contours[i]

        area = cv2.contourArea(c)
        if area < 70 or area > 1800:
            del contours[i]
            continue

        rectangle = cv2.boundingRect(c)
        point = get_center(rectangle)
        candidates.append(point)
        cv2.circle(im2, point, 20, (0,0,255), 2)
        i += 1

    # cv2.imwrite('C:/Users/felipe/b.png', im2)
    return candidates


class Candidates_extractor_params:
    def __init__(self, file_list):
        self.save_candidates_dir_path = P().saveCutCandidatesDir
        self.save_mitosis_dir_path = P().saveCutMitosDir
        self.file_list = file_list
        self.candidates_json_save_path = P().candidatesTrainingJsonPath
        self.img_with_keypoints_save_path = P().saveCandidatesWholeImgDir
        self.candidates_save_as_tar = True
        self.mitosis_save_as_tar = False
        self.write_img_to_disk = True
        self.bsave_img_keypoints = False
        self.bappend_mitosis_to_json = False
        self.extract_windows = P().candidates_size

class Candidates_extractor:
    def __init__(self, params):
        assert isinstance(params, Candidates_extractor_params)
        self.params = params
        self.candidate_cutter = cut.ImCutter(params.save_candidates_dir_path,
                                             save_as_tar=params.candidates_save_as_tar,
                                             cut_size=params.extract_windows)
        if params.save_mitosis_dir_path is not None:
            self.mitos_cutter = cut.ImCutter(params.save_mitosis_dir_path,
                                             save_as_tar=params.mitosis_save_as_tar,
                                             cut_size=params.extract_windows)
        else:
            self.mitos_cutter = None

        self.verificator = utils.MitosVerification()
        self.candidates_count = 0
        self.candidates_dict = {}

    def extract(self):
        progress = 0

        utils.print_progress_bar(0, len(self.params.file_list))

        for file in self.params.file_list:
            progress += 1
            color_image = cv2.imread(file.absoluteFilePath())

            color_image_copy = np.copy(color_image)

            keypoints = morph_extract(color_image)

            self.candidate_cutter.set_image(color_image, file.baseName())
            if self.mitos_cutter is not None:
                self.mitos_cutter.set_image(color_image, file.baseName())
                self.verificator.set_base_name(file.baseName())

            candidates = self._extract_candidates_position(keypoints, file.baseName())
            self.candidates_dict[file.baseName()] = candidates

            if self.params.write_img_to_disk:
                mitos_list = self.verificator.get_mitos_points(file.baseName())
                self._write_img_disk(candidates, mitos_list)

            if self.params.bsave_img_keypoints:
                self.__save_img_with_key_points(color_image_copy, candidates)

            if self.mitos_cutter is not None:
                self.verificator.print_verification_result()
            utils.print_progress_bar(progress, len(self.params.file_list))

        self.finish()


    def _extract_candidates_position(self, keypoints, base_name):
        candidates_list = []

        for k in keypoints:
            #point = (int(k.pt[0]), int(k.pt[1]))
            point = k

            merged, new_point = self._merge_if_close(candidates_list, cand_point=point)
            if self.mitos_cutter is not None and self.verificator.is_mitos(point):
                #mitosis_list.append(point)
                if not merged and self.params.bappend_mitosis_to_json:
                     candidates_list.append({"row": point[0], "col": point[1]})
                elif merged and self.params.bappend_mitosis_to_json == False:
                # esta línea nunca se ejecutó, ni idea si funciona... debería...espero
                    candidates_list.remove(new_point)
            else:
                if not merged:
                    candidates_list.append({"row": point[0], "col": point[1]})
                    self.candidates_count += 1

        return candidates_list

    def _merge_if_close(self, cand_list, cand_point, max_dist = 35):
        for k in cand_list:
            point = (k['row'], k['col'])
            dist = utils.euclidianDistance(point, cand_point)
            if dist < max_dist:
                new_point = self._get_center_point(point, cand_point)
                k['row'] = new_point[0]
                k['col'] = new_point[1]
                return True, k

        return False, None

    def _get_center_point(self, point1, point2):
        x = int((point1[0] + point2[0]) / 2)
        y = int((point1[1] + point2[1]) / 2)
        return (x,y)

    def _cand_2_dict(self, candidates):
        candidate_point = {}
        count = 0
        for point in candidates:
            candidate_point[count] = point
            count += 1

        return candidate_point

    def _write_img_disk(self, candidates_list, mitosis_list):
        for point in candidates_list:
            # i think the dimensions of opencv python are too confusing for me,
            # but at least this works... i hope so
            self.candidate_cutter.cut_and_save(point["row"], point["col"])

        if self.mitos_cutter is not None:
            for point in mitosis_list:
                # just dont ask why...
                self.mitos_cutter.cut_and_save(point["col"], point["row"])

    def __save_img_with_key_points(self, image, keypoints):
        im_with_keypoints = image.copy()
        base_name = self.candidate_cutter._base_name + '.jpg'
        save_path = join_path(self.params.img_with_keypoints_save_path, base_name)
        for k in keypoints:
            point = (int(k['row']), int(k['col']))
            # point = (k[0], k[1])
            size = 20 # int(k.size)
            cv2.circle(im_with_keypoints, point, size, (0,0,255), 2)

        not_found_mitos = self.verificator.not_found_points
        for p in not_found_mitos:
            point = (p['col'], p['row'])
            size = 25
            cv2.circle(im_with_keypoints, point, size, (0,255,0), 2)

        mitosis_points = self.verificator.get_mitos_points(self.candidate_cutter._base_name)
        for m in mitosis_points:
            point = (m['col'], m['row'])
            size = 35
            # cv2.circle(im_with_keypoints, point, size, color=(255,153,77), thickness=3)

        cv2.imwrite(save_path, im_with_keypoints)

    def finish(self):
        self.candidate_cutter.close_tar()

        if self.mitos_cutter is not None:
            self.mitos_cutter.close_tar()

        json_string = json.dumps(self.candidates_dict, sort_keys=True, indent=4)

        with open(self.params.candidates_json_save_path, 'w') as file:
            file.write(json_string)

        print('Total de candidatos: {}'.format(self.candidates_count))


if __name__ == "__main__":
    filter = ['*.bmp', '*.png', '*.jpg']
    file_list = utils.listFiles(P().normHeStainDir, filter)
    train_list = file_list[0:30]
    test_list = file_list[-5:]
    p = Candidates_extractor_params(train_list)
    c = Candidates_extractor(p)
    c.extract()

    # extract testing dataset
    param = Candidates_extractor_params(test_list)
    param.save_candidates_dir_path = P().saveTestCandidates
    param.save_mitosis_dir_path = P().saveTestMitos
    c = Candidates_extractor(param)
    c.extract()

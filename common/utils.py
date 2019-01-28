import io
import json
import math
import sys
import tarfile
import numpy as np

import cv2
from PyQt5.QtCore import QDir

from common.Params import Params as P


# description of the parameters:
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

def createBlobDetector():
    params = P().blobDetectorParams
    return cv2.SimpleBlobDetector_create(params)


def listFiles(folderPath, filters=[]):
    folder = QDir(folderPath)
    fileList = folder.entryInfoList(filters)
    return fileList


def getInputDim():
    import keras.backend as K

    img_width = P().model_input_size
    img_height = P().model_input_size

    if K.image_data_format() == 'channels_first':
        dim = (3, img_width, img_height)
    else:
        dim = (img_width, img_height, 3)

    return dim


def keras_deep_copy_model(model):
    from keras.models import Sequential, Model

    config = model.get_config()
    weights = model.get_weights()
    if isinstance(model, Sequential):
        new_model = Sequential.from_config(config)
    else:
        new_model = Model.from_config(config)

    new_model.set_weights(weights)
    loss = model.loss
    metrics = model.metrics
    optimizer = model.optimizer
    new_model.compile(optimizer, loss, metrics)
    return new_model


def write_test_output(true_output, pred_output, name=None):
    i = 0
    if name is None:
        file = open('resulados.txt', 'w')
    else:
        file = open(name + '.txt', 'w')
    file.write('true\tpred\n')
    while i < len(true_output):
        file.write('{}\t\t{}\n'.format(true_output[i], pred_output[i]))
        i += 1

    file.close()


# snippet found in http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


def print_progress_bar(progress, total):
    suffix = "{}/{} Completed".format(progress, total)  # message for progress bar
    prefix = "Progress "
    printProgressBar(progress, total, prefix, suffix, length=40, fill='=')


class Coordinate:
    def __init__(self, x, y, img_base_name):
        self.x = x
        self.y = y
        self.img_base_name = img_base_name

    def to_dict(self):
        return {'col': self.y, 'row': self.x}


def euclidianDistance(p1, p2):
    dy = p1[0] - p2[0]
    dx = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


class MitosVerification:
    COLUMN = 0
    ROW = 1

    def __init__(self):
        file = open(P().mitosAnotationJsonPath)
        string = file.read()
        self.jsonDict = json.loads(string)
        file.close()
        self.verificated_mitos = 0
        self.base_name = None
        self.not_found_points = []

    def is_mitos(self, candidatePoint, minDist=20):
        mitosCenterList = self.get_mitos_points(self.base_name)

        for point in mitosCenterList:
            p = (point["col"], point["row"])
            dist = euclidianDistance(candidatePoint, p)
            if dist < minDist:
                self.verificated_mitos += 1
                try:
                    self.not_found_points.remove(point)
                except ValueError:
                    # this happens when two points are close to the same
                    # mitotic cell
                    self.verificated_mitos -= 1

                return True

        return False

    def set_base_name(self, base_name):
        self.base_name = base_name
        self.not_found_points = list(self.get_mitos_points(base_name))
        self.verificated_mitos = 0

    def get_mitos_count(self):
        return len(self.jsonDict[self.base_name])

    def get_mitos_points(self, base_name):
        return self.jsonDict[base_name]

    def print_verification_result(self):
        mitos_count = self.get_mitos_count()
        # if self.verificated_mitos < mitos_count:
        if len(self.not_found_points) > 0:
            fill = ' '
            clean_text = fill * 90
            # for some reason when i call this function from the main program, carriage return(\r)
            # doesnt clean the line, so i 'clean' it manually by printing spaces
            sys.stdout.write('\r' + clean_text + '\r')
            print('\r{} {}/{}'.format(self.base_name,
                                      self.verificated_mitos,
                                      mitos_count))


class Testing_candidate:
    def __init__(self, im, pos, label, base_im_name):
        self.im = im
        self.pos = pos
        self.label = label
        self.base_im_name = base_im_name
        self.predicted_label = -1

    def __str__(self):
        return 'pos:{},{} | {} | true:{} pred:{}'.format(self.pos[0],
                                                         self.pos[1],
                                                         self.base_im_name,
                                                         self.label,
                                                         self.predicted_label)


class Mitos_test_evaluator:
    def __init__(self, json_dict, HPF_dirpath=P().normHeStainDir):
        self.json_dict = json_dict
        self.verificator = MitosVerification()
        self.HPF_dirpath = HPF_dirpath
        self.not_detected = 0
        self.not_detected_points = {}
        self.testing_candidates_list = []
        self._pos = 0
        self._labels = []
        self._predicted_labels = []
        self._map_to_testing_candidates()

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= len(self.testing_candidates_list):
            raise StopIteration
        else:
            im = [self.testing_candidates_list[self._pos].im]
            self._pos += 1
            return np.asarray(im)

    def _map_to_testing_candidates(self):
        for base_name in sorted(self.json_dict):
            point_list = self.json_dict[base_name]
            self.verificator.set_base_name(base_name)
            im_path = self.HPF_dirpath + base_name + '.bmp'
            im = cv2.imread(im_path)
            cand_list = self._extract_test_candidate(im, point_list, base_name)
            self.testing_candidates_list.extend(cand_list)
            not_detected = len(self.verificator.not_found_points)
            self.not_detected += not_detected

            if not_detected > 0:
                self.not_detected_points[base_name] = self.verificator.not_found_points

    def _extract_test_candidate(self, im, point_list, base_name):
        from mitos_extract_anotations.ImCutter import No_save_ImCutter

        imcutter = No_save_ImCutter(im)
        test_candidate_list = []

        save_dir = P().saveTestCandidates
        sufix_num = 0

        for p in point_list:
            point = (p['row'], p['col'])
            candidate_im = imcutter.cut(point[1], point[0])
            # returns true (1) if point is close to a mitotic cell,
            # but in the model, label 0 stands for mitosis,
            # so we need to negate it
            candidate_im = self.normalize(candidate_im)
            label = int(not self.verificator.is_mitos(point))
            candidate = Testing_candidate(im=candidate_im,
                                          pos=point,
                                          label=label,
                                          base_im_name=base_name)

            test_candidate_list.append(candidate)

            # if label == 1:
            #     save_path = '{}{}-{}.png'.format(save_dir, base_name, sufix_num)
            # else:
            #     save_path = '{}{}-{}-mitosis.png'.format(save_dir, base_name, sufix_num)
            # cv2.imwrite(save_path, candidate_im)
            # sufix_num += 1

        return test_candidate_list

    def add_prediction(self, pred_label):
        self._predicted_labels.append(pred_label)
        self._labels.append(self.testing_candidates_list[self._pos - 1].label)
        self.testing_candidates_list[self._pos - 1].predicted_label = pred_label

    def evaluate(self):
        if len(self._predicted_labels) == 0:
            raise ValueError('No predicted labels available')

        from mitosCalsification import metrics
        fscore = metrics.fscore(self._labels, self._predicted_labels, self.not_detected)
        print('fscore: {}'.format(fscore))

    def print_res_to_img(self):
        im = cv2.imread('C:/Users/felipe/mitos dataset/normalizado/A04_02.bmp')
        base_name = 'A04_02'
        i = 0
        while base_name == 'A04_02':
            candidate = self.testing_candidates_list[i]
            i += 1
            base_name = candidate.base_im_name
            pos = candidate.pos
            prediction = candidate.predicted_label
            label = candidate.label
            if label == 0:
                if prediction == 0:
                    color = (255, 0, 0)  # blue color
                else:
                    color = (0, 255, 0)  # green color

                cv2.circle(im, pos, 25, color, thickness=2)
            elif prediction == 0 and label == 1:
                cv2.circle(im, pos, 25, (0, 0, 255), thickness=2)

        base_dir = P().basedir
        save_path = base_dir + 'test/print/A04_02.jpg'
        cv2.imwrite(save_path, im)

    def print_conf_matrix(self):
        if len(self._predicted_labels) == 0:
            raise ValueError('No predicted labels available')

        from mitosCalsification.metrics import print_conf_matrix
        print_conf_matrix(self._labels, self._predicted_labels)

    def get_candidates(self):
        candidates = []
        for cand in self.testing_candidates_list:
            candidates.append(cand.im)

        return np.asarray(candidates)

    def normalize(self, im):
        im = np.asarray(im, np.float32)
        im /= 255
        return im


if __name__ == "__main__":
    from common.Params import Params as P

    test_json_path = P().basedir + 'anotations/test_cand.json'
    with open(test_json_path) as file:
        json_string = file.read()
        cand_dict = json.loads(json_string)
    mte = Mitos_test_evaluator(cand_dict, P().basedir + 'normalizado/heStain/')
    for c in mte:
        j = 0
    print(mte.not_detected)
    print(mte.not_detected_points)
    i = 0

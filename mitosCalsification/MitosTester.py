import cv2
import numpy as np
import cv2
import json

from common.utils import MitosVerification, Testing_candidate
from common.Params import Params as P
from mitosCalsification.loadDataset import extract_anotation_single_image


class MitosTester:
    def __init__(self, anotations_dict, model):
        self.anotations_dict = anotations_dict
        self.verificator = MitosVerification()
        self.HPF_dirpath = P().basedir + 'normalizado/testHeStain/'
        self.not_detected = 0
        self.not_detected_points = {}
        self.testing_candidates_list = []
        self.model = model
        self.is_dataset_loaded = False
        self.predicted_labels = []
        self.base_name_list = []

    def evaluate_all(self):
        if self.is_dataset_loaded == False:
            self._load_dataset()

        xt, yt = self._get_testing_data()
        y_pred = self.model.predict(xt)
        y_pred = np.round(y_pred, decimals=0).astype(int)
        self._print_metrics(yt, y_pred)
        self.print_detections_images(y_pred, self.base_name_list)

    def _get_testing_data(self):
        xt = []
        yt = []
        for candidate in self.testing_candidates_list:
            xt.append(candidate.im)
            yt.append(candidate.label)

        return np.asarray(xt), np.asarray(yt)

    def _print_metrics(self, y_true, y_pred):
        from mitosCalsification import metrics
        metrics.print_conf_matrix(y_true, y_pred, self.not_detected)
        fscore = metrics.fscore(y_true, y_pred, self.not_detected)
        print('fscore: {}'.format(fscore))

    def _load_dataset(self):
        for base_name in sorted(self.anotations_dict):
            self.base_name_list.append(base_name)
            point_list = self.anotations_dict[base_name]
            self.verificator.set_base_name(base_name)
            im_path = self.HPF_dirpath + base_name + '.bmp'
            im = cv2.imread(im_path)
            if im is None:
                raise FileNotFoundError('The image {} anotated as testing candidates is not'
                                        ' found in the directory {}'.format(base_name,
                                                                            self.HPF_dirpath))

            im_list = extract_anotation_single_image(im,
                                                     point_list,
                                                     invert_dim=True)

            self._map_extracted_to_candidate_class(im_list, point_list, base_name)
            not_detected = len(self.verificator.not_found_points)
            self.not_detected += not_detected

            if not_detected > 0:
                self.not_detected_points[base_name] = self.verificator.not_found_points

    def _map_extracted_to_candidate_class(self, im_list, point_list, base_name):
        for cand_im, point in zip(im_list, point_list):
            p = (point['row'], point['col'])
            label = int(not self.verificator.is_mitos(p))

            candidate = Testing_candidate(im=cand_im,
                                          pos=point,
                                          label=label,
                                          base_im_name=base_name)

            self.testing_candidates_list.append(candidate)

    def print_detections_images(self, y_pred, base_name_list):
        mitosis_anotations_path = P().mitosAnotationJsonPath
        with open(mitosis_anotations_path) as file:
            json_string = file.read()
            mitosis_anotations = json.loads(json_string)

        detection_dict = self._map_detections_to_dict(y_pred)

        for base_name in base_name_list:
            image_path = P().basedir + 'normalizado/testHeStain/' + base_name +'.bmp'
            image = cv2.imread(image_path)
            self._print_mitosis(image, mitosis_anotations[base_name])
            save_im_path = P().basedir + 'resultado/' + base_name + '.png'

            detection_list = detection_dict[base_name]
            self._print_detected(image, detection_list)

            cv2.imwrite(save_im_path, image)

    def _print_mitosis(self, im, mitosis_list):
        lightblue_color = (204, 122, 0)
        for mitosis in mitosis_list:
            point = (mitosis['col'], mitosis['row'])
            cv2.circle(im, point, 30, lightblue_color, thickness=4)

    def _print_detected(self,im, point_list):
        green_color = (0,255,0)
        for point in point_list:
            p = (point['row'], point['col'])
            cv2.circle(im, p, 20, green_color, thickness=2)

    def _map_detections_to_dict(self, y_pred):
        detection_dict = {}
        for pred, candidate in zip(y_pred, self.testing_candidates_list):
            if candidate.base_im_name not in detection_dict:
                detection_dict[candidate.base_im_name] = []

            if pred == 0:
                detection_dict[candidate.base_im_name].append(candidate.pos)

        return detection_dict



if __name__ == '__main__':
    from common.utils import listFiles

    json_path = P().basedir + 'anotations/test_cand.json'
    with open(json_path) as file:
        string = file.read()
        cand_dict = json.loads(string)

    tester = MitosTester(cand_dict, None)
    tester._load_dataset()
    not_detected = tester.not_detected
    print(not_detected)

    i= 0
    for cand in tester.testing_candidates_list:
        if cand.label == 0:
            i += 1

    print(i)

import json
import os

import cv2
from sklearn.model_selection import KFold
import numpy as np

from common.utils import listFiles, print_progress_bar
from PyQt5.QtCore import QFileInfo, QDir
from common.Params import Params as P
from mitosCalsification.Preprocess import preprocess_in_memory
from mitosCalsification.loadDataset import extract_anotations
from mitosCalsification.mitosClasificator import MitosisClasificatorTrainer
from mitos_extract_anotations.ImCutter import No_save_ImCutter
from models.SimpleModel import create_simple_model


def _load_json_data():
    # loads the anotations and combines them in one dictionary
    with open(P().candidatesTrainingJsonPath) as f:
        json_string = f.read()
        candidates_dict = json.loads(json_string)

    with open(P().candidatesTestJsonPath) as f:
        json_string = f.read()
        test_candidates_dict = json.loads(json_string)

    candidates_dict.update(test_candidates_dict)

    with open(P().mitosAnotationJsonPath) as f:
        json_string = f.read()
        mitosis_anotations = json.loads(json_string)

    return mitosis_anotations, candidates_dict


def _fileInfo_to_baseName(fileInfo_list):
    base_name_list = []
    for f in fileInfo_list:
        base_name_list.append(f.baseName())

    return base_name_list


def save_im(baseName, im_list):
    i = 0
    for im in im_list:
        path = 'C:/Users/felipe/Desktop/New folder/{}_{}.jpg'.format(baseName, i)
        cv2.imwrite(path, im)
        i += 1


def save_im_list(path):
    pass


def crossval(n_fold=10):
    if os.path.exists('res.txt'):
        os.remove('res.txt')

    # create a list of all High Power Field images path
    filters = ['*.bmp', '*.png', '*.jpg']
    train_file_list = listFiles(P().basedir + 'normalizado/heStain/', filters)
    test_file_list = listFiles(P().basedir + 'normalizado/testHeStain', filters)
    # train_file_list = listFiles(P().basedir + 'normalizado/fullStainTrain/', filters)
    # test_file_list = listFiles(P().basedir + 'normalizado/fullStainTest', filters)
    file_list = train_file_list
    file_list.extend(test_file_list)
    file_list = np.asarray(file_list)

    mitosis_anotations, candidates_anotations = _load_json_data()

    i = 1
    kfold = KFold(n_splits=n_fold, shuffle=True)
    for train_index, test_index in kfold.split(file_list):
        print('iteración: {}/{}'.format(i, n_fold))
        i += 1

        train_im_fileInfo = file_list[train_index]
        test_im_fileInfo = file_list[test_index]

        x_train, y_train, x_test, y_test = extract_anotations(train_im_fileInfo,
                                                              test_im_fileInfo,
                                                              mitosis_anotations,
                                                              candidates_anotations)

        model = create_simple_model()
        trainer = MitosisClasificatorTrainer(model,
                                             (x_train, y_train),
                                             (x_test, y_test),
                                             epochs=40)

        trainer.train()
        best_score = trainer.best_score
        with open('res.txt', 'a') as file:
            file.write('{}\n'.format(best_score))


if __name__ == '__main__':
    crossval()

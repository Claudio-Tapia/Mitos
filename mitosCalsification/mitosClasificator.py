import sys
import time

sys.path.append('C:/Users/PelaoT/Desktop/Practica/codigo')

import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import model_from_json, model_from_config, Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from common.Params import Params as P
from common.utils import listFiles, keras_deep_copy_model
from common.utils import write_test_output
from mitosCalsification import metrics, loadDataset as ld
from mitosCalsification.End_training_callback import End_training_callback
from mitosCalsification.MitosTester import MitosTester
from mitosCalsification.plot import print_plots, dump_metrics_2_file, plot_roc, plot_precision_recall

#from doqu.Bagging import Bagging
#from models.Bagging import Bagging
from models.SimpleModel import create_fel_res, create_simple_model, create_simple2
import json

from models.SqueezeNet import create_squeeze_net


def save_model(model, name):
    json_string = model.get_config()
    json_string = json.dumps(json_string)
    file = open('./saved_models/' + name + '.json', 'w')
    file.write(json_string)
    file.close()
    model.save_weights('./saved_models/' + name + '_weights.h5')


def load_model(name):
    file = open('./saved_models/' + name + '.json')
    json_string = file.read()
    file.close()
    model_config = json.loads(json_string)
    if isinstance(model_config, dict):
        model = Model.from_config(model_config)
    else:
        model = Sequential.from_config(model_config)
    model.load_weights('./saved_models/' + name + '_weights.h5')
    sgd = SGD(momentum=0.02, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=[metrics.mitos_fscore])
    return model


def save_bagging_model(bagging):
    i = 1
    for estimator in bagging.estimators:
        model_base_name = 'model' + str(i)
        save_model(estimator, model_base_name)
        i += 1


def load_bagging_model():
    filter = ['model*.json']
    info_list = listFiles('./saved_models/', filter)
    if len(info_list) == 0:
        raise FileNotFoundError('There is no model saved')

    estimators = []
    for file_info in info_list:
        model = load_model(file_info.baseName())
        estimators.append(model)

    bag = Bagging()
    bag.set_estimator(estimators)
    return bag


def load_test_data():
    if sys.platform == 'win32':
        cand_path = 'D:/dataset/test/no-mitosis/candidates.tar'
        mit_path = 'D:/dataset/test/mitosis/'
    else:
        cand_path = '/home/facosta/dataset/test/no-mitosis/candidates.tar'
        mit_path = '/home/facosta/dataset/test/mitosis/'

    # test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
    test = ld.dataset(cand_path, mit_path)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    yt_cat = np_utils.to_categorical(yt)

    return xt, yt


def _get_class_weights(labels):
    import math
    weight_dict = {}
    total = len(labels)
    unique = np.unique(labels, return_counts=True)
    i = 0
    while i < len(unique):
        classes_count = unique[1]
        class_count = classes_count[i]
        weight = math.log(total / class_count, 1.7)  # base = 1.7
        if weight < 1:
            weight = 1
        weight_dict[i] = weight
        i += 1

    return weight_dict


class MitosisClasificatorTrainer:
    def __init__(self, model,
                 train_data,
                 test_data,
                 val_data=None,
                 epochs=40,
                 batch_size=128):

        self.best_model = None
        self.best_score = 0.0
        self.best_test_pred = None
        self.train_history_list = []
        self.val_history_list = []
        self.test_history_list = []

        self.epochs = epochs
        self.batch_size = batch_size
        self.generator = ImageDataGenerator()

        self.model = model
        self.xe = train_data[0]
        self.ye = train_data[1]

        self.xt = test_data[0]
        self.yt = test_data[1]
        self.iteration_test_fscore = 0.0
        self.iteration_test_prec = 0.0
        self.best_test_pred = None

        if val_data is not None:
            self.bval_data = True
            self.xv = val_data[0]
            self.yv = val_data[1]
            self.val_fscore = None
            self.val_loss = None
        else:
            self.bval_data = False

    def plot_metrics_to_disk(self):
        if self.bval_data:
            val_history = np.transpose(self.val_history_list)
        else:
            val_history = None
        print_plots(self.model.metrics_names,
                    np.transpose(self.train_history_list),
                    val_history,
                    self.test_history_list)

    def train(self):
        class_weight = _get_class_weights(self.ye)

        for e in range(self.epochs):
            print('Epoch: {}/{}'.format(e + 1, self.epochs))

            start_time = time.time()
            self._shuffle_epoch()
            self._train_epoch(class_weight)
            self._validate()
            self._test()
            end_time = time.time()

            self._print_epoch_sumary(end_time - start_time)

        save_model(self.best_model, name='b_model')
        dump_metrics_2_file(train_metrics=self.train_history_list,
                            val_metrics=None,
                            test_metrics=self.test_history_list)

    def _train_epoch(self, class_weight):
        batches = 0
        history_list = []
        for x_batch, y_batch in self.generator.flow(self.xe, self.ye, self.batch_size):
            history = self.model.train_on_batch(x_batch,
                                                y_batch,
                                                class_weight=class_weight)
            history_list.append(history)
            batches += 1
            if batches >= int(len(self.xe) / self.batch_size):
                break

        history = np.asarray(history_list).mean(axis=0)
        self.train_history_list.append(history)

    def _shuffle_epoch(self):
        idx = np.arange(len(self.ye))
        np.random.permutation(idx)
        self.xe = self.xe[idx]
        self.ye = self.ye[idx]

    def _validate(self):
        if not self.bval_data:
            return

        val_pred = self.model.predict(self.xv, self.batch_size)
        val_pred = np.amax(val_pred, axis=1)
        val_loss = binary_crossentropy(K.variable(self.yv),
                                       K.variable(val_pred))
        self.val_loss = K.eval(K.mean(val_loss))
        val_pred = np.round(val_pred, decimals=0).astype(int)
        self.val_fscore = metrics.fscore(self.yv, val_pred)
        self.val_history_list.append(self.val_fscore)

    def _test(self, model=None, return_metrics=False):
        if model is None:
            model = self.model

        test_pred = model.predict(self.xt)
        round_test_pred = np.round(test_pred, decimals=0).astype(int)

        test_fscore = metrics.fscore(self.yt, round_test_pred)
        test_prec = K.eval(K.mean(binary_accuracy(K.variable(self.yt),
                                                  K.variable(round_test_pred))))

        if test_fscore > self.best_score:
            self.best_score = test_fscore
            self.best_model = keras_deep_copy_model(model)

        if return_metrics == False:
            self.iteration_test_fscore = test_fscore
            self.iteration_test_prec = test_prec
            self.test_history_list.append(test_fscore)
        else:
            return test_fscore, test_prec

    def _print_epoch_sumary(self, time_delta):
        print('time: {:.1f}'.format(time_delta), end=' - ')

        train_metrics_names = self.model.metrics_names
        train_metrics_values = self.train_history_list[len(self.train_history_list) - 1]
        for name, value in zip(train_metrics_names, train_metrics_values):
            print(name + ': {:.4f}'.format(value), end=' ', flush=False)

        if self.bval_data:
            print('val_loss: {:.4f} val_mitos_fscore: {:.4f}'.
                  format(self.val_loss, self.val_fscore), end=' ', flush=False)

        print('test_fscore: {:.4f}'.format(self.iteration_test_fscore), flush=True)


def train_model(ratio, use_all):
    selection = True
    if use_all:
        selection = False
    elif ratio <= 0:
        raise ValueError('ratio cannot be neither negative nor 0')
    train = ld.dataset(P().saveCutCandidatesDir + 'candidates.tar', P().saveMitosisPreProcessed)
    xe, ye = train.get_training_sample(ratio=ratio, selection=selection)
    xt, yt = load_test_data()

    # from mitosCalsification.crossval import _load_json_data
    # filters = ['*.bmp', '*.png', '*.jpg']
    # train_file_info = listFiles(P().basedir + 'normalizado/fullStainTrain', filters)
    # test_file_info = listFiles(P().basedir + 'normalizado/fullStainTest', filters)
    # mitosis_anotations, candidates_anotations = _load_json_data()
    # from mitosCalsification.loadDataset import extract_anotations
    # xe, ye, xt, yt = extract_anotations(train_file_info,
    #                                     test_file_info,
    #                                     mitosis_anotations,
    #                                     candidates_anotations)

    # model = create_fel_res()
    model = create_simple_model()
    # model = create_squeeze_net()
    # model = create_simple2()
    # model = load_model('b_model') # best model previusly trained

    clasificator = MitosisClasificatorTrainer(model,
                                              (xe, ye),
                                              (xt, yt),
                                              epochs=100,
                                              batch_size=128)

    generator = ImageDataGenerator(rotation_range=44,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.21,
                                   zoom_range=0.3,
                                   fill_mode='wrap',
                                   horizontal_flip=True,
                                   vertical_flip=True)

    # clasificator.generator = generator
    clasificator.train()
    clasificator.plot_metrics_to_disk()

    save_model(model, 'model1')
    K.clear_session()


def _do_test(model, xt, yt):
    if isinstance(model, Bagging):
        yt2 = np.argmax(yt, axis=1)
        res_rounded = model.predict_on_batch(xt, yt2)
        res = res_rounded
    else:
        res = model.predict(xt)
        # res = np.argmax(res, axis=1)
        res_rounded = np.round(res, decimals=0).astype(int)

    cat_res = np_utils.to_categorical(res_rounded)
    fscore = metrics.fscore(yt, res_rounded)
    prec = K.eval(K.mean(binary_accuracy(K.variable(yt),
                                         K.variable(res_rounded))))

    return fscore, prec, res_rounded, res


def test_model():
    import time
    model = load_model('b_model')
    # xt, yt = load_test_data()
    #
    # fscore, prec, res_round, res = _do_test(model, xt, yt)
    # # idx = res < 0.2
    # # res_round = np.copy(res)
    # # res_round[idx] = 0
    # # res_round[np.logical_not(idx)] = 1
    #
    # metrics.print_conf_matrix(yt, res_round)
    # print('fscore: {}'.format(fscore))
    # print('precision: {}'.format(prec))
    # write_test_output(yt, res)
    # plot_roc(yt, res)
    # plot_precision_recall(yt, res)
    #
    test_json_path = P().candidatesTestJsonPath
    with open(test_json_path) as file:
        json_string = file.read()
        cand_dict = json.loads(json_string)

    tester = MitosTester(cand_dict, model)
    t0 = time.time()
    tester.evaluate_all()

    t1 = time.time()
    print(t1 - t0)
    K.clear_session()

if __name__ == '__main__':
    test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    yt_cat = np_utils.to_categorical(yt)

    model = load_model('model3')
    res = model.predict_classes(xt)

    cat_res = np_utils.to_categorical(res)
    fscore = K.eval(metrics.mitos_fscore(K.variable(yt_cat),
                                         K.variable(cat_res)))

    metrics.print_conf_matrix(yt, res)
    print('fscore: {}'.format(fscore))
    write_test_output(yt, res)

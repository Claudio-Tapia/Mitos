from keras import backend as K
from keras import layers
from keras.engine.training import Model
from keras.layers import Conv2D, Input
from keras.layers.core import Flatten, Dense
from keras.layers.pooling import MaxPooling2D

from common.utils import getInputDim

module_count = 1

def create_fire_mod(prev_layer, squeeze_filters_num, expand_filters_num):
    if K.image_data_format() == 'channels_last':
        merge_channel = 3
    else:
        merge_channel = 1

    global module_count
    x = Conv2D(squeeze_filters_num,
               kernel_size=(1,1),
               padding='same',
               activation='relu',
               name='fire{}_squeeze'.format(module_count))(prev_layer)

    exp1 = Conv2D(expand_filters_num,
                  kernel_size=(1,1),
                  padding='same',
                  activation='relu',
                  name='fire{}_expand_1x1'.format(module_count))(x)

    exp3 = Conv2D(expand_filters_num,
                  kernel_size=(3,3),
                  padding='same',
                  activation='relu',
                  name='fire{}_expand_3x3'.format(module_count))(x)

    conc = layers.concatenate([exp1, exp3], axis=merge_channel,
                              name='fire{}_concatenate'.format(module_count))

    module_count += 1
    return conc

def create_squeeze_net():
    inp = Input(shape=getInputDim())
    x = Conv2D(64, (3,3), padding='same', activation='relu')(inp)
    x = MaxPooling2D((3,3), strides=2)(x)
    x = create_fire_mod(x, 64, 128)
    x = MaxPooling2D((3,3), strides=2)(x)
    x = create_fire_mod(x, 64, 128)
    x = MaxPooling2D((3,3), strides=2)(x)

    x = Conv2D(32,(1,1), activation='relu')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=['binary_accuracy',])

    return model

if __name__ == '__main__':
    model = create_squeeze_net()
    model.summary()
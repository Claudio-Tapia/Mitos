from keras.callbacks import Callback
import sys

class End_training_callback(Callback):
    def __init__(self, min_train_loss=0.1, min_val_loss=0.04):
        super().__init__()
        self.min_train_loss = min_train_loss
        self.min_val_loss = min_val_loss
        self.prev_train_loss = 1000.0
        self.delta = 0.01
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if train_loss <= self.min_train_loss and val_loss <= self.min_val_loss:
            self.model.stop_training = True
            self.stopped_epoch = epoch

        loss_delta = self.prev_train_loss - train_loss
        if abs(loss_delta)< self.delta and train_loss > 2.0:
            print('error loss:{}'.format(train_loss))
            sys.exit(-1) # a veces no entrena la red,con un error muy alto, ni idea por que


        self.prev_train_loss = train_loss

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Training stopped on epoch {}'.format(self.stopped_epoch))

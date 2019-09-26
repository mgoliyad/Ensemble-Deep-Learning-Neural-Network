from keras.callbacks import Callback
from keras import backend 
from math import pi
from math import cos
from math import floor

'''class my_Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.39):
    #if(logs.get('loss')<0.4):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True'''

class SnapshotEnsemble(Callback):
    def __init__(self, configs, val_vecs, predictions):
        self.nb_epochs = configs['training']['nb_epochs']
        self.nb_cycles = configs['training']['nb_cycles']
        self.lr_max = configs['training']['lrate_max']
        self.lrates = list()
        self.predictions = predictions
        self.val_vecs = val_vecs
 
# calculate learning rate for epoch
    def cosine_annealing(self, epoch):
        epochs_per_cycle = floor(self.nb_epochs/self.nb_cycles)
        assert epochs_per_cycle != 0,"Please check nb_epochs and nb_cycles values in config."
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return self.lr_max/2 * (cos(cos_inner) + 1)
 
# calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch)
        backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)
 
# make prediction at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        assert self.nb_cycles != 0,"Please check nb_cycles value in config."
        epochs_per_cycle = floor(self.nb_epochs / self.nb_cycles)
        assert epochs_per_cycle != 0,"Please check nb_epochs and nb_cycles values in config."
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            prediction = self.model.predict(self.val_vecs)
            self.predictions.append(prediction)

        


import tensorflow as tf
import os
import pdb
import glob

class CustomCheckpointREAL(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        job_id = self.filepath.split('_epoch')[0]
        self.filepath = f'{job_id}_epoch{epoch}.h5'
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq == 'epoch':
            job_id = self.filepath.split('_epoch')[0]
            for f in glob.glob(f'{job_id}_epoch*'):                
                os.remove(f)
            self._save_model(epoch=epoch, batch=None, logs=logs)

class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        job_id = self.filepath.split('_epoch')[0]
        self.filepath = f'{job_id}_epoch{epoch}.h5'
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq == 'epoch':
            job_id = self.filepath.split('_epoch')[0]
            for f in glob.glob(f'{job_id}_epoch*'):                
                os.remove(f)
            with open(f'{job_id}_epoch{epoch}.h5', 'w') as f:
                pass



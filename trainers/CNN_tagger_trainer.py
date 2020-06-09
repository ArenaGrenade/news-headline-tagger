from base.abstract_trainer import AbstractTrainer
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


class CNNModelTrainer(AbstractTrainer):
    def __init__(self, model, data, labels, config):
        super(CNNModelTrainer, self).__init__(model, data, labels, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_freq=5
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
                histogram_freq=1
            )
        )

        self.callbacks.append(
            EarlyStopping(
                'val_sparse_categorical_accuracy',
                patience=5
            )
        )

    def train(self):
        history = self.model.fit(
            self.data, self.labels,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks,
            validation_split=0.2,
            shuffle=True
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])

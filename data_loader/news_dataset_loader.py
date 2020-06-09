from base.abstract_data_loader import AbstractDataLoader
from pre_processing.train_embedder import *


class NewsDatasetLoader(AbstractDataLoader):
    def __init__(self, config):
        super(NewsDatasetLoader, self).__init__(config)
        print("processing data")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = getProcessedData(self.config)
#        print("creating and training a model")
#        trainEmbedder(self.X_test + self.X_train, 100, "model_trial_1")
#        print("created model and saved")

    def getTrainingData(self):
        return self.X_train, self.y_train

    def getTestData(self):
        return self.X_test, self.y_test

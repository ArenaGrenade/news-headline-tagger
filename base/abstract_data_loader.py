class AbstractDataLoader(object):
    def __init__(self, config):
        self.config = config

    def getTrainingData(self):
        """
        This is to be overridden when preparing a model by extending this class. This Abstract function only raises
        an NotImplementedError if it is not overridden in the model created.
        :return:
        """
        raise NotImplementedError

    def getTestData(self):
        """
        This is to be overridden when preparing a model by extending this class. This Abstract function only raises
        an NotImplementedError if it is not overridden in the model created.
        :return:
        """
        raise NotImplementedError

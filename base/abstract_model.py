class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    def saveModel(self, checkpoint_path):
        """
        This function is used to save a checkpoint in the location specified in the config file.
        :param checkpoint_path: path to the configuration
        :return:
        """
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def loadModel(self, checkpoint_path):
        """
        This is the getter version of the save function
        :param checkpoint_path:
        :return:
        """
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def buildModel(self):
        """
        This is to be overridden when preparing a model by extending this class. This Abstract function only raises
        an NotImplementedError if it is not overridden in the model created.
        :return:
        """
        raise NotImplementedError

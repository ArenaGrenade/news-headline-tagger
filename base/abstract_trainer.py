class AbstractTrainer(object):
    def __init__(self, model, data, labels, config):
        self.model = model
        self.data = data
        self.labels = labels
        self.config = config

    def train(self):
        raise NotImplementedError

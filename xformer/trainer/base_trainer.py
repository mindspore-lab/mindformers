class BaseTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        print("Now Running Model is: {}".format(model_name))

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

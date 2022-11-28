"""Base Trainer."""


class BaseTrainer:
    """Base Trainer."""
    def __init__(self, model_name):
        self.model_name = model_name
        print("Now Running Model is: {}".format(model_name))

    def train(self):
        """train function for Trainer."""
        raise NotImplementedError()

    def evaluate(self):
        """evaluate function for Trainer."""
        raise NotImplementedError()

    def predict(self):
        """predict function for Trainer."""
        raise NotImplementedError()

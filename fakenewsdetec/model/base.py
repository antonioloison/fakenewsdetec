from abc import ABC
from abc import abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self):
        """
        Performs training of model. The exact train implementations are model specific.
        :param train_datapoints: List of train datapoints
        :param val_datapoints: List of validation datapoints that can be used
        :param cache_featurizer: Whether or not to cache the model featurizer
        :return:
        """
        pass
   
    @abstractmethod
    def predict(self, test_data):
        """
        Performs inference of model on collection of datapoints. Returns an
        array of model predictions. This should only be called after the model
        has been trained.
        :param datapoints: List of datapoints to perform inference on
        :return: Array of predictions
        """
        pass
    
    @abstractmethod
    def compute_metrics(self):
        """
        Compute a set of model-specifc metrics on the provided set of datapoints.
        :param eval_datapoints: Datapoints to compute metrics for
        :param split: Data split on which metrics are being computed
        :return: A dictionary mapping from the name of the metric to its value
        """
        pass

from data_processor import CDataProcessor
from models import CNNModel


class CModelTrainer:
    def __init__(self, model_name, dataset, table, credentials):
        self.processor = CDataProcessor(credentials, dataset, table)
        self.model = CNNModel(model_name=model_name)

    def train_model(self, plot_learning_curve=False):
        #
        # get, cleanup and transform data for training
        self.processor.read_data()
        x_train, y_train, x_val, y_val = self.processor.prepare_data()
        self.model.info['columns'] = self.processor.predictable_columns()
        #
        # construct and  fit NN model
        self.model.construct(x_train.shape[1], 1)
        self.model.fit(x_train, y_train, x_val, y_val, batch_size=512)
        if plot_learning_curve:
            self.plot()

    def plot(self):
        self.model.plot_learning_curve()

    def save_model(self):
        self.model.save()

from trainer import CModelTrainer
from argparse import ArgumentParser


class CApplication:
    def __init__(self):
        self.trainer = None
        self.arg_parser = ArgumentParser('Python based tool to analyze London house prices and create NN model based on data.')
        self.arg_parser.add_argument('-cred',
                                     help="Path to google cloud credentials file.",
                                     dest='cred',
                                     required=True,
                                     type=str
                                     )
        self.arg_parser.add_argument('-d',
                                     help="Dataset name on google cloud.",
                                     dest='dataset',
                                     required=True,
                                     type=str
                                     )
        self.arg_parser.add_argument('-t',
                                     help="Dataset table name on google cloud.",
                                     dest='table',
                                     required=True,
                                     type=str
                                     )
        self.arg_parser.add_argument('-n',
                                     help="Trained model name for serializing.",
                                     dest='name',
                                     required=True,
                                     type=str
                                     )
        self.arg_parser.add_argument('-plot',
                                     help="Plot trained models learning curve.",
                                     dest='plot',
                                     action='store_true',
                                     required=False,
                                     default=False
                                     )

    def run(self):
        args = self.arg_parser.parse_args()
        self.trainer = CModelTrainer(args.name, args.dataset, args.table, args.cred)
        self.trainer.train_model()
        self.trainer.save_model()
        if args.plot:
            self.trainer.plot()
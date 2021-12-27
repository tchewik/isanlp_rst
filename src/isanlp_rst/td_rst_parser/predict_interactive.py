import os
import pickle

from .src import PointingDiscourseGoldsegmentationEduRepParser


class TrainedPredictor:
    def __init__(self, model_path):
        self.parser = PointingDiscourseGoldsegmentationEduRepParser.load(model_path)
        self._tempdir = './tmp'
        if not os.path.isdir(self._tempdir):
            os.mkdir(self._tempdir)
        self._input_data_path = os.path.join(self._tempdir, 'pred_examples')
        self._output_data_path = os.path.join(self._tempdir, 'predictions')
        self.parser.args.update({'predict_output_path': self._output_data_path})

    def predict(self, examples):
        with open(self._input_data_path, 'wb') as f:
            pickle.dump(examples, f)

        self.parser.predict(data=self._input_data_path)
        with open(self._output_data_path, 'rb') as f:
            pred = pickle.load(f)

        os.remove(self._input_data_path)
        os.remove(self._output_data_path)

        return pred

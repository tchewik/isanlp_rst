import os

import numpy as np
from allennlp.predictors import Predictor
from isanlp.annotation_rst import DiscourseUnit
from symbol_map import SYMBOL_MAP


class AllenNLPSegmenter:

    def __init__(self, model_dir_path, cuda_device=-1):
        self._model_path = os.path.join(model_dir_path, 'segmenter_neural', 'model.tar.gz')
        self._cuda_device = cuda_device
        self.predictor = Predictor.from_path(self._model_path, cuda_device=self._cuda_device)
        self._separator = 'U-S'
        self._symbol_map = SYMBOL_MAP

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_postag, annot_synt_dep_tree,
                 start_id=0):
        return self._build_discourse_units(annot_text, annot_tokens,
                                           self._predict(annot_tokens, annot_sentences), start_id)

    def _predict(self, tokens, sentences):
        """
        :return: numbers of tokens predicted as EDU left boundaries
        """
        _sentences = []
        for sentence in sentences:
            text = ' '.join([self._prepare_token(token.text) for token in tokens[sentence.begin:sentence.end]]).strip()
            if text:
                _sentences.append(text)

        predictions = self.predictor.predict_batch_json([{'sentence': sentence} for sentence in _sentences])
        result = []
        for i, prediction in enumerate(predictions):
            pred = np.array(prediction['tags'][:sentences[i].end - sentences[i].begin]) == self._separator

            # The first token in a sentence is a separator
            # if it is not a point in a list
            if len(pred) > 0:
                if i > 0:
                    if predictions[i - 1]['words'][1] == '.' and predictions[i - 1]['words'][0] in "0123456789":
                        pred[0] = False
                else:
                    pred[0] = True

            # No single-token EDUs
            for j, token in enumerate(pred[:-1]):
                if token and pred[j + 1]:
                    if j == 0:
                        pred[j + 1] = False
                    else:
                        pred[j] = False

            result += list(pred)

        return np.argwhere(np.array(result) == True)[:, 0]

    def _build_discourse_units(self, text, tokens, numbers, start_id):
        """
        :param text: original text
        :param list tokens: isanlp.annotation.Token
        :param numbers: positions of tokens predicted as EDU left boundaries (beginners)
        :return: list of DiscourseUnit
        """

        edus = []

        if numbers.shape[0]:
            for i in range(0, len(numbers) - 1):
                new_edu = DiscourseUnit(start_id + i,
                                        start=tokens[numbers[i]].begin,
                                        end=tokens[numbers[i + 1]].begin - 1,
                                        text=text[tokens[numbers[i]].begin:tokens[numbers[i + 1]].begin],
                                        relation='elementary',
                                        nuclearity='_')
                edus.append(new_edu)

            if numbers.shape[0] == 1:
                i = -1

            new_edu = DiscourseUnit(start_id + i + 1,
                                    start=tokens[numbers[-1]].begin,
                                    end=tokens[-1].end,
                                    text=text[tokens[numbers[-1]].begin:tokens[-1].end],
                                    relation='elementary',
                                    nuclearity='_')
            edus.append(new_edu)

        return edus

    def _prepare_token(self, token):

        for key, value in self.symbol_map.items():
            token = token.replace(key, value)

        for keyword in ['www', 'http']:
            if keyword in token:
                return '_html_'
        return token

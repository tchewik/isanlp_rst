import os

import numpy as np
from allennlp.predictors import Predictor
from isanlp.annotation_rst import DiscourseUnit


class AllenNLPSegmentator:

    def __init__(self, model_dir_path):
        self._model_path = os.path.join(model_dir_path, 'tony_segmentator', 'model.tar.gz')
        self.predictor = Predictor.from_path(self._model_path)
        self._separator = 'U-S'

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_postag, annot_synt_dep_tree, start_id=0):
        return self._build_discourse_units(annot_text, annot_tokens,
                                           self._predict(annot_tokens, annot_sentences), start_id)

    def _predict(self, tokens, sentences):
        """
        :return: numbers of tokens predicted as EDU left boundaries
        """
        result = []
        for sentence in sentences:
            text = ' '.join([self._prepare_token(token.text) for token in tokens[sentence.begin:sentence.end]]).strip()
            if text:
                prediction = self.predictor.predict(text)['tags']
                prediction[0] = self._separator
                result += prediction

        result = np.array(result)
        return np.argwhere(result == self._separator)[:, 0]

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
                new_edu = DiscourseUnit(start_id+i,
                                        start=tokens[numbers[i]].begin,
                                        end=tokens[numbers[i + 1]].begin - 1,
                                        text=text[tokens[numbers[i]].begin:tokens[numbers[i + 1]].begin],
                                        relation='elementary')
                edus.append(new_edu)

            if numbers.shape[0] == 1:
                i = -1

            new_edu = DiscourseUnit(start_id+i + 1,
                                    start=tokens[numbers[-1]].begin,
                                    end=tokens[-1].end,
                                    text=text[tokens[numbers[-1]].begin:tokens[-1].end],
                                    relation='elementary')
            edus.append(new_edu)

        return edus

    def _prepare_token(self, token):
        for keyword in ['www', 'http']:
            if keyword in token:
                return '_html_'
        return token

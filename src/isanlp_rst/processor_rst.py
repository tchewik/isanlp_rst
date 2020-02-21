import os

from features_processor_default import FeaturesProcessor
from greedy_rst_parser import GreedyRSTParser
from model_segmentator import ModelSegmentator
from rst_tree_predictor import CustomTreePredictor
from sklearn_classifier import SklearnClassifier


class ProcessorRST:
    def __init__(self, model_dir_path):
        self._model_dir_path = model_dir_path

        self.segmentator = ModelSegmentator(self._model_dir_path)

        self._features_processor = FeaturesProcessor(self._model_dir_path)
        self._relation_predictor = SklearnClassifier(
            model_dir_path=os.path.join(self._model_dir_path, 'structure_predictor'))
        self._label_predictor = SklearnClassifier(
            model_dir_path=os.path.join(self._model_dir_path, 'label_predictor'))
        self._nuclearity_predictor = None
        self._tree_predictor = CustomTreePredictor(
            features_processor=self._features_processor,
            relation_predictor=self._relation_predictor,
            label_predictor=self._label_predictor,
            nuclearity_predictor=self._nuclearity_predictor)

        self.parser = GreedyRSTParser(self._tree_predictor, confidence_threshold=0.1)

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        edus = self.segmentator(annot_text, annot_tokens, annot_sentences, annot_lemma, annot_postag,
                 annot_syntax_dep_tree)
        
        if len(edus) == 1:
            return []

        trees = self.parser(edus,
                           annot_text,
                           annot_tokens,
                           annot_sentences,
                           annot_lemma,
                           annot_morph,
                           annot_postag,
                           annot_syntax_dep_tree)

        return trees

import os

from features_processor_default import FeaturesProcessor
from greedy_rst_parser import GreedyRSTParser
from model_segmentator import ModelSegmentator
from rst_tree_predictor import CustomTreePredictor
from sklearn_classifier import SklearnClassifier


class ProcessorRST:
    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        self.segmentator = ModelSegmentator(self.model_dir_path)
        self.default_features_processor = FeaturesProcessor(model_dir_path)
        self.relation_predictor = SklearnClassifier(
            model_dir_path=os.path.join(self.model_dir_path, 'relation_predictor'))
        self.label_predictor = SklearnClassifier(model_dir_path=os.path.join(self.model_dir_path, 'label_predictor'))

        self.predictor = CustomTreePredictor(
            features_processor=self.default_features_processor,
            relation_predictor=self.relation_predictor,
            label_predictor=None)

        self.parser = GreedyRSTParser(self.predictor, forest_threshold=0.)

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_postag, annot_morph, annot_lemma,
                 annot_syntax_dep_tree):
        edus = self.segmentator(annot_text,
                                annot_tokens,
                                annot_sentences,
                                annot_postag,
                                annot_lemma,
                                annot_syntax_dep_tree)

        tree = self.parser(edus,
                           annot_text,
                           annot_tokens,
                           annot_sentences,
                           annot_postag,
                           annot_morph,
                           annot_lemma,
                           annot_syntax_dep_tree)

        return tree

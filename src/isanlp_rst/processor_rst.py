import os

from allennlp_segmentator import AllenNLPSegmentator
from features_processor_default import FeaturesProcessor
from greedy_rst_parser import GreedyRSTParser
from isanlp.annotation import Token, Sentence
# from model_segmentator import ModelSegmentator
from rst_tree_predictor import CustomTreePredictor
from sklearn_classifier import SklearnClassifier


class ProcessorRST:
    def __init__(self, model_dir_path):
        self._model_dir_path = model_dir_path

        # self.segmentator = ModelSegmentator(self._model_dir_path)
        self.segmentator = AllenNLPSegmentator(self._model_dir_path)

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

        # 1. Split text and annotations on paragraphs and process separately
        dus = []
        for chunk in self.split_by_paragraphs(
                annot_text,
                annot_tokens,
                annot_sentences,
                annot_lemma,
                annot_morph,
                annot_postag,
                annot_syntax_dep_tree):

            edus = self.segmentator(chunk['text'], chunk['tokens'], chunk['sentences'], chunk['lemma'],
                                    chunk['postag'], chunk['syntax_dep_tree'])

            if len(edus) == 1:
                dus += edus

            elif len(edus) > 1:
                trees = self.parser(edus,
                                    chunk['text'], chunk['tokens'], chunk['sentences'], chunk['lemma'],
                                    chunk['morph'], chunk['postag'], chunk['syntax_dep_tree'])
                dus += trees

        # 2. Process paragraphs into the document-level annotation
        trees = self.parser(dus,
                            annot_text,
                            annot_tokens,
                            annot_sentences,
                            annot_lemma,
                            annot_morph,
                            annot_postag,
                            annot_syntax_dep_tree)

        return trees


    def split_by_paragraphs(self,
                            annot_text,
                            annot_tokens,
                            annot_sentences,
                            annot_lemma,
                            annot_morph,
                            annot_postag,
                            annot_syntax_dep_tree):
        chunks = []
        previous_boundary = 0
        previous_token = 0
        previous_sentence = 0
        current_sentence = 0
        previous_intersentence_boundary = -1

        for i, token in enumerate(annot_tokens[:-1]):

            if '\n' in annot_text[token.end:annot_tokens[i + 1].begin]:
                current_sentence = \
                    [(j, sentence) for j, sentence in enumerate(annot_sentences) if
                     sentence.begin < i and sentence.end > i][0][
                        0]
                chunk = {
                    'text': annot_text[previous_boundary:token.end + 1].strip(),
                    'tokens': annot_tokens[previous_token:i + 1],
                    'sentences': annot_sentences[previous_sentence:current_sentence + 1],
                }
                sentence_length = annot_sentences[current_sentence].end - annot_sentences[current_sentence].begin
                j = min(i + 1 - previous_token, sentence_length)

                chunk.update({
                    'lemma': [annot_lemma[previous_sentence][previous_intersentence_boundary:]] * (
                            previous_intersentence_boundary > -1) + \
                             [annot_lemma[i] for i in range(previous_sentence + 1, current_sentence)] + \
                             [annot_lemma[current_sentence][:j]],
                    'morph': [annot_morph[previous_sentence][previous_intersentence_boundary:]] * (
                            previous_intersentence_boundary > -1) + \
                             [annot_morph[i] for i in range(previous_sentence + 1, current_sentence)] + \
                             [annot_morph[current_sentence][:j]],
                    'postag': [annot_postag[previous_sentence][previous_intersentence_boundary:]] * (
                            previous_intersentence_boundary > -1) + \
                              [annot_postag[i] for i in range(previous_sentence + 1, current_sentence)] + \
                              [annot_postag[current_sentence][:j]],
                    'syntax_dep_tree': [annot_syntax_dep_tree[previous_sentence][previous_intersentence_boundary:]] * (
                            previous_intersentence_boundary > -1) + \
                                       [annot_syntax_dep_tree[i] for i in range(previous_sentence + 1, current_sentence)] + \
                                       [annot_syntax_dep_tree[current_sentence][:j]],
                })
                previous_boundary = token.end + 1
                previous_intersentence_boundary = j
                previous_token = i + 1
                previous_sentence = current_sentence
                chunks.append(chunk)

        j = min(len(annot_tokens[previous_token:]),
                annot_sentences[current_sentence].end - annot_sentences[current_sentence].begin)
        chunk = {
            'text': annot_text[previous_boundary:].strip(),
            'tokens': annot_tokens[previous_token:],
            'sentences': annot_sentences[previous_sentence:],
            'lemma': annot_lemma[previous_sentence][previous_intersentence_boundary:] * (
                    previous_intersentence_boundary > -1) + \
                     annot_lemma[previous_sentence + 1:],
            'morph': annot_morph[previous_sentence][previous_intersentence_boundary:] * (
                    previous_intersentence_boundary > -1) + \
                     annot_morph[previous_sentence + 1:],
            'postag': annot_postag[previous_sentence][previous_intersentence_boundary:] * (
                    previous_intersentence_boundary > -1) + \
                      annot_postag[previous_sentence + 1:],
            'syntax_dep_tree': annot_syntax_dep_tree[previous_sentence][previous_intersentence_boundary:] * (
                    previous_intersentence_boundary > -1) + \
                               annot_syntax_dep_tree[previous_sentence + 1:],
        }
        chunks.append(chunk)

        def recount_boundaries(chunk):
            begin = chunk['tokens'][0].begin
            chunk['tokens'] = [Token(tok.text, tok.begin - begin, tok.end - begin) for tok in chunk['tokens']]

            return chunk

        def recount_sentences(chunk):
            sentences = []
            lemma = []
            morph = []
            postag = []
            syntax_dep_tree = []

            cursor = 0

            for i, sent in enumerate(chunk['syntax_dep_tree']):
                if len(sent) > 0:
                    new_cursor = cursor + len(sent)
                    sentences.append(Sentence(cursor, new_cursor))
                    lemma.append(chunk['lemma'][i])
                    morph.append(chunk['morph'][i])
                    postag.append(chunk['postag'][i])
                    syntax_dep_tree.append(sent)
                    cursor = new_cursor

            chunk['sentences'] = sentences
            chunk['lemma'] = lemma
            chunk['morph'] = morph
            chunk['postag'] = postag
            chunk['syntax_dep_tree'] = syntax_dep_tree

            return chunk

        result = []
        for chunk in chunks:
            chunk = recount_boundaries(chunk)
            result.append(recount_sentences(chunk))

        return result

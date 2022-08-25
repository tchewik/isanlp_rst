import cky_rst_parser
import classifier_wrappers as clf
import features_processors as feat
import greedy_rst_parser
import os
import segmenters as sgm
import topdown_rst_parser
from isanlp.annotation import Sentence

import rst_tree_predictors as treep

# Segmenter: (class, directory)
_SEGMENTER = {
    'lstm': (sgm.AllenNLPSegmenter, 'segmenter/elmo_ft'),
}

# Feature processing: class
_FEATURE_PROCESSOR = {
    'neural': feat.FeaturesProcessorTokenizer,
    'featurerich': feat.FeaturesProcessor,
}

# Bottom-up tree prediction: (class, directory, paragraph probability threshold, document probability threshold)
_SPAN_PREDICTOR = {
    'neural': (clf.AllenNLPCustomBiMPMClassifier, os.path.join('structure_predictor_bimpm', 'elmo_ft'), 0., 0.6),
    'featurerich': (clf.SklearnClassifier, 'structure_predictor_featurerich', 0.15, 0.2),
    'ensemble': (clf.EnsembleClassifier, '', 0., 0.961)
}

_SPAN_PREDICTOR_ENSEMBLE_PARAMS = {
    'ru': {'voting_type': 'stacking',
           'stacking_model_name': 'structure_ensemble.pkl'},
    'en': {},
}

# Relation classification: (class, directory)
_LABEL_PREDICTOR = {
    'neural': (clf.AllenNLPBiMPMClassifier, os.path.join('label_predictor_bimpm', 'elmo_ft')),
    'featurerich': (clf.SklearnClassifier, 'label_predictor_featurerich'),
    'ensemble': (clf.EnsembleClassifier,)
}

_LABEL_PREDICTOR_ENSEMBLE_PARAMS = {
    'ru': {'voting_type': 'stacking',
           'stacking_model_name': 'relation_ensemble.pkl'},
    'en': {},
}

# Tree prediction (SpanPredictorType_RelationPredictorType): class
_TREE_PREDICTOR = {
    'neural_neural': treep.LargeNNTreePredictor,
    'neural_ensemble': treep.EnsembleNNTreePredictor,
    'ensemble_ensemble': treep.DoubleEnsembleNNTreePredictor,
}

# Full parsing from EDUs: class
_PARSER = {
    'greedy_bottom_up': greedy_rst_parser.GreedyRSTParser,
    'cky_bottom_up': cky_rst_parser.CKYRSTParser,
    'top_down': topdown_rst_parser.TopDownRSTParser
}


class ProcessorRST:
    def __init__(self, model_dir_path, language='ru',
                 segmenter_type='lstm', span_predictor_type='featurerich', label_predictor_type='featurerich',
                 fully_connected=False, add_document_parser=False):

        self._model_dir_path = os.path.join(model_dir_path, language)
        self._language = language
        self._fully_connected = fully_connected

        # Initialize segmenter
        self.segmenter = _SEGMENTER[segmenter_type][0](self._model_dir_path, _SEGMENTER[segmenter_type][1])

        # Initialize feature processor
        _features_type = sorted([span_predictor_type, label_predictor_type])[0]
        _features_type = 'featurerich' if _features_type == 'ensemble' else _features_type
        self._features_processor = _FEATURE_PROCESSOR[_features_type](language=self._language, verbose=0)

        # Initialize bottom-up greedy structure predictor (if necessary)
        if not self._fully_connected:
            if span_predictor_type in ['featurerich', 'neural']:
                self._span_predictor_text = _SPAN_PREDICTOR[span_predictor_type][0](
                    model_dir_path=os.path.join(self._model_dir_path, _SPAN_PREDICTOR[span_predictor_type][1]))
            elif span_predictor_type == 'ensemble':
                types = ['featurerich', 'neural']
                _span_classifiers = []
                for _type in types:
                    _span_classifiers.append(
                        _SPAN_PREDICTOR[_type][0](
                            model_dir_path=os.path.join(self._model_dir_path, _SPAN_PREDICTOR[_type][1])))

                self._span_predictor_text = _SPAN_PREDICTOR['ensemble'][0](
                    _span_classifiers, model_path=self._model_dir_path,
                    **_SPAN_PREDICTOR_ENSEMBLE_PARAMS[self._language])

        # Initialize relation type + nuclearity predictor
        if label_predictor_type in ['featurerich', 'neural']:
            self._label_predictor = _LABEL_PREDICTOR[label_predictor_type][0](
                model_dir_path=os.path.join(self._model_dir_path, _LABEL_PREDICTOR[label_predictor_type][1]))
        elif label_predictor_type == 'ensemble':
            types = ['featurerich', 'neural']
            _label_classifiers = []
            for _type in types:
                _label_classifiers.append(
                    _LABEL_PREDICTOR[_type][0](
                        model_dir_path=os.path.join(self._model_dir_path, _LABEL_PREDICTOR[_type][1])))

            self._label_predictor = _LABEL_PREDICTOR['ensemble'][0](
                _label_classifiers, model_path=self._model_dir_path,
                **_LABEL_PREDICTOR_ENSEMBLE_PARAMS[self._language])

        # Initialize TreePredictor object containing all the required classifiers and feature processors
        _span_typename = span_predictor_type if span_predictor_type != 'featurerich' else 'ensemble'
        _label_typename = label_predictor_type if label_predictor_type != 'featurerich' else 'ensemble'
        self._tree_predictor = _TREE_PREDICTOR['_'.join([_span_typename, _label_typename])](
            features_processor=self._features_processor,
            relation_predictor_sentence=None,
            relation_predictor_text=self._span_predictor_text,
            label_predictor=self._label_predictor,
            nuclearity_predictor=None)

        # Initialize a single top-down or both top-down and bottom-up parsers
        if self._fully_connected:
            self.document_parser = _PARSER['top_down'](
                self._tree_predictor, trained_model_path=os.path.join(self._model_dir_path, 'topdown_model.pt'))
        else:
            self.paragraph_parser = _PARSER['top_down'](
                self._tree_predictor, trained_model_path=os.path.join(self._model_dir_path, 'topdown_model.pt'))
            self.document_parser = _PARSER['greedy_bottom_up'](
                self._tree_predictor, confidence_threshold=_SPAN_PREDICTOR[span_predictor_type][3],
                _same_sentence_bonus=0.)

            self.AVG_TREE_LENGTH = 100  # Varies in different genres (96-116), roughly assuming that's 100 tokens per tree
            self.additional_document_parser = _PARSER['greedy_bottom_up'](
                self._tree_predictor, confidence_threshold=_SPAN_PREDICTOR[span_predictor_type][3] - 0.15,
                _same_sentence_bonus=0.) if add_document_parser else None

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        if self._fully_connected:
            return self.parse_fully_connected(annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph,
                                              annot_postag, annot_syntax_dep_tree)
        else:
            return self.parse_partly_annotated(annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph,
                                               annot_postag, annot_syntax_dep_tree)

    def parse_fully_connected(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                              annot_syntax_dep_tree):

        if '\n' in annot_text:
            chunks = self.split_by_paragraphs(
                annot_text=annot_text,
                annot_tokens=annot_tokens,
                annot_lemma=annot_lemma,
                annot_morph=annot_morph,
                annot_postag=annot_postag,
                annot_syntax_dep_tree=annot_syntax_dep_tree)

            edus = []
            for chunk in chunks:
                chunk_edus = self.segmenter(annot_text, chunk['tokens'], chunk['sentences'], chunk['lemma'],
                                            chunk['postag'], chunk['syntax_dep_tree'], start_id=start_id)
                edus += chunk_edus

        else:
            edus = self.segmenter(annot_text, annot_tokens, annot_sentences, annot_lemma, annot_postag,
                                  annot_syntax_dep_tree, start_id=start_id)

        if len(edus) == 1:
            return edus

        trees = self.document_parser(edus, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph,
                                     annot_postag, annot_syntax_dep_tree)

        return [ProcessorRST.merge_terminal_sameunits(tree, annot_text) for tree in trees]

    def parse_partly_annotated(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                               annot_syntax_dep_tree):

        # 1. Split text and annotations on paragraphs and process separately
        dus = []
        start_id = 0

        if '\n' in annot_text:
            chunks = self.split_by_paragraphs(
                annot_text=annot_text,
                annot_tokens=annot_tokens,
                annot_lemma=annot_lemma,
                annot_morph=annot_morph,
                annot_postag=annot_postag,
                annot_syntax_dep_tree=annot_syntax_dep_tree)

            for chunk in chunks:

                edus = self.segmenter(annot_text, chunk['tokens'], chunk['sentences'], chunk['lemma'],
                                      chunk['postag'], chunk['syntax_dep_tree'], start_id=start_id)

                if len(edus) == 1:
                    dus += edus
                    start_id = edus[-1].id + 1

                elif len(edus) > 1:
                    trees = self.paragraph_parser(edus,
                                                  annot_text, chunk['tokens'], chunk['sentences'], chunk['lemma'],
                                                  chunk['morph'], chunk['postag'], chunk['syntax_dep_tree'])

                    dus += trees
                    start_id = max([tree.id for tree in dus]) + 1

            # 2. Process paragraphs into the document-level annotation
            trees = self.document_parser(dus,
                                         annot_text,
                                         annot_tokens,
                                         annot_sentences,
                                         annot_lemma,
                                         annot_morph,
                                         annot_postag,
                                         annot_syntax_dep_tree)

            # 3. (Optionally) lower the document-level threshold if there were predicted inadequately many trees
            if self.additional_document_parser:
                if len(trees) > len(annot_text.split()) // self.AVG_TREE_LENGTH:
                    trees = self.additional_document_parser(
                        trees,
                        annot_text,
                        annot_tokens,
                        annot_sentences,
                        annot_lemma,
                        annot_morph,
                        annot_postag,
                        annot_syntax_dep_tree
                    )

            return [ProcessorRST.merge_terminal_sameunits(tree, annot_text) for tree in trees]

        else:
            edus = self.segmenter(annot_text, annot_tokens, annot_sentences, annot_lemma,
                                  annot_postag, annot_syntax_dep_tree, start_id=start_id)

            if len(edus) == 1:
                return edus

            trees = self.paragraph_parser(edus, annot_text, annot_tokens, annot_sentences, annot_lemma,
                                          annot_morph, annot_postag, annot_syntax_dep_tree)

            return [ProcessorRST.merge_terminal_sameunits(tree, annot_text) for tree in trees]

    def split_by_paragraphs(self, annot_text, annot_tokens, annot_lemma, annot_morph, annot_postag,
                            annot_syntax_dep_tree):

        def split_on_two(sents, boundary):
            list_sum = lambda l: sum([len(sublist) for sublist in l])

            i = 1
            while list_sum(sents[:i]) < boundary and i < len(sents):
                i += 1

            intersentence_boundary = min(len(sents[i - 1]), boundary - list_sum(sents[:i - 1]))
            return (sents[:i - 1] + [sents[i - 1][:intersentence_boundary]],
                    [sents[i - 1][intersentence_boundary:]] + sents[i:])

        def recount_sentences(chunk):
            sentences = []
            lemma = []
            morph = []
            postag = []
            syntax_dep_tree = []
            tokens_cursor = 0

            for i, sent in enumerate(chunk['syntax_dep_tree']):
                if len(sent) > 0:
                    sentences.append(Sentence(tokens_cursor, tokens_cursor + len(sent)))
                    lemma.append(chunk['lemma'][i])
                    morph.append(chunk['morph'][i])
                    postag.append(chunk['postag'][i])
                    syntax_dep_tree.append(chunk['syntax_dep_tree'][i])
                    tokens_cursor += len(sent)

            chunk['sentences'] = sentences
            chunk['lemma'] = lemma
            chunk['morph'] = morph
            chunk['postag'] = postag
            chunk['syntax_dep_tree'] = syntax_dep_tree

            return chunk

        chunks = []
        prev_right_boundary = -1

        for i, token in enumerate(annot_tokens[:-1]):

            if '\n' in annot_text[token.end:annot_tokens[i + 1].begin]:
                if prev_right_boundary > -1:
                    chunk = {
                        'text': annot_text[annot_tokens[prev_right_boundary].end:token.end + 1].strip(),
                        'tokens': annot_tokens[prev_right_boundary + 1:i + 1]
                    }
                else:
                    chunk = {
                        'text': annot_text[:token.end + 1].strip(),
                        'tokens': annot_tokens[:i + 1]
                    }

                lemma, annot_lemma = split_on_two(annot_lemma, i - prev_right_boundary)
                morph, annot_morph = split_on_two(annot_morph, i - prev_right_boundary)
                postag, annot_postag = split_on_two(annot_postag, i - prev_right_boundary)
                syntax_dep_tree, annot_syntax_dep_tree = split_on_two(annot_syntax_dep_tree, i - prev_right_boundary)

                chunk.update({
                    'lemma': lemma,
                    'morph': morph,
                    'postag': postag,
                    'syntax_dep_tree': syntax_dep_tree,
                })
                chunks.append(recount_sentences(chunk))

                prev_right_boundary = i  # number of last token in the last chunk

        chunk = {
            'text': annot_text[annot_tokens[prev_right_boundary].end:].strip(),
            'tokens': annot_tokens[prev_right_boundary + 1:],
            'lemma': annot_lemma,
            'morph': annot_morph,
            'postag': annot_postag,
            'syntax_dep_tree': annot_syntax_dep_tree,
        }

        chunks.append(recount_sentences(chunk))
        return chunks

    @staticmethod
    def merge_terminal_sameunits(tree, text):
        if tree.relation == 'elementary':
            return tree

        if tree.relation == 'same-unit':
            if tree.left.relation == 'elementary' and tree.right.relation == 'elementary':
                tree.relation = 'elementary'
                tree.start = tree.left.start
                tree.end = tree.right.end
                tree.proba = 1.
                tree.text = text[tree.start:tree.end]
                tree.left = None
                tree.right = None
                return tree

        tree.left = ProcessorRST.merge_terminal_sameunits(tree.left, text)
        tree.right = ProcessorRST.merge_terminal_sameunits(tree.right, text)
        return tree
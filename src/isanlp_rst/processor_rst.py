import os

from allennlp_classifier import AllenNLPClassifier
from allennlp_segmenter import AllenNLPSegmenter
from features_processor_default import FeaturesProcessor as FP_feature_rich
from features_processor_tokenizer import FeaturesProcessor as FP_tokenizer
from greedy_rst_parser import GreedyRSTParser
from isanlp.annotation import Sentence
from model_segmenter import ModelSegmenter
from rst_tree_predictor import CustomTreePredictor, NNTreePredictor
from sklearn_classifier import SklearnClassifier

_SEGMENTER = {
    'lstm': AllenNLPSegmenter,
    'feedforward': ModelSegmenter,
}

_FEATURE_PROCESSOR = {
    'lstm': FP_tokenizer,
    'ensemble': FP_feature_rich,
}

_SPAN_PREDICTOR = {
    'lstm': (AllenNLPClassifier, 'structure_predictor_lstm', 0.1, 0.5),
    'ensemble': (SklearnClassifier, 'structure_predictor', 0.15, 0.2),
}

_LABEL_PREDICTOR = {
    'lstm': (AllenNLPClassifier, 'label_predictor_lstm'),
    'ensemble': (SklearnClassifier, 'label_predictor'),
}

_TREE_PREDICTOR = {
    'lstm': NNTreePredictor,
    'ensemble': CustomTreePredictor
}


class ProcessorRST:
    def __init__(self, model_dir_path, segmenter_type='lstm', span_predictor_type='lstm',
                 label_predictor_type='lstm'):
        self._model_dir_path = model_dir_path

        self.segmenter = _SEGMENTER[segmenter_type](self._model_dir_path)

        fp_type = sorted([span_predictor_type, label_predictor_type])[0]
        self._features_processor = _FEATURE_PROCESSOR[fp_type](self._model_dir_path)

        self._relation_predictor_sentence = None
#         self._relation_predictor_sentence = _SPAN_PREDICTOR[span_predictor_type][0](
#             model_dir_path=os.path.join(self._model_dir_path, 'sent_' + _SPAN_PREDICTOR[span_predictor_type][1]))
        
        self._relation_predictor_text = _SPAN_PREDICTOR[span_predictor_type][0](
            model_dir_path=os.path.join(self._model_dir_path, _SPAN_PREDICTOR[span_predictor_type][1]))

        self._label_predictor = _LABEL_PREDICTOR[label_predictor_type][0](
            model_dir_path=os.path.join(self._model_dir_path, _LABEL_PREDICTOR[label_predictor_type][1]))

        self._nuclearity_predictor = None
        self._tree_predictor = _TREE_PREDICTOR[span_predictor_type](
            features_processor=self._features_processor,
            relation_predictor_sentence=self._relation_predictor_sentence,
            relation_predictor_text=self._relation_predictor_text,
            label_predictor=self._label_predictor,
            nuclearity_predictor=self._nuclearity_predictor)

        self.paragraph_parser = GreedyRSTParser(self._tree_predictor,
                                                confidence_threshold=_SPAN_PREDICTOR[span_predictor_type][2])
        self.document_parser = GreedyRSTParser(self._tree_predictor,
                                               confidence_threshold=_SPAN_PREDICTOR[span_predictor_type][3])

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        # 1. Split text and annotations on paragraphs and process separately
        dus = []
        start_id = 0

        if '\n' in annot_text:
            chunks = self.split_by_paragraphs(
                annot_text,
                annot_tokens,
                annot_sentences,
                annot_lemma,
                annot_morph,
                annot_postag,
                annot_syntax_dep_tree)

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
                    start_id = dus[-1].id + 1

            # 2. Process paragraphs into the document-level annotation
            trees = self.document_parser(dus,
                                         annot_text,
                                         annot_tokens,
                                         annot_sentences,
                                         annot_lemma,
                                         annot_morph,
                                         annot_postag,
                                         annot_syntax_dep_tree)

            return trees

        else:
            edus = self.segmenter(annot_text, annot_tokens, annot_sentences, annot_lemma,
                                  annot_postag, annot_syntax_dep_tree, start_id=start_id)

            if len(edus) == 1:
                return edus

            trees = self.paragraph_parser(edus, annot_text, annot_tokens, annot_sentences, annot_lemma,
                                          annot_morph, annot_postag, annot_syntax_dep_tree)

            return trees

    def split_by_paragraphs(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
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

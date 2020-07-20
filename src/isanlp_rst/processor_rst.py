import os

from allennlp_segmenter import AllenNLPSegmenter
from classifier_wrappers import *
from features_processor_default import *
from greedy_rst_parser import GreedyRSTParser
from isanlp.annotation import Sentence
from model_segmenter import ModelSegmenter  # deprecated
from rst_tree_predictor import *

_SEGMENTER = {
    'lstm': AllenNLPSegmenter,
    'feedforward': ModelSegmenter,
}

_FEATURE_PROCESSOR = {
    'neural': FeaturesProcessorTokenizer,
    'baseline': FeaturesProcessor,
}

_SPAN_PREDICTOR = {
    'neural': (AllenNLPCustomBiMPMClassifier, 'structure_predictor_bimpm', 0., 0.6),
    'baseline': (SklearnClassifier, 'structure_predictor_baseline', 0.15, 0.2),
    'ensemble': (EnsembleClassifier, '', 0., 0.6)
}

_LABEL_PREDICTOR = {
    'neural': (AllenNLPBiMPMClassifier, 'label_predictor_bimpm'),
    'baseline': (SklearnClassifier, 'label_predictor_baseline'),
    'ensemble': (EnsembleClassifier,)
}

_TREE_PREDICTOR = {
    'neural_neural': LargeNNTreePredictor,
    'neural_ensemble': EnsembleNNTreePredictor,
    'ensemble_ensemble': DoubleEnsembleNNTreePredictor,
}


class ProcessorRST:
    def __init__(self, model_dir_path, segmenter_type='lstm', span_predictor_type='baseline',
                 label_predictor_type='baseline'):

        self._model_dir_path = model_dir_path

        self.segmenter = _SEGMENTER[segmenter_type](self._model_dir_path)

        _features_type = sorted([span_predictor_type, label_predictor_type])[0]
        _features_type = 'baseline' if _features_type == 'ensemble' else _features_type
        self._features_processor = _FEATURE_PROCESSOR[_features_type](self._model_dir_path)

        self._span_predictor_sentence = None

        if span_predictor_type != 'ensemble':
            self._span_predictor_text = _SPAN_PREDICTOR[span_predictor_type][0](
                model_dir_path=os.path.join(self._model_dir_path, _SPAN_PREDICTOR[span_predictor_type][1]))
        else:
            _span_classifiers = []

            for _type in ('baseline', 'neural'):
                _span_classifiers.append(
                    _SPAN_PREDICTOR[_type][0](
                        model_dir_path=os.path.join(self._model_dir_path, _SPAN_PREDICTOR[_type][1])
                    )
                )

            self._span_predictor_text = _SPAN_PREDICTOR['ensemble'][0](
                _span_classifiers, weights=[1., 1.]
            )

        if label_predictor_type != 'ensemble':
            self._label_predictor = _LABEL_PREDICTOR[label_predictor_type][0](
                model_dir_path=os.path.join(self._model_dir_path, _LABEL_PREDICTOR[label_predictor_type][1]))
        else:
            _label_classifiers = []

            for _type in ('baseline', 'neural'):
                _label_classifiers.append(
                    _LABEL_PREDICTOR[_type][0](
                        model_dir_path=os.path.join(self._model_dir_path, _LABEL_PREDICTOR[_type][1])
                    )
                )

            self._label_predictor = _LABEL_PREDICTOR['ensemble'][0](
                _label_classifiers, weights=[1., 1.51]
            )

        self._nuclearity_predictor = None
        _span_typename = span_predictor_type if span_predictor_type != 'baseline' else 'ensemble'
        _label_typename = label_predictor_type if label_predictor_type != 'baseline' else 'ensemble'
        self._tree_predictor = _TREE_PREDICTOR['_'.join([_span_typename, _label_typename])](
            features_processor=self._features_processor,
            relation_predictor_sentence=self._span_predictor_sentence,
            relation_predictor_text=self._span_predictor_text,
            label_predictor=self._label_predictor,
            nuclearity_predictor=self._nuclearity_predictor)

        self.AVG_TREE_LENGTH = 400

        self.paragraph_parser = GreedyRSTParser(self._tree_predictor,
                                                confidence_threshold=_SPAN_PREDICTOR[span_predictor_type][2],
                                                _same_sentence_bonus=1.)
        self.document_parser = GreedyRSTParser(self._tree_predictor,
                                               confidence_threshold=_SPAN_PREDICTOR[span_predictor_type][3],
                                               _same_sentence_bonus=0.)
        self.additional_document_parser = GreedyRSTParser(self._tree_predictor,
                                                          confidence_threshold=_SPAN_PREDICTOR[
                                                                                   span_predictor_type][3] - 0.15,
                                                          _same_sentence_bonus=0.)

        self._possible_missegmentations = ("\nIMG",
                                           "\nгимнастический коврик;",
                                           "\nгантели или бутылки с песком;",
                                           "\nнебольшой резиновый мяч;",
                                           "\nэластичная лента (эспандер);",
                                           "\nхула-хуп (обруч).",
                                           "\n200?",
                                           "\n300?",
                                           "\nНе требуйте странного.",
                                           "\nИспользуйте мою модель.",
                                           '\n"А чего вы от них требуете?"',
                                           '\n"Решить проблемы с тестерами".',
                                           "\nКак гончая на дичь.",
                                           "\nИ крупная.",
                                           "\nВ прошлом году компания удивила рынок",
                                           "\nЧужой этики особенно.",
                                           "\nНо и своей тоже.",
                                           "\nАэропорт имени,",
                                           "\nА вот и монголы.",
                                           "\nЗолотой Будда.",
                                           "\nДворец Богдо-Хана.",
                                           "\nПлощадь Сухэ-Батора.",
                                           "\nОдноклассники)",
                                           "\nВечерняя площадь.",
                                           "\nТугрики.",
                                           "\nВнутренние монголы.",
                                           "\nВид сверху.",
                                           "\nНациональный парк Тэрэлж. IMG IMG",
                                           '\nГора "Черепаха".',
                                           "\nПуть к медитации.",
                                           "\nЖить надо высоко,",
                                           "\nЧан с кумысом.",
                                           "\nЖилая юрта.",
                                           "\nКумыс.",
                                           "\nТрадиционное занятие монголов",
                                           "\nДвугорбый верблюд мало где",
                                           "\nМонгол Шуудан переводится",
                                           "\nОвощные буузы.",
                                           "\nЗнаменитый чай!",
                                           "\nменя приняли кандидатом",
                                           )

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        # 1. Split text and annotations on paragraphs and process separately
        dus = []
        start_id = 0

        for missegmentation in self._possible_missegmentations:
            annot_text = annot_text.replace(missegmentation, ' ' + missegmentation[1:])

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

            # 3. lower the document-level threshold if there were predicted inadequately many trees
            if len(trees) > len(annot_text) // self.AVG_TREE_LENGTH:
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

            return trees

        else:
            edus = self.segmenter(annot_text, annot_tokens, annot_sentences, annot_lemma,
                                  annot_postag, annot_syntax_dep_tree, start_id=start_id)

            if len(edus) == 1:
                return edus

            trees = self.paragraph_parser(edus, annot_text, annot_tokens, annot_sentences, annot_lemma,
                                          annot_morph, annot_postag, annot_syntax_dep_tree)

            return trees

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
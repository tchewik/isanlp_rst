import copy
import glob
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import fire
from tqdm import tqdm

from isanlp_rst.dmrst_parser.src.corpus.binary_tree import BinaryTree
from isanlp_rst.dmrst_parser.src.corpus.data import Rs3Document
from isanlp_rst.dmrst_parser.src.parser.data import Data
from isanlp_rst.dmrst_parser.src.parser.data import RelationTableGUM, RelationTableRSTDT, RelationTableRuRSTB

random.seed(42)


class ParserInput:
    def __init__(self):
        self.sentences = []
        self.edu_breaks = []
        self.label_for_metrics_list = []
        self.label_for_metrics = ''
        self.parsing_index = []
        self.relation = []
        self.decoder_inputs = []
        self.parents = []
        self.siblings = []
        self.sentence_span = []


class DataManager:
    def __init__(self, corpus,
                 cross_validation=False, nfolds=5, ):
        """
        :param corpus: str  - from {'GUM', 'RST-DT', 'RuRSTB'}
        :param cross_validation: bool  - whether to split to stratified train/dev/tests randomly
        :param nfolds: int  - [If cross_validation == True] number of splits for cross validation
        """
        self.corpus_name = corpus
        if self.corpus_name == 'GUM':
            self._init_gum_corpus(cross_validation, nfolds)

        elif self.corpus_name == 'RST-DT':
            self._init_rstdt_corpus(nfolds)

        elif corpus == 'RuRSTB':
            self._init_rurstb_corpus()

    def _init_gum_corpus(self, cross_validation, nfolds):
        self.input_path = 'data/gum_rs3'
        self.output_path = Path('data/gum_prepared')
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cross_validation = cross_validation
        if self.cross_validation:
            self.nfolds = nfolds
            self.folds = defaultdict(dict[int, dict])
            self.mixed_folds_en = defaultdict(dict[int, list])
            self.mixed_folds_ru = defaultdict(dict[int, list])
        else:
            self.corpus = dict()

            self.mixed_train_en = defaultdict(list[dict[int, list]])
            self.mixed_train_ru = defaultdict(list[dict[int, list]])
            self.mixed_folds = 5
            for i in [25, 50, 75, 100]:
                self.mixed_train_en[i] = []
                self.mixed_train_ru[i] = []

        self.langs = ['en', 'ru']

        self.relation_table = RelationTableGUM
        self.relation_dic = {word.lower(): i for i, word in enumerate(RelationTableGUM)}
        self.relation_fixer = {
            'topic_ns': 'contingency_ns',  # One example of this type in GUM v9.1
            'restatement_sn': 'restatement_ns'  # 4 examples in GUM_conversation_gossip
        }

    def _init_rstdt_corpus(self, nfolds):
        # The corpus is converted to *.rs3 with https://github.com/rst-workbench/rst-converter-service
        self.input_path = 'data/rstdt_rs3'
        self.output_path = Path('data/rstdt_prepared')
        self.output_path.mkdir(parents=True, exist_ok=True)

        # There is no fixed validation part in RST-DT,
        # so we'll take random parts of training for validation for each "fold"
        self.nfolds = nfolds
        self.folds = defaultdict(dict[int, dict])

        class2rel = {
            'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative'],
            'Background': ['background', 'background-e', 'circumstance', 'circumstance-e'],
            'Cause': ['cause', 'cause-result', 'result', 'result-e', 'consequence', 'consequence-n-e',
                      'consequence-n', 'consequence-s-e', 'consequence-s'],
            'Comparison': ['comparison', 'comparison-e', 'preference', 'preference-e', 'analogy', 'analogy-e',
                           'proportion'],
            'Condition': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise'],
            'Contrast': ['contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e'],
            'Elaboration': ['elaboration-additional', 'elaboration-additional-e', 'elaboration-general-specific-e',
                            'elaboration-general-specific', 'elaboration-part-whole', 'elaboration-part-whole-e',
                            'elaboration-process-step', 'elaboration-process-step-e',
                            'elaboration-object-attribute-e', 'elaboration-object-attribute',
                            'elaboration-set-member', 'elaboration-set-member-e', 'example', 'example-e',
                            'definition', 'definition-e'],
            'Enablement': ['purpose', 'purpose-e', 'enablement', 'enablement-e'],
            'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'interpretation-n',
                           'interpretation-s-e', 'interpretation-s', 'interpretation', 'conclusion', 'comment',
                           'comment-e', 'comment-topic'],
            'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e',
                            'reason', 'reason-e'],
            'Joint': ['list', 'disjunction'],
            'Manner-Means': ['manner', 'manner-e', 'means', 'means-e'],
            'Topic-Comment': ['problem-solution', 'problem-solution-n', 'problem-solution-s', 'question-answer',
                              'question-answer-n', 'question-answer-s', 'statement-response',
                              'statement-response-n', 'statement-response-s', 'topic-comment', 'comment-topic',
                              'rhetorical-question'],
            'Summary': ['summary', 'summary-n', 'summary-s', 'restatement', 'restatement-e'],
            'Temporal': ['temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e',
                         'temporal-same-time', 'temporal-same-time-e', 'sequence', 'inverted-sequence'],
            'Topic-Change': ['topic-shift', 'topic-drift'],
            'textual-organization': ['textualorganization'],
            'span': ['span'],
            'same-unit': ['same-unit']
        }
        # rel_status_classes = []
        # for rel in class2rel:
        #     rel_status_classes.append(rel + '_NS')
        #     rel_status_classes.append(rel + '_NN')
        #     rel_status_classes.append(rel + '_SN')

        self.rel2class = {}
        for cl in class2rel:
            self.rel2class[cl.lower()] = cl
            for rel in class2rel[cl]:
                self.rel2class[rel] = cl

        self.relation_table = RelationTableRSTDT
        self.relation_dic = {word.lower(): i for i, word in enumerate(RelationTableRSTDT)}
        self.relation_fixer = dict()

    def _init_rurstb_corpus(self):
        # The corpus is splitted into separate trees (docname_part*.rs3)
        # "##### " are replaced with <P> tag as in the rst-dt
        # (although it still marks the beginning of a paragraph here, not the ending)
        # Also the corpus converted from rs3 -> isanlp -> rs3 to fix empty spans

        self.input_path = 'data/rurstb_rs3'
        self.output_path = Path('data/rurstb_prepared')
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cross_validation = False
        self.corpus = {'train': [], 'dev': [], 'test': []}
        class2rel = {
            'Attribution': ['attribution', 'antithesis'],  # Corpus analysis shows often mislabeling
            'Background': ['background'],
            'Cause-effect': ['cause', 'effect', 'cause-effect'],
            'Comparison': ['comparison'],
            'Concession': ['concession'],
            'Condition': ['condition', 'motivation'],
            'Contrast': ['contrast'],
            'Elaboration': ['elaboration'],
            'Preparation': ['preparation'],
            'Purpose': ['purpose'],
            'Interpretation-evaluation': ['evaluation', 'interpretation', 'interpretation-evaluation'],
            'Evidence': ['evidence'],
            'Joint': ['joint'],
            'Solutionhood': ['solutionhood'],
            'Restatement': ['restatement', 'conclusion'],
            'Sequence': ['sequence'],
            'span': ['span'],
            'same-unit': ['same-unit']
        }

        self.rel2class = {}
        for cl in class2rel:
            self.rel2class[cl.lower()] = cl
            for rel in class2rel[cl]:
                self.rel2class[rel] = cl

        self.relation_table = RelationTableRuRSTB
        self.relation_dic = {word.lower(): i for i, word in enumerate(RelationTableRuRSTB)}
        self.relation_fixer = {
            'restatement_sn': 'condition_sn',
            'restatement_ns': 'elaboration_ns',
            'solutionhood_ns': 'solutionhood_sn',
            'preparation_ns': 'elaboration_ns',
            'elaboration_sn': 'preparation_sn',
            'background_ns': 'elaboration_ns',
        }

    def from_rs3(self):
        # Collect all *.edus, *.lisp in the same directory
        self.prepare_lisp_format()

        # Collect pickled binaries for each document
        self.prepare_parser_format()

        if self.corpus_name == 'GUM':
            if self.cross_validation:
                # Prepare documents listings for each fold and split, including mixed variants.
                # Populate self.folds = {1: {'train': [...], 'dev': [...], 'test': [...]}, 2: ...}
                self.construct_folds()
                # Populate self.mixed_folds_en = {25: {1: ...}, 50: {1: ...}, 75: {1: ...}}, and self.mixed_folds_ru
                self._mixed_folds(25)
                self._mixed_folds(50)
                self._mixed_folds(75)
                self._mixed_folds(100)
            else:
                self.construct_corpus()
                self._mixed_train(25)
                self._mixed_train(50)
                self._mixed_train(75)
                self._mixed_train(100)

        elif self.corpus_name in ['RST-DT', 'RuRSTB']:
            self.construct_corpus()

    def from_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def prepare_lisp_format(self):
        if self.corpus_name == 'GUM':
            for lang in self.langs:
                for rs3_file in glob.glob(os.path.join(self.input_path, lang, '*.rs3')):
                    self.convert_doc(filename=os.path.basename(rs3_file),
                                     input_dir=os.path.join(self.input_path, lang),
                                     output_dir=self.output_path)

        elif self.corpus_name == 'RST-DT':
            for part in ('TRAINING', 'TEST'):
                for rs3_file in sorted(glob.glob(os.path.join(self.input_path, part, '*.rs3'))):
                    self.convert_doc(filename=os.path.basename(rs3_file),
                                     input_dir=os.path.join(self.input_path, part),
                                     output_dir=self.output_path)

        elif self.corpus_name == 'RuRSTB':
            for rs3_file in sorted(glob.glob(os.path.join(self.input_path, '*.rs3'))):
                try:
                    self.convert_doc(filename=os.path.basename(rs3_file),
                                     input_dir=self.input_path,
                                     output_dir=self.output_path)
                except Exception as e:
                    print(rs3_file)
                    raise e

    def prepare_parser_format(self):
        files = list(self.output_path.glob('*.edus'))
        for edu_path in tqdm(files, desc='Reading *.lisp files'):
            lisp_path = edu_path.parent.joinpath(edu_path.name[:-5] + '.lisp')
            try:
                parser_input = self.generate_input(lisp_path, edu_path, edu_path)
            except Exception as e:
                print('Exception is evoked by:', edu_path)
                raise e
            with open(edu_path.parent.joinpath(edu_path.name[:-5] + '.pkl'), 'wb') as f:
                pickle.dump(parser_input, f)

    def get_fold(self, number, lang='en', mixed=0):
        """
        :param number: int  - fold number
        :param lang: str  - (main) language
        :param mixed: int  - percentage for other part mixing
        :return: tuple(src.parser.data.Data)  - train, dev, test:
        """
        if self.corpus_name == 'GUM':
            if mixed == 0:
                fold = copy.deepcopy(self.folds[number])
                if lang == 'ru':
                    for key in ['train', 'dev', 'test']:
                        fold[key] = [docname + '_RU' for docname in fold[key]]
            else:
                if lang == 'en':
                    fold = copy.deepcopy(self.mixed_folds_en[mixed][number])
                elif lang == 'ru':
                    fold = copy.deepcopy(self.mixed_folds_ru[mixed][number])
                else:
                    raise KeyError('No such language in the current data manager.')

        elif self.corpus_name == 'RST-DT':
            fold = copy.deepcopy(self.folds[number])

        result = {'train': None, 'dev': None, 'test': None}
        for key in result.keys():
            docs = [pickle.load(open(self.output_path.joinpath(docname + '.pkl'), 'rb')) for docname in fold[key]]
            input_sentences = [doc.sentences for doc in docs]
            edu_breaks = [doc.edu_breaks for doc in docs]
            decoder_input = [doc.decoder_inputs for doc in docs]
            relation_label = [doc.relation for doc in docs]
            parsing_breaks = [doc.parsing_index for doc in docs]
            golden_metric = [' '.join(doc.label_for_metrics_list) for doc in docs]
            parents_index = [doc.parents for doc in docs]
            sibling = [doc.siblings for doc in docs]
            result[key] = Data(input_sentences, edu_breaks, decoder_input,
                               relation_label, parsing_breaks, golden_metric,
                               parents_index, sibling)

        return result['train'], result['dev'], result['test']

    def get_data(self, lang='en', mixed=0, mixed_fold=0):
        """
        :param lang: str  - (main) language
        :param mixed: int  - percentage for other part mixing
        :return: tuple(src.parser.data.Data)  - train, dev, test:
        """
        corpus = copy.deepcopy(self.corpus)
        if self.corpus_name == 'GUM':
            if lang == 'ru':
                for key in ['train', 'dev', 'test']:
                    corpus[key] = [docname + '_RU' for docname in corpus[key]]

            if mixed:
                if lang == 'en':
                    corpus['train'] = copy.deepcopy(self.mixed_train_en[mixed][mixed_fold])
                elif lang == 'ru':
                    corpus['train'] = copy.deepcopy(self.mixed_train_ru[mixed][mixed_fold])
                else:
                    raise KeyError('No such language in the current data manager.')

        result = {'train': None, 'dev': None, 'test': None}
        for key in result.keys():
            docs = []
            for docname in corpus[key]:
                filename = self.output_path.joinpath(docname + '.pkl')
                try:
                    docs.append(pickle.load(open(filename, 'rb')))
                except FileNotFoundError:
                    print('No such file in the corpus:', filename)

            input_sentences = [doc.sentences for doc in docs]
            edu_breaks = [doc.edu_breaks for doc in docs]
            decoder_input = [doc.decoder_inputs for doc in docs]
            relation_label = [doc.relation for doc in docs]
            parsing_breaks = [doc.parsing_index for doc in docs]
            golden_metric = [' '.join(doc.label_for_metrics_list) for doc in docs]
            parents_index = [doc.parents for doc in docs]
            sibling = [doc.siblings for doc in docs]
            result[key] = Data(input_sentences, edu_breaks, decoder_input,
                               relation_label, parsing_breaks, golden_metric,
                               parents_index, sibling)

        return result['train'], result['dev'], result['test']

    def construct_corpus(self):
        if self.corpus_name == 'GUM':
            for part in ('train', 'dev', 'test'):
                self.corpus[part] = open(os.path.join('data', 'gum_file_lists', 'files.' + part),
                                         'r').read().splitlines()

        elif self.corpus_name == 'RST-DT':
            test_files = [os.path.basename(filename)[:-4]
                          for filename in glob.glob(os.path.join(self.input_path, 'TEST', '*.rs3'))]
            all_train_files = [os.path.basename(filename)[:-4]
                               for filename in glob.glob(os.path.join(self.input_path, 'TRAINING', '*.rs3'))]

            for fold in range(self.nfolds):
                train_n = int(len(all_train_files) * 0.9)
                train_files = random.sample(all_train_files, train_n)
                dev_files = [file for file in all_train_files if not file in train_files]

                self.folds[fold]['train'] = train_files
                self.folds[fold]['dev'] = dev_files
                self.folds[fold]['test'] = test_files

        elif self.corpus_name == 'RuRSTB':
            for filename in glob.glob(os.path.join(self.output_path, '*.pkl')):
                clear_filename = os.path.basename(filename)[:-4]
                part = clear_filename.split('.')[0]
                self.corpus[part].append(clear_filename)

    def _collect_mixed_train(self, train_data, genres: list, n: int, another_lang: str):
        mixed_train = train_data[:]
        for genre in genres:
            g_train = [filename for filename in train_data if filename.startswith(f'GUM_{genre}')]

            if another_lang == 'ru':
                length_of_replacements = int(len(g_train) * n / 100)
                another_lang_sample = random.sample(list(range(len(g_train))), length_of_replacements)
                for ind in another_lang_sample:
                    mixed_train.append(g_train[ind] + '_RU')

            else:
                length_of_replacements = int(len(g_train) * (100 - n) / 100)
                ru_sample_ind = random.sample(list(range(len(g_train))), length_of_replacements)
                for ind in ru_sample_ind:
                    mixed_train.append(g_train[ind] + '_RU')

        return mixed_train

    def _mixed_train(self, n):
        """ Makes self.mixed_train_* versions with 100% train files from first language and n% from the second. """

        if self.corpus_name == 'GUM':
            genres = ['academic', 'bio', 'conversation', 'fiction', 'interview', 'news', 'reddit',
                      'speech', 'textbook', 'vlog', 'voyage', 'whow']

        elif self.corpus_name == 'RuRSTB':
            genres = ['news', 'blogs']

        # Base English, mixing Russian #############
        for _ in range(self.mixed_folds):
            self.mixed_train_en[n].append(
                self._collect_mixed_train(self.corpus['train'], genres, n=n, another_lang='ru'))

        # mixed_train = train[:]
        # for genre in genres:
        #     g_train = [filename for filename in train if filename.startswith(f'GUM_{genre}')]
        #     length_of_replacements = int(len(g_train) * n / 100)
        #     ru_sample_ind = random.sample(list(range(len(g_train))), length_of_replacements)
        #     for ind in ru_sample_ind:
        #         mixed_train.append(g_train[ind] + '_RU')
        #
        # self.mixed_train_en[n] = mixed_train

        # Base Russian, mixing English #############
        # train = self.corpus['train'][:]
        # mixed_train = train[:]
        # for genre in genres:
        #     g_train = [filename for filename in train if filename.startswith(f'GUM_{genre}')]

        # Base English, mixing Russian #############
        for _ in range(self.mixed_folds):
            self.mixed_train_ru[n].append(
                self._collect_mixed_train(self.corpus['train'], genres, n=n, another_lang='en'))

    def generate_input(self, lisp_path, text_path, edus_path, is_depth_manner=True):
        tree = BinaryTree(lisp_path, text_path, edus_path)
        edus_list = [edu.split() for edu in open(edus_path, 'r').read().splitlines()]  # GUM is pre-tokenized
        return self.find_document_span(tree.root, edus_list, is_depth_manner, tree.sentence_span)

    def find_document_span(self, node, edus_list, is_depth_manner, sentence_span_dic):
        parser_input = self.parse_sentence(node, edus_list, is_depth_manner)
        parser_input.sentence_span = self.get_sentence_span_list(sentence_span_dic)
        return parser_input

    @staticmethod
    def get_sentence_span_list(sentence_span_dic):
        sentence_list = []
        for key in sentence_span_dic:
            tem_str = key.replace('[', '').replace(']', '')
            tokens = tem_str.split(',')
            left = int(tokens[0])
            right = int(tokens[1])
            sentence_list.append([left, right])
        return sentence_list

    def parse_sentence(self, root_node, edus_list, is_depth_manner, coarse=True):
        def get_depth_manner_node_list(root):
            node_list = []
            stack = []
            stack.append(root)
            while len(stack) > 0:
                node = stack.pop()
                node_list.append(node)
                if node.right is not None:
                    stack.append(node.right)
                if node.left is not None:
                    stack.append(node.left)
            return node_list

        def get_width_manner_node_list(root):
            node_list = []
            queue = []
            if root is not None:
                queue.append(root)
            while len(queue) != 0:
                node = queue.pop(0)
                node_list.append(node)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            return node_list

        root_node.parent = None
        parser_input = ParserInput()
        if is_depth_manner:
            node_list = get_depth_manner_node_list(root_node)
        else:
            node_list = get_width_manner_node_list(root_node)

        sentences_list = []

        edu_start = root_node.span[0]
        for node in node_list:
            if node.edu_id is not None:
                sentences_list.append([node.edu_id, edus_list[node.edu_id - 1]])
            else:
                parser_input.parsing_index.append(node.left.span[1] - edu_start)
                parser_input.decoder_inputs.append(node.span[0] - edu_start)

                parent_index = node.parent.span[1] - edu_start if node.parent is not None else 0
                parser_input.parents.append(parent_index)

                if node.parent is None:
                    sibling_index = 99
                else:
                    if node == node.parent.left:
                        sibling_index = 99
                    else:
                        sibling_index = node.parent.left.span[1] - edu_start

                parser_input.siblings.append(sibling_index)

                #   LabelforMetric:
                left_child_span = node.left.span
                right_child_span = node.right.span
                nuclearity = node.relation[:2]
                relation = node.relation[3:]

                # Label to Class
                if self.corpus_name == 'GUM':
                    if coarse and relation != 'same-unit':
                        relation = relation.split('-')[0]
                elif self.corpus_name in ['RST-DT', 'RuRSTB']:
                    relation = self.rel2class.get(relation.lower())

                #   Relation:
                lookup_relation = (relation + '_' + nuclearity).lower()
                if lookup_relation in self.relation_fixer:
                    lookup_relation = self.relation_fixer.get(lookup_relation)
                    relation, nuclearity = lookup_relation.split('_')
                    nuclearity = nuclearity.upper()
                    if relation != 'same-unit':
                        relation = relation[0].upper() + relation[1:]

                parser_input.relation.append(self.relation_dic[lookup_relation])
                left_nuclearity = 'Nucleus' if nuclearity[0] == 'N' else 'Satellite'
                right_nuclearity = 'Nucleus' if nuclearity[1] == 'N' else 'Satellite'
                if nuclearity == 'NS' or nuclearity == 'SN':
                    if nuclearity == 'NS':
                        left_relation = 'span'
                        right_relation = relation
                    else:
                        left_relation = relation
                        right_relation = 'span'
                else:
                    left_relation = relation
                    right_relation = relation
                label_string = '(' + str(
                    left_child_span[0] - edu_start + 1) + ':' + left_nuclearity + '=' + left_relation + ':' + str(
                    left_child_span[1] - edu_start + 1) + ',' + str(
                    right_child_span[0] - edu_start + 1) + ':' + right_nuclearity + '=' + right_relation + ':' + str(
                    right_child_span[1] - edu_start + 1) + ')'
                parser_input.label_for_metrics_list.append(label_string)

        parser_input.LabelforMetric = [' '.join(parser_input.label_for_metrics_list)]
        Sentences_list = sorted(sentences_list, key=lambda x: x[0])

        for i in range(len(Sentences_list)):
            parser_input.sentences += Sentences_list[i][1]
            parser_input.edu_breaks.append(len(parser_input.sentences) - 1)

        return parser_input

    def convert_doc(self, filename, input_dir, output_dir):
        """ Take all rs3 documents and save them in the same directory
            as *.edus and *.lisp files ready for processing. """
        rs3 = Rs3Document(os.path.join(input_dir, filename))
        rs3.read()
        rs3.writeEdu(output_dir)
        out_ext = '.lisp'
        rs3.writeTree(output_dir, out_ext)

    def construct_folds(self):
        """ Scatter examples on folds divided into train/val/test.
            Preserve subclasses distribution in each fold and split. """

        documents = defaultdict(list)
        for edu_file in self.output_path.glob('*.edus'):
            name = edu_file.stem
            doc_lang = 'ru' if '_RU' in name else 'en'
            name = name[:-3] if '_RU' in name else name
            documents[doc_lang].append(name)

        all_docs = documents['en']
        docs_by_class = defaultdict(list)
        for doc_name in all_docs:
            cls = doc_name.split('_')[1]
            docs_by_class[cls].append(doc_name)

        for i in range(self.nfolds):
            fold_docs = {}
            for cls, doc_names in docs_by_class.items():
                fold_cls_docs = doc_names[:]
                random.shuffle(fold_cls_docs)
                fold_docs[cls] = fold_cls_docs

            train_docs = []
            val_docs = []
            test_docs = []

            train_props = {c: int(len(fold_docs[c]) * 0.7) for c in docs_by_class}
            val_props = {c: int(len(fold_docs[c]) * 0.15) for c in docs_by_class}
            remaining = copy.deepcopy(fold_docs)

            for c in docs_by_class:
                train_docs.extend(random.sample(remaining[c], train_props[c]))
                remaining[c] = [d for d in remaining[c] if d not in train_docs]
                val_docs.extend(random.sample(remaining[c], val_props[c]))
                remaining[c] = [d for d in remaining[c] if d not in val_docs]
                test_docs.extend(remaining[c])

            self.folds[i] = {
                'train': train_docs,
                'dev': val_docs,
                'test': test_docs
            }

    def _mixed_folds(self, n):
        """ Populates a self.mixed_folds_en{25: ..., 75: ..., 100: ...} dictionary
            with n% train files from first language and 100-n% from the second. """

        mixed_folds_en = defaultdict(dict[str, list])
        mixed_folds_ru = defaultdict(dict[str, list])
        for fold_num, fold in self.folds.items():
            # Base English, mixing Russian #############
            mixed_folds_en[fold_num]['dev'] = fold['dev'][:]
            mixed_folds_en[fold_num]['test'] = fold['test'][:]

            length_of_replacements = int(len(fold['train']) * n / 100)
            ru_sample_ind = random.sample(list(range(len(fold['train']))), length_of_replacements)
            train = fold['train'][:]
            for ind in ru_sample_ind:
                train[ind] += '_RU'


            mixed_folds_en[fold_num]['train'] = train

            # Base Russian, mixing English #############
            mixed_folds_ru[fold_num]['dev'] = fold['dev'][:]
            mixed_folds_ru[fold_num]['test'] = fold['test'][:]
            for ind in range(len(fold['dev'])):
                mixed_folds_ru[fold_num]['dev'][ind] += '_RU'

            length_of_replacements = int(len(fold['train']) * (100 - n) / 100)
            ru_sample_ind = random.sample(list(range(len(fold['train']))), length_of_replacements)
            train = fold['train'][:]
            for ind in ru_sample_ind:
                train[ind] += '_RU'

            mixed_folds_ru[fold_num]['train'] = train

        self.mixed_folds_en[n] = mixed_folds_en
        self.mixed_folds_ru[n] = mixed_folds_ru


def collect(corpus='GUM', output_path='data/data_manager.pickle'):
    dp = DataManager(corpus=corpus)
    dp.from_rs3()
    dp.save(output_path)


if __name__ == '__main__':
    fire.Fire(collect)

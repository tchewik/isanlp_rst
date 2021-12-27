from __future__ import division
from __future__ import print_function

import functools
import os
import pickle
import re
import sys
import time
import warnings

import nltk
import numpy as np
import pandas as pd
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.translate import bleu_score, chrf_score
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import paired_euclidean_distances
from utils.features_processor_variables import MORPH_FEATS, FPOS_COMBINATIONS, count_words_x, \
    count_words_y, pairs_words, relations_related
from utils.synonyms_vocabulary import synonyms_vocabulary

warnings.filterwarnings('ignore')

import razdel

def tokenize(text):
    result = ' '.join([tok.text for tok in razdel.tokenize(text)])
    return result

class FeaturesProcessor:
    CATEGORY = 'category_id'

    def __init__(self,
                 model_dir_path: str,
                 verbose=0,
                 use_markers=True,
                 use_morphology=True,
                 embed_model_stopwords=True,
                 use_w2v=True,
                 use_sentiment=True):
        """
        :param str model_dir_path: path with all the models
        :param int verbose: 0 for no logging, 1 for time counts logging and 2 for all warnings
        :param bool embed_model_stopwords: do count stopwords during w2v vectorization
        :param use_w2v: do w2v vectorization
        :param use_sentiment: do sentiment extraction
        """

        self._model_dir_path = model_dir_path
        self._verbose = verbose
        self._use_markers = use_markers
        self._use_morphology = use_morphology
        self._embed_model_stopwords = embed_model_stopwords
        self._use_w2v = use_w2v
        self._use_sentiment = use_sentiment

        if self._verbose:
            print("Processor initialization...\t", end="", flush=True)

        self.relations_related = relations_related
        self.stop_words = nltk.corpus.stopwords.words('russian')
        self.count_words_x = count_words_x
        self.count_words_y = count_words_y
        self.pairs_words = pairs_words

        self.vectorizer = pickle.load(open(os.path.join(model_dir_path, 'tf_idf', 'pipeline.pkl'), 'rb'))

        # preprocessing functions
        self._uppercased = lambda snippet, length: sum(
            [word[0].isupper() if len(word) > 0 else False for word in snippet.split()]) / length
        self._start_with_uppercase = lambda snippet, length: sum(
            [word[0].isupper() if len(word) > 0 else False for word in snippet.split(' ')]) / length

        if self._use_w2v:
            self.embed_model_path = os.path.join(model_dir_path, 'w2v', 'default', 'model.vec')
            self._synonyms_vocabulary = synonyms_vocabulary

            # embeddings
            if self.embed_model_path[-4:] in ['.vec', '.bin']:
                self.word2vec_model = KeyedVectors.load_word2vec_format(self.embed_model_path,
                                                                        binary=self.embed_model_path[-4:] == '.bin')
            else:
                self.word2vec_model = Word2Vec.load(self.embed_model_path)

            test_word = ['дерево', 'NOUN']
            try:
                self.word2vec_vector_length = len(self.word2vec_model.wv.get_vector(test_word[0]))
                self.word2vec_tag_required = False
            except KeyError:
                self.word2vec_vector_length = len(self.word2vec_model.wv.get_vector('_'.join(test_word)))
                self.word2vec_tag_required = True

            self.word2vec_stopwords = embed_model_stopwords

        self._remove_stop_words = lambda lemmatized_snippet: [word for word in lemmatized_snippet if
                                                              word not in self.stop_words]

        self.fpos_combinations = FPOS_COMBINATIONS

        if self._use_sentiment:
            self.sentiment_model = FastTextSocialNetworkModel(tokenizer=RegexTokenizer())

        # self._context_length = 3

        if self._verbose:
            print('[DONE]')

    def _find_y(self, snippet_x, snippet_y, loc_x):
        result = self.annot_text.find(snippet_y, loc_x + len(snippet_x) - 1)
        if result < 1:
            result = self.annot_text.find(snippet_y, loc_x + 1)
        if result < 1:
            result = loc_x + 1
        return result

    def __call__(self, df_, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        df = df_[:]
        df.snippet_x = df.snippet_x.replace('\n', ' ', regex=True).replace('  ', ' ', regex=True)
        df.snippet_x = df.snippet_x.map(tokenize)
        df.snippet_y = df.snippet_y.replace('\n', ' ', regex=True).replace('  ', ' ', regex=True)
        df.snippet_y = df.snippet_y.map(tokenize)

        # self.annot_text = annot_text.replace('\n', ' ').replace('  ', ' ')
        self.annot_text = tokenize(annot_text)
        self.annot_tokens = annot_tokens
        self.annot_sentences = annot_sentences
        self.annot_lemma = annot_lemma
        self.annot_morph = annot_morph
        self.annot_postag = annot_postag
        self.annot_syntax_dep_tree = annot_syntax_dep_tree

        t, t_final = None, None
        if self._verbose:
            t = time.time()
            t_final = t
            print('1\t', end="", flush=True)

        df['is_broken'] = False

        # map discourse units to annotations
        if not 'loc_x' in df.keys():
            df['loc_x'] = df.snippet_x.map(self.annot_text.find)
        if not 'loc_y' in df.keys():
            df['loc_y'] = df.apply(lambda row: self._find_y(row.snippet_x, row.snippet_y, row.loc_x - 1), axis=1)

        df['token_begin_x'] = df.loc_x.map(self.locate_token)
        df['token_begin_y'] = df.loc_y.map(self.locate_token)

        try:
            df['token_end_y'] = df.apply(lambda row: self.locate_token(row.loc_y + len(row.snippet_y)), axis=1)  # -1
            df['token_end_y'] = df['token_end_y'] + (df['token_end_y'] == df['token_begin_y']) * 1
        except:
            if self._verbose == 2:
                print(f'Unable to locate second snippet >>> {(df.snippet_x.values, df.snippet_y.values)}',
                      file=sys.stderr)
                print(self.annot_text, file=sys.stderr)
            df['tokens_x'] = df.snippet_x.map(lambda row: row.split())
            df['tokens_y'] = df.snippet_y.map(lambda row: row.split())
            # df['left_context'] = ['_END_'] * self._context_length
            # df['right_context'] = ['_END_'] * self._context_length
            df['same_sentence'] = 0
            df['is_broken'] = True
            return df

        # length of tokens sequence
        df['len_w_x'] = df['token_begin_y'] - df['token_begin_x']
        df['len_w_y'] = df['token_end_y'] - df['token_begin_y']  # +1

        df['snippet_x_locs'] = df.apply(lambda row: [[pair for pair in [self.token_to_sent_word(token) for token in
                                                                        range(row.token_begin_x, row.token_begin_y)]]],
                                        axis=1)
        df['snippet_x_locs'] = df.snippet_x_locs.map(lambda row: row[0])
        # print(df[['snippet_x', 'snippet_y', 'snippet_x_locs']].values)
        broken_pair = df[df.snippet_x_locs.map(len) < 1]
        if not broken_pair.empty:
            print(
                f"Unable to locate first snippet >>> {df[df.snippet_x_locs.map(len) < 1][['snippet_x', 'snippet_y', 'token_begin_x', 'token_begin_y', 'loc_x', 'loc_y']].values}",
                file=sys.stderr)
            print(self.annot_text, file=sys.stderr)
            df = df[df.snippet_x_locs.map(len) > 0]

        # print(df[['snippet_x', 'snippet_y', 'token_begin_y', 'token_end_y']])
        df['snippet_y_locs'] = df.apply(lambda row: [
            [pair for pair in [self.token_to_sent_word(token) for token in range(row.token_begin_y, row.token_end_y)]]],
                                        axis=1)
        df['snippet_y_locs'] = df.snippet_y_locs.map(lambda row: row[0])
        broken_pair = df[df.snippet_y_locs.map(len) < 1]

        if not broken_pair.empty:
            print(
                f"Unable to locate second snippet >>> {df[df.snippet_y_locs.map(len) < 1][['snippet_x', 'snippet_y', 'token_begin_x', 'token_begin_y', 'token_end_y', 'loc_x', 'loc_y']].values}",
                file=sys.stderr)
            print(self.annot_text, file=sys.stderr)
            df2 = df[df.snippet_y_locs.map(len) < 1]
            _df2 = pd.DataFrame({
                'snippet_x': df2['snippet_y'].values,
                'snippet_y': df2['snippet_x'].values,
                'loc_y': df2['loc_x'].values,
                'token_begin_y': df2['token_begin_x'].values,
            })

            df2 = _df2[:]
            df2['loc_x'] = df2.apply(lambda row: self.annot_text.find(row.snippet_x, row.loc_y - 3), axis=1)
            df2['token_begin_x'] = df2.loc_x.map(self.locate_token)
            # df2['loc_y'] = df2.apply(lambda row: self._find_y(row.snippet_x, row.snippet_y, row.loc_x), axis=1)
            df2['token_end_y'] = df2.apply(lambda row: self.locate_token(row.loc_y + len(row.snippet_y)),  # + 1,
                                           axis=1)  # -1
            # df2['token_begin_x'] = df2['token_begin_y']
            # df2['token_begin_y'] = df2.loc_y.map(self.locate_token)
            df2['len_w_x'] = df2['token_begin_y'] - df2['token_begin_x']
            df2['len_w_y'] = df2['token_end_y'] - df2['token_begin_y']  # +1
            df2['snippet_x_locs'] = df2.apply(
                lambda row: [[pair for pair in [self.token_to_sent_word(token) for token in
                                                range(row.token_begin_x, row.token_begin_y)]]], axis=1)
            df2['snippet_x_locs'] = df2.snippet_x_locs.map(lambda row: row[0])
            df2['snippet_y_locs'] = df2.apply(
                lambda row: [[pair for pair in [self.token_to_sent_word(token) for token in
                                                range(row.token_begin_y, row.token_end_y)]]], axis=1)
            df2['snippet_y_locs'] = df2.snippet_y_locs.map(lambda row: row[0])
            broken_pair = df2[df2.snippet_y_locs.map(len) < 1]
            if not broken_pair.empty:
                print(
                    f"Unable to locate second snippet AGAIN >>> {df2[df2.snippet_y_locs.map(len) < 1][['snippet_x', 'snippet_y', 'token_begin_x', 'token_begin_y', 'token_end_y', 'loc_x', 'loc_y']].values}",
                    file=sys.stderr)
            df = df[df.snippet_y_locs.map(len) > 0]
            df2 = df2[df2.snippet_x_locs.map(len) > 0]
            df = pd.concat([df, df2])

        # print(df[['snippet_x', 'snippet_y', 'snippet_y_locs', 'loc_x', 'loc_y']].values)
        df.drop(columns=['loc_x', 'loc_y'], inplace=True)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('2\t', end="", flush=True)

        # define a number of sentences and whether x and y are in the same sentence
        df['sentence_begin_x'] = df.snippet_x_locs.map(lambda row: row[0][0])
        df['sentence_begin_y'] = df.snippet_y_locs.map(lambda row: row[0][0])
        df['sentence_end_y'] = df.snippet_y_locs.map(lambda row: row[-1][0])
        df['number_sents_x'] = (df['sentence_begin_y'] - df['sentence_begin_x']) | 1
        df['number_sents_y'] = (df['sentence_end_y'] - df['sentence_begin_y']) | 1
        df['same_sentence'] = (df['sentence_begin_x'] == df['sentence_begin_y']).astype(int)
        df['same_paragraph'] = df.apply(
            lambda row: annot_text.find('\n', row.sentence_begin_x, row.sentence_end_y) != -1,
            axis=1).astype(int)
        df['same_paragraph'] = df['same_sentence'] | df['same_paragraph']

        # find the common syntax root of x and y
        df['common_root'] = df.apply(lambda row: [self.locate_root(row)], axis=1)

        # find its relative position in text
        # df['common_root_position'] = df.common_root.map(lambda row: self.map_to_token(row[0])) / len(annot_tokens)

        # define its fPOS
        # df['common_root_fpos'] = df.common_root.map(lambda row: self.get_postag(row)[0])

        # 1 if it is located in y
        df['root_in_y'] = df.apply(
            lambda row: self.map_to_token(row.common_root[0]) > row.token_begin_y, axis=1).astype(int)

        df.drop(columns=['common_root'], inplace=True)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('3\t', end="", flush=True)

        # find certain markers for various relations
        if self._use_markers:
            for relation in self.relations_related:
                df[relation + '_count' + '_x'] = df.snippet_x.map(lambda row: self._relation_score(relation, row))
                df[relation + '_count' + '_y'] = df.snippet_y.map(lambda row: self._relation_score(relation, row))

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('4\t', end="", flush=True)

        # get tokens
        df['tokens_x'] = df.apply(lambda row: self.get_tokens(row.token_begin_x, row.token_begin_y), axis=1)
        df['tokens_x'] = df.apply(lambda row: row.tokens_x if len(row.tokens_x) > 0 else row.snippet_x.split(), axis=1)
        df['tokens_y'] = df.apply(lambda row: self.get_tokens(row.token_begin_y, row.token_end_y - 1), axis=1)
        df['tokens_y'] = df.apply(lambda row: row.tokens_y if len(row.tokens_y) > 0 else row.snippet_y.split(), axis=1)

        # get lemmas
        df['lemmas_x'] = df.snippet_x_locs.map(self.get_lemma)
        df['lemmas_y'] = df.snippet_y_locs.map(self.get_lemma)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('5\t', end="", flush=True)

        # ratio of uppercased words
        df['upper_x'] = df.tokens_x.map(lambda row: sum(token.isupper() for token in row) / (len(row) + 1e-5))
        df['upper_y'] = df.tokens_y.map(lambda row: sum(token.isupper() for token in row) / (len(row) + 1e-5))

        # ratio of the words starting with upper case
        df['st_up_x'] = df.tokens_x.map(lambda row: sum(token[0].isupper() for token in row) / (len(row) + 1e-5))
        df['st_up_y'] = df.tokens_y.map(lambda row: sum(token[0].isupper() for token in row) / (len(row) + 1e-5))

        # whether DU starts with upper case
        df['du_st_up_x'] = df.tokens_x.map(lambda row: row[0][0].isupper()).astype(int)
        df['du_st_up_y'] = df.tokens_y.map(lambda row: row[0][0].isupper()).astype(int)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('6\t', end="", flush=True)

        # get morphology
        if self._use_morphology:
            df['morph_x'] = df.snippet_x_locs.map(self.get_morph)
            df['morph_y'] = df.snippet_y_locs.map(self.get_morph)

            # count presence and/or quantity of various language features in the whole DUs and at the beginning/end of them
            df = df.apply(lambda row: self._linguistic_features(row, tags=MORPH_FEATS), axis=1)
            df = df.apply(lambda row: self._first_and_last_pair(row), axis=1)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('7\t', end="", flush=True)

        # count various vectors similarity metrics for morphology
        if self._use_morphology:
            linknames_for_snippet_x = df[[name + '_x' for name in MORPH_FEATS]]
            linknames_for_snippet_y = df[[name + '_y' for name in MORPH_FEATS]]
            df.reset_index(inplace=True)
            df['morph_vec_x'] = pd.Series(self.columns_to_vectors_(linknames_for_snippet_x))
            df['morph_vec_y'] = pd.Series(self.columns_to_vectors_(linknames_for_snippet_y))
            df['morph_correlation'] = df[['morph_vec_x', 'morph_vec_y']].apply(
                lambda row: spatial.distance.correlation(*row), axis=1)
            df['morph_hamming'] = df[['morph_vec_x', 'morph_vec_y']].apply(lambda row: spatial.distance.hamming(*row),
                                                                           axis=1)
            df['morph_matching'] = df[['morph_vec_x', 'morph_vec_y']].apply(
                lambda row: self.get_match_between_vectors_(*row), axis=1)
            df.set_index('index', drop=True, inplace=True)
            df = df.drop(columns=['morph_vec_x', 'morph_vec_y'])

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('8\t', end="", flush=True)

        # detect discourse markers
        if self._use_markers:
            for word in self.count_words_x:
                df[word + '_count' + '_x'] = df.snippet_x.map(lambda row: self.count_marker_(word, row))

            for word in self.count_words_y:
                df[word + '_count' + '_y'] = df.snippet_y.map(lambda row: self.count_marker_(word, row))

        # count stop words in the texts
        df['stopwords_x'] = df.lemmas_x.map(self._count_stop_words)
        df['stopwords_y'] = df.lemmas_y.map(self._count_stop_words)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('9\t', end="", flush=True)

        # dummy function needed for self.vectorizer (do NOT remove)
        def dummy(x):
            return x

        df.reset_index(drop=True, inplace=True)
        tf_idf_x = self.vectorizer.transform(df['tokens_x'].map(lambda row: [_token.lower() for _token in row]))
        tf_idf_y = self.vectorizer.transform(df['tokens_y'].map(lambda row: [_token.lower() for _token in row]))
        df['cos_tf_idf_dist'] = paired_cosine_distances(tf_idf_x, tf_idf_y)
        df['ang_cos_tf_idf_sim'] = 1. - np.arccos(df['cos_tf_idf_dist']) * 2. / np.pi

        tf_idf_x = pd.DataFrame(tf_idf_x).add_prefix('tf_idf_x_')
        tf_idf_y = pd.DataFrame(tf_idf_y).add_prefix('tf_idf_y_')

        df = pd.concat([df, tf_idf_x, tf_idf_y], axis=1)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('10\t', end="", flush=True)

        # count lexical similarity
        df['bleu'] = df.apply(lambda row: self.get_bleu_score(row.lemmas_x, row.lemmas_y), axis=1)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('11\t', end="", flush=True)

        if self._use_w2v:
            # get average vector for each text
            df = self._get_vectors(df)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('12\t', end="", flush=True)

        # get relative positions in text
        df['token_begin_x'] = df['token_begin_x'] / len(annot_tokens)
        df['token_begin_y'] = df['token_begin_y'] / len(annot_tokens)
        df['token_end_y'] = df['token_end_y'] / len(annot_tokens)
        df['sentence_begin_x'] = df['sentence_begin_x'] / len(annot_sentences)
        df['sentence_begin_y'] = df['sentence_begin_y'] / len(annot_sentences)
        df['sentence_end_y'] = df['sentence_end_y'] / len(annot_sentences)

        df['snippet_x_tmp'] = df.lemmas_x.map(lambda lemmas: ' '.join(lemmas).strip())
        df['snippet_y_tmp'] = df.lemmas_y.map(lambda lemmas: ' '.join(lemmas).strip())

        df['postags_x'] = df.snippet_x_locs.map(self.get_postags)
        df['postags_y'] = df.snippet_y_locs.map(self.get_postags)

        # count sentiments
        if self._use_sentiment:
            df = self._get_sentiments(df)

        df = df.drop(columns=[
            'lemmas_x', 'lemmas_y',
            'snippet_x_locs', 'snippet_y_locs',
        ])

        if self._use_morphology:
            df = df.drop(columns=[
                'morph_x', 'morph_y',
            ])

        if self._verbose:
            print(time.time() - t)
            print('[DONE]')
            print('estimated time:', time.time() - t_final)

        return df.fillna(0.)

    def locate_token(self, start):
        for i, token in enumerate(self.annot_tokens):
            if token.begin >= start:
                return i
        return i

    def map_to_token(self, pair):
        if pair == -1:
            return -1

        sentence, word = pair
        if type(word) == list and len(word) == 1:
            word = word[0]

        return self.annot_sentences[sentence].begin + word

    def token_to_sent_word(self, token):
        for i, sentence in enumerate(self.annot_sentences):
            if sentence.begin <= token < sentence.end:
                return i, token - sentence.begin
        for i, sentence in enumerate(self.annot_sentences):
            if sentence.begin <= token + 1 < sentence.end:
                return i, token - sentence.begin
        return -1, -1

    def locate_root(self, row):
        if row.same_sentence:
            for i, wordsynt in enumerate(self.annot_syntax_dep_tree[row.sentence_begin_x]):
                if wordsynt.parent == -1:
                    return row.sentence_begin_x, [i]
        return -1

    def get_roots(self, locations):
        res = []
        for word in locations:
            parent = self.annot_syntax_dep_tree[word[0]][word[1][0]].parent
            if parent == -1:
                res.append(word)
        return res

    def locate_attached(self, row):
        res = []
        sent_begin = self.annot_sentences[row.sentence_begin_x].begin
        for i, wordsynt in enumerate(self.annot_syntax_dep_tree[row.sentence_begin_x]):
            if row.token_begin_x - sent_begin <= i < row.token_end_y - sent_begin:
                if wordsynt.parent == -1:
                    res.append(i)
        return res

    def get_tokens(self, begin, end):
        return [self.annot_tokens[i].text for i in range(begin, end)]

    def get_lemma(self, positions):
        return [self.annot_lemma[position[0]][position[1]] for position in positions]

    def get_postag(self, positions):
        if positions:
            if positions[0] == -1:
                return ['']
            result = [self.annot_postag[position[0]][position[1]] for position in positions]
            if not result:
                return ['X']
            return result
        return ['']

    def get_morph(self, positions):
        return [self.annot_morph[position[0]][position[1]] for position in positions]

    def columns_to_vectors_(self, columns):
        return [row + 1e-05 for row in np.array(columns.values.tolist())]

    def get_match_between_vectors_(self, vector1, vector2):
        return spatial.distance.hamming([k > 0.01 for k in vector1], [k > 0.01 for k in vector2])

    def _get_fpos_vectors(self, row):
        result = {}

        for header in ['VERB', 'NOUN', '', 'ADV', 'ADJ', 'ADP', 'CONJ', 'PART', 'PRON' 'NUM']:
            result[header + '_common_root'] = int(row.common_root_fpos == header)

        for header in ['VERB', '', 'NOUN', 'ADJ', 'ADV', 'ADP', 'CONJ', 'PRON', 'PART', 'NUM', 'INTJ']:
            result[header + '_common_root_att'] = int(row.common_root_att == header)

        return row.append(pd.Series(list(result.values()), index=list(result.keys())))

    @functools.lru_cache(maxsize=2048)
    def count_marker_(self, word, row):
        return bool(re.match(word, row, re.IGNORECASE))

    @functools.lru_cache(maxsize=2048)
    def locate_marker_(self, word, row):
        for m in re.finditer(word, row):
            index = m.start()
            return (index + 1.) / len(row) * 100.
        return -1.

    def _svd_tfidf_matrix(self, matrix):
        svd = TruncatedSVD(n_components=300)
        return svd.fit_transform(matrix)

    def _linguistic_features(self, row, tags):
        """ Count occurences of each feature from MORPH_FEATS and/or SYNTAX_LINKS """
        tags = MORPH_FEATS

        def get_tags_for_snippet(morph_annot, mark='_x'):
            result = dict.fromkeys(['%s%s' % (tag, mark) for tag in tags], 0)

            for record in morph_annot:
                for key, value in record.items():
                    try:
                        result['%s_%s%s' % (key, value, mark)] += 1
                    except KeyError as e:
                        if self._verbose == 2:
                            print(f"::: Did not find such key in MORPH_FEATS: {e} :::", file=sys.stderr)
                        else:
                            pass

            return result

        tags_for_snippet_x = get_tags_for_snippet(row.morph_x, '_x')
        tags_for_snippet_y = get_tags_for_snippet(row.morph_y, '_y')

        tags = dict(tags_for_snippet_x, **tags_for_snippet_y)

        return row.append(pd.Series(list(tags.values()), index=list(tags.keys())))

    def _count_stop_words(self, lemmatized_text, threshold=0):
        return len([1 for token in lemmatized_text if len(token) >= threshold and token in self.stop_words])

    def _relation_score(self, relation, row):
        return sum([1 for value in self.relations_related[relation] if value in row])

    def _postag(self, location):
        return self.annot_postag[location[0]][location[1]]

    def get_postags(self, locations):
        result = []
        for location in locations:
            result.append(self._postag(location))
        return ' '.join(result)

    def _first_postags(self, locations, n=2):
        result = []
        for location in locations[:n]:
            sent, word = location[0], location[1]
            postag = self.annot_postag[sent][word]
            result.append(postag)
        return result

    def _last_postags(self, locations, n=2):
        result = []
        for location in locations[-n:]:
            sent, word = location[0], location[1]
            postag = self.annot_postag[sent][word]
            result.append(postag)
        return result

    def _first_and_last_pair(self, row):
        def get_features_for_snippet(first_pair_text, first_pair_morph, last_pair_text, last_pair_morph, mark='_x'):
            result = {}

            for pos_combination in self.fpos_combinations:
                result['first_' + pos_combination + mark] = int(pos_combination == first_pair_morph)
                result['last_' + pos_combination + mark] = int(pos_combination == last_pair_morph)

            for key in self.pairs_words:
                if mark == key[-2:]:
                    if key[:-2] == 'first_pair':
                        for word in self.pairs_words[key]:
                            result[key[:-1] + word + mark] = int(bool(re.findall(word, first_pair_text, re.IGNORECASE)))
                    else:
                        for word in self.pairs_words[key]:
                            result[key[:-1] + word + mark] = int(bool(re.findall(word, last_pair_text, re.IGNORECASE)))

            return result

        # snippet X
        first_pair_text_x = ' '.join([token for token in row.tokens_x[:2]]).lower()
        first_pair_morph_x = '_'.join(self._first_postags(row.snippet_x_locs))

        if len(row.tokens_x) > 2:
            last_pair_text_x = ' '.join([token for token in row.tokens_x[-2:]]).lower()
            last_pair_morph_x = '_'.join(self._last_postags(row.snippet_x_locs))
        else:
            last_pair_text_x = ' '
            last_pair_morph_x = 'X'

        features_of_snippet_x = get_features_for_snippet(first_pair_text_x, first_pair_morph_x,
                                                         last_pair_text_x, last_pair_morph_x,
                                                         '_x')

        # snippet Y
        first_pair_text_y = ' '.join([token for token in row.tokens_y[:2]]).lower()
        first_pair_morph_y = '_'.join(self._first_postags(row.snippet_y_locs))

        if len(row.tokens_y) > 2:
            last_pair_text_y = ' '.join([token for token in row.tokens_y[-2:]]).lower()
            last_pair_morph_y = '_'.join(self._last_postags(row.snippet_y_locs))

        else:
            last_pair_text_y = ' '
            last_pair_morph_y = 'X'

        features_of_snippet_y = get_features_for_snippet(first_pair_text_y, first_pair_morph_y,
                                                         last_pair_text_y, last_pair_morph_y,
                                                         '_y')

        tags = dict(features_of_snippet_x, **features_of_snippet_y)

        return row.append(pd.Series(list(tags.values()), index=list(tags.keys())))

    def get_jaccard_sim(self, text1, text2):
        txt1 = set(text1)
        txt2 = set(text2)
        c = len(txt1.intersection(txt2))
        return float(c) / (len(txt1) + len(txt2) - c + 1e-05)

    def get_bleu_score(self, text1, text2):
        return bleu_score.sentence_bleu([text1], text2, weights=(0.5,))

    def get_chrf_score(self, text1, text2):
        try:
            return chrf_score.corpus_chrf([text1], [text2], min_len=2)
        except ZeroDivisionError:
            return 0.

    def _tag_postags_morph(self, locations):
        result = []
        for location in locations:
            sent, word = location[0], location[1]
            _postag = self.annot_morph[sent][word].get('fPOS')
            if _postag:
                result.append(self.annot_lemma[sent][word] + '_' + _postag)
            else:
                result.append(self.annot_lemma[sent][word])
        return result

    def _get_vectors(self, df):
        def mean_vector(lemmatized_text):
            res = list([np.zeros(self.word2vec_vector_length), ])
            for word in lemmatized_text:
                try:
                    res.append(self.word2vec_model[word])
                except KeyError:
                    second_candidate = self._synonyms_vocabulary.get(word)
                    if second_candidate:
                        res.append(self.word2vec_model[second_candidate])
                    elif self.word2vec_stopwords and ('NOUN' in word or 'VERB' in word):
                        # print(f'There is no "{word}" in vocabulary of the given model; ommited', file=sys.stderr)
                        pass

            mean = sum(np.array(res)) / (len(res) - 1 + 1e-25)
            return mean

        if not self.word2vec_stopwords:
            df.lemmas_x = df.lemmas_x.map(self._remove_stop_words)
            df.lemmas_y = df.lemmas_y.map(self._remove_stop_words)

        # Add the required UPoS postags (as in the rusvectores word2vec model's vocabulary)
        if self.word2vec_tag_required:
            df.lemmas_x = df.snippet_x_locs.map(self._tag_postags_morph)
            df.lemmas_y = df.snippet_y_locs.map(self._tag_postags_morph)

        # Make two dataframes with average vectors for x and y,
        # merge them with the original dataframe
        df_embed_x = df.lemmas_x.apply(mean_vector).values.tolist()
        df_embed_y = df.lemmas_y.apply(mean_vector).values.tolist()
        embeddings = pd.DataFrame(df_embed_x).merge(pd.DataFrame(df_embed_y), left_index=True, right_index=True)
        embeddings['cos_embed_dist'] = paired_cosine_distances(df_embed_x, df_embed_y)
        embeddings['eucl_embed_dist'] = paired_euclidean_distances(df_embed_x, df_embed_y)
        df = pd.concat([df.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)

        return df

    def _get_sentiments(self, df):
        try:
            temp = df.snippet_x.map(lambda row: self.sentiment_model.predict([row]))
        except:
            temp = df.snippet_x.map(lambda row: [{'positive': 0., 'negative': 0.}])

        for key in ['positive', 'negative']:
            df['sm_x_' + key] = temp.map(lambda row: row[0].get(key))

        try:
            temp = df.snippet_y.map(lambda row: self.sentiment_model.predict([row]))
        except:
            temp = df.snippet_y.map(lambda row: [{'positive': 0., 'negative': 0.}])

        for key in ['positive', 'negative']:
            df['sm_y_' + key] = temp.map(lambda row: row[0].get(key))

        return df


class FeaturesProcessorTokenizer(FeaturesProcessor):
    def __init__(self, model_dir_path, verbose=0):
        super(FeaturesProcessorTokenizer, self).__init__(model_dir_path=model_dir_path,
                                                         verbose=verbose,
                                                         use_w2v=False,
                                                         use_sentiment=True,
                                                         use_markers=False,
                                                         use_morphology=False)

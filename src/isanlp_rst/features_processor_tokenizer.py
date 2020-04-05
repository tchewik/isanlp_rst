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
from nltk.translate import bleu_score, chrf_score
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from utils.features_processor_variables import MORPH_FEATS, FPOS_COMBINATIONS, count_words_x, \
    count_words_y, pairs_words, relations_related
from utils.synonyms_vocabulary import synonyms_vocabulary

warnings.filterwarnings('ignore')


class FeaturesProcessor:
    CATEGORY = 'category_id'

    def __init__(self,
                 model_dir_path,
                 verbose=False):

        self.verbose = verbose
        if self.verbose:
            print("Processor initialization...\t", end="", flush=True)

        self.embed_model_path = os.path.join(model_dir_path, 'w2v', 'default', 'model.vec')
        self._synonyms_vocabulary = synonyms_vocabulary

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

        self.fpos_combinations = FPOS_COMBINATIONS

        if self.verbose:
            print('[DONE]')

    def _find_y(self, snippet_x, snippet_y, loc_x):
        result = self.annot_text.find(snippet_y, loc_x + len(snippet_x) - 1)
        if result < 1:
            result = self.annot_text.find(snippet_y)
        return result

    def __call__(self, df_, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        df = df_[:]
        self.annot_text = annot_text
        self.annot_tokens = annot_tokens
        self.annot_sentences = annot_sentences
        self.annot_lemma = annot_lemma
        self.annot_morph = annot_morph
        self.annot_postag = annot_postag
        self.annot_syntax_dep_tree = annot_syntax_dep_tree

        t, t_final = None, None
        if self.verbose:
            t = time.time()
            t_final = t
            print('1\t', end="", flush=True)

        # map discourse units to annotations
        if not 'loc_x' in df.keys():
            df['loc_x'] = df.snippet_x.map(self.annot_text.find)
        if not 'loc_y' in df.keys():
            df['loc_y'] = df.apply(lambda row: self._find_y(row.snippet_x, row.snippet_y, row.loc_x), axis=1)

        df['token_begin_x'] = df.loc_x.map(self.locate_token)
        df['token_begin_y'] = df.loc_y.map(self.locate_token)

        # ToDO: bug in ling_20 (in progress)
        df = df[df['loc_y'] != -1]

        try:
            df['token_end_y'] = df.apply(lambda row: self.locate_token(row.loc_y + len(row.snippet_y)),  # + 1,
                                         axis=1)  # -1
            df['token_end_y'] = df['token_end_y'] + (df['token_end_y'] == df['token_begin_y']) * 1
        except:
            print(f'Unable to locate second snippet >>> {(df.snippet_x.values, df.snippet_y.values)}', file=sys.stderr)
            return -1

        # length of tokens sequence
        df['len_w_x'] = df['token_begin_y'] - df['token_begin_x']
        df['len_w_y'] = df['token_end_y'] - df['token_begin_y']  # +1

        df['snippet_x_locs'] = df.apply(lambda row: [[pair for pair in [self.token_to_sent_word(token) for token in
                                                                        range(row.token_begin_x, row.token_begin_y)] if
                                                      pair]], axis=1)
        df['snippet_x_locs'] = df.snippet_x_locs.map(lambda row: row[0])
        # print(df[['snippet_x', 'snippet_y', 'snippet_x_locs']].values)
        broken_pair = df[df.snippet_x_locs.map(len) < 1]
        if not broken_pair.empty:
            print(
                f"Unable to locate first snippet >>> {df[df.snippet_x_locs.map(len) < 1][['snippet_x', 'snippet_y', 'token_begin_x', 'token_begin_y', 'loc_x', 'loc_y']].values}",
                file=sys.stderr)
            df = df[df.snippet_x_locs.map(len) > 0]

        df['snippet_y_locs'] = df.apply(lambda row: [[pair for pair in [self.token_to_sent_word(token) for token in
                                                                        range(row.token_begin_y, row.token_end_y)] if
                                                      pair]], axis=1)
        df['snippet_y_locs'] = df.snippet_y_locs.map(lambda row: row[0])
        broken_pair = df[df.snippet_y_locs.map(len) < 1]
        if not broken_pair.empty:
            print(
                f"Unable to locate second snippet >>> {df[df.snippet_y_locs.map(len) < 1][['snippet_x', 'snippet_y', 'token_begin_x', 'token_begin_y', 'token_end_y', 'loc_x', 'loc_y']].values}",
                file=sys.stderr)
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
                                                range(row.token_begin_x, row.token_begin_y)] if
                              pair]], axis=1)
            df2['snippet_x_locs'] = df2.snippet_x_locs.map(lambda row: row[0])
            df2['snippet_y_locs'] = df2.apply(
                lambda row: [[pair for pair in [self.token_to_sent_word(token) for token in
                                                range(row.token_begin_y, row.token_end_y)] if
                              pair]], axis=1)
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

        if self.verbose:
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

        # find the common syntax root of x and y
        df['common_root'] = df.apply(lambda row: [self.locate_root(row)], axis=1)

        if self.verbose:
            print(time.time() - t)
            t = time.time()
            print('3\t', end="", flush=True)

        # find certain markers for various relations
        for relation in self.relations_related:
            df[relation + '_count' + '_x'] = df.snippet_x.map(lambda row: self._relation_score(relation, row))
            df[relation + '_count' + '_y'] = df.snippet_y.map(lambda row: self._relation_score(relation, row))

        if self.verbose:
            print(time.time() - t)
            t = time.time()
            print('4\t', end="", flush=True)

        # get tokens
        df['tokens_x'] = df.apply(lambda row: self.get_tokens(row.token_begin_x, row.token_begin_y), axis=1)
        df['tokens_y'] = df.apply(lambda row: self.get_tokens(row.token_begin_y, row.token_end_y), axis=1)

        # get lemmas
        df['lemmas_x'] = df.snippet_x_locs.map(self.get_lemma)
        df['lemmas_y'] = df.snippet_y_locs.map(self.get_lemma)

        if self.verbose:
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

        if self.verbose:
            print(time.time() - t)
            t = time.time()
            print('6\t', end="", flush=True)

        # get morphology
        df['morph_x'] = df.snippet_x_locs.map(self.get_morph)
        df['morph_y'] = df.snippet_y_locs.map(self.get_morph)

        df['postags_x'] = df.snippet_x_locs.map(self.get_postags)
        df['postags_y'] = df.snippet_y_locs.map(self.get_postags)

        if self.verbose:
            print(time.time() - t)
            print('[DONE]')
            print('estimated time:', time.time() - t_final)

        return df.fillna(0.)

    def locate_token(self, start):
        for i, token in enumerate(self.annot_tokens):
            if token.begin > start:
                return i  # - 1
            elif token.begin == start:
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
        return ()

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
                        # print(f"::: Did not find such key in MORPH_FEATS: {e} :::", file=sys.stderr)
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
            sent, word = location[0], location[1][0]
            postag = self.annot_postag[sent][word]
            result.append(postag)
        return result

    def _last_postags(self, locations, n=2):
        result = []
        for location in locations[-n:]:
            sent, word = location[0], location[1][0]
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
            sent, word = location[0], location[1][0]
            _postag = self.annot_morph[sent][word].get('fPOS')
            if _postag:
                result.append(self.annot_lemma[sent][word] + '_' + _postag)
            else:
                result.append(self.annot_lemma[sent][word])
        return result

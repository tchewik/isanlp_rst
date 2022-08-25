import functools
import nltk
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import warnings
from isanlp.annotation import Span
from nltk.translate import bleu_score, chrf_score
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import paired_distances
from textblob import TextBlob
from utils.features_processor_variables import FPOS_COMBINATIONS, VOCAB_RU, VOCAB_EN

warnings.filterwarnings('ignore')


class FeaturesProcessor:
    CATEGORY = 'category_id'

    def __init__(self,
                 verbose=0,
                 language='ru',
                 use_markers=True,
                 use_morphology=True,
                 use_use=True,
                 use_sentiment=True):
        """
        Args
        verbose (int): 0 for no logging, 1 for time counts logging and 2 for all warnings
        language (str): 'ru' or 'en'
        use_markers (bool): do count vocabulary-related marker features
        use_use (bool): do vectorization with universal sentence encoder
        use_sentiment (bool): do sentiment prediction
        """

        self._verbose = verbose
        self._use_markers = use_markers
        self._use_morphology = use_morphology
        self._language = language
        self._use_use = use_use
        self._use_sentiment = use_sentiment

        if self._verbose:
            print("Processor initialization...\t", end="", flush=True)

        if self._language == 'ru':
            self._STOP_WORDS = nltk.corpus.stopwords.words('russian')
            vocabulary = VOCAB_RU

            if self._use_sentiment:
                self._get_sentiments = self._get_sentiments_dostoevsky
                from dostoevsky.models import FastTextSocialNetworkModel
                from dostoevsky.tokenization import RegexTokenizer
                self._dost_sentiment_model = FastTextSocialNetworkModel(tokenizer=RegexTokenizer())


        elif self._language == 'en':
            self._STOP_WORDS = nltk.corpus.stopwords.words('english')
            vocabulary = VOCAB_EN

            if self._use_sentiment:
                self._get_sentiments = self._get_sentiment_textblob

        self._MORPH_FEATS = vocabulary.get('morph_feats', [])
        self._COUNT_WORDS_X = vocabulary.get('count_words_x', [])
        self._COUNT_WORDS_Y = vocabulary.get('count_words_y', [])
        self._MARKERS_BY_CLASSES = vocabulary.get('markers_by_classes', [])
        self._PAIRS_MARKERS = vocabulary.get('pairs_words', [])
        self._NER_TAGS = vocabulary.get('ner_tags', [])

        self._remove_stop_words = lambda lemmatized_snippet: [word for word in lemmatized_snippet if
                                                              word not in self._STOP_WORDS]

        self._FPOS_COMBINATIONS = FPOS_COMBINATIONS

        # preprocessing functions
        self._uppercased = lambda snippet, length: sum(
            [word[0].isupper() if len(word) > 0 else False for word in snippet.split()]) / length
        self._start_with_uppercase = lambda snippet, length: sum(
            [word[0].isupper() if len(word) > 0 else False for word in snippet.split(' ')]) / length

        if self._use_use:
            import tensorflow_hub as hub
            import tensorflow_text  # tensorflow_text>=2.0.0rc0
            self._universal_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

        if self._verbose:
            print('[DONE]')

    def tokenize(self, text, start):
        """ Returns isanlp annotation of tokens in span by start position and text string
        Args:
            text (str): span text
            start (int): position of the first character in the original document
        Returns:
            list[isanlp.annotation.Token]: tokens found in the span
        """

        def get_tokens(char_start, length):
            span = Span(char_start, char_start + length)
            return [token for token in self.annot_tokens if token.overlap(span)]

        return get_tokens(start, len(text))

    def _find_y(self, snippet_x, snippet_y, loc_x):
        result = self.annot_text.find(snippet_y, loc_x + len(snippet_x) - 1)
        if result < 1:
            result = self.annot_text.find(snippet_y, loc_x + 1)
        if result < 1:
            result = loc_x + 1
        return result

    def __call__(self, df_, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        self.annot_text = annot_text
        self.annot_tokens = annot_tokens
        self.annot_sentences = annot_sentences
        self.annot_lemma = annot_lemma
        self.annot_morph = annot_morph
        self.annot_postag = annot_postag
        self.annot_syntax_dep_tree = annot_syntax_dep_tree
        # self.annot_entities = annot_entities  # Don't use for now (no success)

        df = df_.copy()
        df.snippet_x_tokens = df.apply(lambda row: self.tokenize(text=row.snippet_x, start=row.loc_x), axis=1)
        df.snippet_y_tokens = df.apply(lambda row: self.tokenize(text=row.snippet_y, start=row.loc_y), axis=1)

        t, t_final = None, None
        if self._verbose:
            t = time.time()
            t_final = t
            print('1 (Positional and length features)\t', end="", flush=True)

        # Get tokens, find location and length in raw text ###############

        df['snippet_x_tokens'] = df.apply(lambda row: self.tokenize(text=row.snippet_x, start=row.loc_x), axis=1)
        df['snippet_y_tokens'] = df.apply(lambda row: self.tokenize(text=row.snippet_y, start=row.loc_y), axis=1)

        df['token_begin_x'] = df.snippet_x_tokens.map(lambda row: self.locate_token(row[0]))
        df['token_begin_y'] = df.snippet_y_tokens.map(lambda row: self.locate_token(row[0]))

        df['len_w_x'] = df.snippet_x_tokens.map(len)
        df['len_w_y'] = df.snippet_y_tokens.map(len)

        df['token_end_x'] = df.token_begin_x + df.len_w_x
        df['token_end_y'] = df.token_begin_y + df.len_w_y

        # Find them in the sentenced text ###############

        df['snippet_x_locs'] = df.apply(
            lambda row: list(self.get_sent_words_locations(span_begin=row.token_begin_x, span_end=row.token_end_x)),
            axis=1)
        df['snippet_y_locs'] = df.apply(
            lambda row: list(self.get_sent_words_locations(span_begin=row.token_begin_y, span_end=row.token_end_y)),
            axis=1)

        df['sentence_begin_x'] = df.snippet_x_locs.map(lambda row: row[0][0])
        df['sentence_begin_y'] = df.snippet_y_locs.map(lambda row: row[0][0])
        df['sentence_end_y'] = df.snippet_y_locs.map(lambda row: row[-1][0])
        df['number_sents_x'] = (df['sentence_begin_y'] - df['sentence_begin_x']) | 1
        df['number_sents_y'] = (df['sentence_end_y'] - df['sentence_begin_y']) | 1

        # Get relative positions ###############
        df['token_begin_x'] = df['token_begin_x'] / len(annot_tokens)
        df['token_begin_y'] = df['token_begin_y'] / len(annot_tokens)
        df['token_end_y'] = df['token_end_y'] / len(annot_tokens)
        df['sentence_begin_relative_x'] = df['sentence_begin_x'] / len(annot_sentences)
        df['sentence_begin_relative_y'] = df['sentence_begin_y'] / len(annot_sentences)
        df['sentence_end_relative_y'] = df['sentence_end_y'] / len(annot_sentences)

        if self._verbose:
            print('time:', time.time() - t)
            t = time.time()
            print('2 (Simple string features)\t', end="", flush=True)

        ##### Simple string-based features ##########

        # ratio of uppercased words
        df['upper_ratio_x'] = df.snippet_x_tokens.map(
            lambda row: sum(token.text.isupper() for token in row) / (len(row) + 1e-5))
        df['upper_ratio_y'] = df.snippet_y_tokens.map(
            lambda row: sum(token.text.isupper() for token in row) / (len(row) + 1e-5))

        # ratio of the words starting with upper case
        df['first_up_ratio_x'] = df.snippet_x_tokens.map(
            lambda row: sum(token.text[0].isupper() for token in row) / (len(row) + 1e-5))
        df['first_up_ratio_y'] = df.snippet_y_tokens.map(
            lambda row: sum(token.text[0].isupper() for token in row) / (len(row) + 1e-5))

        # whether DU starts with upper case
        df['first_up_x'] = df.snippet_x_tokens.map(lambda row: row[0].text[0].isupper()).astype(int)
        df['first_up_y'] = df.snippet_y_tokens.map(lambda row: row[0].text[0].isupper()).astype(int)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('3 (Granularity and syntax features)\t', end="", flush=True)

        # Granularity features: define a number of sentences and whether x and y are in the same sentence ##########

        df['same_sentence'] = (df['sentence_begin_x'] == df['sentence_end_y']).astype(int)

        df['same_paragraph'] = df.apply(
            lambda row: annot_text.find('\n', row.sentence_begin_x, row.sentence_end_y) != -1, axis=1).astype(int)
        df['same_paragraph'] = df['same_sentence'] | df['same_paragraph']

        at_paragraph_start = lambda row: int(
            row[0].begin == 0 or '\n' in self.annot_text[row[0].begin - 4: row[0].begin])
        df['at_paragraph_start_x'] = df.snippet_x_tokens.map(at_paragraph_start)
        df['at_paragraph_start_y'] = df.snippet_y_tokens.map(at_paragraph_start)

        df['at_sentence_start_x'] = df.at_paragraph_start_x | df.snippet_x_locs.map(lambda row: int(row[0][1] == 0))
        df['at_sentence_start_y'] = df.at_paragraph_start_y | df.snippet_y_locs.map(lambda row: int(row[0][1] == 0))

        # Syntax features ##########

        # find the common syntax root of x and y
        df['common_root'] = df.apply(lambda row: self.locate_root(row), axis=1)

        # 1 if it is located in y
        df['root_in_y'] = df.apply(
            lambda row: self.map_to_token(row.common_root) >= row.token_begin_y, axis=1).astype(int)

        df.drop(columns=['common_root'], inplace=True)

        if self._verbose:
            print('time:', time.time() - t)
            t = time.time()
            print('4 (Discourse markers)\t', end="", flush=True)

        if self._use_markers:
            # find certain markers sets for various relations
            for relation in self._MARKERS_BY_CLASSES:
                df[relation + '_count' + '_x'] = df.snippet_x.map(lambda row: self._relation_score(relation, row))
                df[relation + '_count' + '_y'] = df.snippet_y.map(lambda row: self._relation_score(relation, row))

            # detect discourse markers
            for word in self._COUNT_WORDS_X:
                df[word.pattern + '_count' + '_x'] = df.snippet_x.map(lambda row: self.count_marker_(word, row))

            for word in self._COUNT_WORDS_Y:
                df[word.pattern + '_count' + '_y'] = df.snippet_y.map(lambda row: self.count_marker_(word, row))

        # get lemmas
        df['lemmas_x'] = df.snippet_x_locs.map(self.get_lemma)
        df['lemmas_y'] = df.snippet_y_locs.map(self.get_lemma)

        # count stop words in the texts
        df['stopwords_x'] = df.lemmas_x.map(self._count_stop_words)
        df['stopwords_y'] = df.lemmas_y.map(self._count_stop_words)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('5 (Morphological features)\t', end="", flush=True)

        # Morphological features ##########

        # count number of verbs
        df['Verb_number_x'] = df.snippet_x_locs.map(lambda row: sum([pos == 'VERB' for pos in self.get_postag(row)]))
        df['Verb_number_y'] = df.snippet_y_locs.map(lambda row: sum([pos == 'VERB' for pos in self.get_postag(row)]))

        # count lexical similarity
        df['bleu'] = df.apply(lambda row: self.get_bleu_score(row.lemmas_x, row.lemmas_y), axis=1)

        if self._use_morphology:
            df['morph_x'] = df.snippet_x_locs.map(self.get_morph)
            df['morph_y'] = df.snippet_y_locs.map(self.get_morph)

            # count presence and/or quantity of various language features in the whole DUs
            # and at the beginning/end of them
            df = df.apply(lambda row: self._linguistic_features(row), axis=1)
            df = df.apply(lambda row: self._first_and_last_pair(row), axis=1)

            # count various vectors similarity metrics for morphology
            linknames_for_snippet_x = df[[name + '_x' for name in self._MORPH_FEATS]].values
            linknames_for_snippet_y = df[[name + '_y' for name in self._MORPH_FEATS]].values
            df.reset_index(inplace=True)

            df['morph_correlation'] = paired_distances(linknames_for_snippet_x, linknames_for_snippet_y,
                                                       metric=spatial.distance.hamming)
            df['morph_cos'] = paired_distances(linknames_for_snippet_x, linknames_for_snippet_y,
                                               metric='cosine')
            df['morph_matching'] = paired_distances(linknames_for_snippet_x > 0.01,
                                                    linknames_for_snippet_y > 0.01,
                                                    metric=spatial.distance.hamming)
            df.set_index('index', drop=True, inplace=True)

        # NER features ##########
        ### Deprecated! Not useful.
        if False:
            if self._verbose:
                print(time.time() - t)
                t = time.time()
                print(' (NER features)\t', end="", flush=True)

            # form matrices with columns corresponding to self.vocabulary.get('ner_tags')
            ner_x = np.stack(df.snippet_x_tokens.map(self._get_ner_features).values)
            ner_y = np.stack(df.snippet_y_tokens.map(self._get_ner_features).values)
            df['ner_matching'] = paired_distances(ner_x, ner_y, metric=spatial.distance.hamming)
            for i, ner_tag in enumerate(self._NER_TAGS):
                df[ner_tag + '_x'] = ner_x[:, i]
                df[ner_tag + '_y'] = ner_y[:, i]

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('6 (USE features)\t', end="", flush=True)

        if self._use_use:
            df = self._vectorize_with_use(df)

        if self._verbose:
            print(time.time() - t)
            t = time.time()
            print('7 (Sentiment features)\t', end="", flush=True)

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

    def locate_token(self, token):
        """ Finds the token index in the annotation
        Args:
            token (isanlp.annotation.Token): the token to find
        Returns:
            (int): index in the raw text
        """
        for i, _token in enumerate(self.annot_tokens):
            if token == _token:
                return i

    def get_sent_words_locations(self, span_begin, span_end):
        """ Generator for the (sentence, word) position for each token in span
        Args:
            span_begin (int): absolute position of the first span token in text
            span_end (int): absolute position of the last span token in text + 1
        """

        for current_token in range(span_begin, span_end):
            for sidx, sentence in enumerate(self.annot_sentences):
                if current_token >= sentence.end:
                    # Straight to the next sentence
                    continue

                yield (sidx, current_token - sentence.begin)
                break

    def locate_root(self, row):
        """ If two spans are in the same sentence,
            returns the root position as (sentence_number, word_number)
            Args:
                row (pd.Series): a raw in the data DataFrame
            Returns:
                (int, int): Sentence number, word number
            """

        if row.same_sentence:
            for i, wordsynt in enumerate(self.annot_syntax_dep_tree[row.sentence_begin_x]):
                if wordsynt.parent == -1:
                    return (row.sentence_begin_x, i)

        return -1, -1

    def map_to_token(self, pair):
        """ Finds absolute index for the token by (sentence, word) pair
        Args:
            pair (list[int, int]): sentence number, word number
        Returns:
            (int): absolute position of the token
        """
        if pair == (-1, -1):
            return -1

        sentence, word = pair
        return self.annot_sentences[sentence].begin + word

    def _get_roots(self, locations):
        """ Return the syntactic parents for all the presented (sentence, word) locations
            [NOT USED SINCE 2.0]
        Args:
            locations (list[int, int]): sentence number, word number
        Returns:
            list[int]: head indexes for each word, word itself if it's the root
        """
        res = []
        for word in locations:
            parent = self.annot_syntax_dep_tree[word[0]][word[1][0]].parent
            if parent == -1:
                res.append(word)
        return res

    def get_lemma(self, locations):
        """ Return the lemmas for all the presented (sentence, word) locations
        Args:
            locations (list[int, int]): sentence number, word number
        Returns:
            list[str]: lemmas for each word
        """
        return [self.annot_lemma[location[0]][location[1]] for location in locations]

    def get_postag(self, locations):
        """ Return the postags for each (sentence, word) location pair
        Args:
            locations (list[int, int]): sentence number, word number
        Returns:
            list[str]: postags found in text span, in appearance order
        """
        if not locations or locations[0] == -1:
            return ['']

        result = [self.annot_postag[location[0]][location[1]] for location in locations] or ['X']
        return result

    def get_morph(self, locations):
        """ Return the postags for each (sentence, word) location pair
        Args:
            locations (list[int, int]): sentence number, word number
        Returns:
            list[dict]: morphological features for each word
        """
        return [self.annot_morph[location[0]][location[1]] for location in locations]

    def _get_ner_features(self, tokens):
        """ Finds how many named entities of each class is in the span
        Args:
            tokens (isanlp.annotation.Token): tokens found in the text span
        Returns:
            list[int]: each element of the list represents the presence of one of the NER features (ORG, PER, etc.)
        """
        snippet_span = Span(begin=tokens[0].begin, end=tokens[-1].end)
        result = dict(zip(self._NER_TAGS, [0] * len(self._NER_TAGS)))
        for entity in self.annot_entities:
            if entity.overlap(snippet_span):
                result[entity.tag] = 1
        return list(result.values())

    def _get_fpos_vectors(self, row):
        """ Deprecated """

        result = {}

        for header in ['VERB', 'NOUN', '', 'ADV', 'ADJ', 'ADP', 'CONJ', 'PART', 'PRON' 'NUM']:
            result[header + '_common_root'] = int(row.common_root_fpos == header)

        for header in ['VERB', '', 'NOUN', 'ADJ', 'ADV', 'ADP', 'CONJ', 'PRON', 'PART', 'NUM', 'INTJ']:
            result[header + '_common_root_att'] = int(row.common_root_att == header)

        return row.append(pd.Series(list(result.values()), index=list(result.keys())))

    @functools.lru_cache(maxsize=2048)
    def count_marker_(self, word, row):
        return bool(word.search(row))

    @functools.lru_cache(maxsize=2048)
    def locate_marker_(self, word, row):
        for m in re.finditer(word, row):
            index = m.start()
            return (index + 1.) / len(row) * 100.
        return -1.

    def _linguistic_features(self, row):
        """ Count occurences of each feature from _MORPH_FEATS """

        def get_tags_for_snippet(morph_annot, mark='_x'):
            result = dict.fromkeys(['%s%s' % (tag, mark) for tag in self._MORPH_FEATS], 0)

            for record in morph_annot:
                for key, value in record.items():
                    try:
                        result['%s_%s%s' % (key, value, mark)] += 1
                    except KeyError as e:
                        if self._verbose == 2:
                            print(f"::: Did not find such key in _MORPH_FEATS: {e} :::", file=sys.stderr)

            return result

        tags_for_snippet_x = get_tags_for_snippet(row.morph_x, '_x')
        tags_for_snippet_y = get_tags_for_snippet(row.morph_y, '_y')

        tags = dict(tags_for_snippet_x, **tags_for_snippet_y)

        return row.append(pd.Series(list(tags.values()), index=list(tags.keys())))

    def _count_stop_words(self, lemmatized_text, threshold=0):
        return len([1 for token in lemmatized_text if len(token) >= threshold and token in self._STOP_WORDS])

    def _relation_score(self, relation, row):
        return sum([1 for value in self._MARKERS_BY_CLASSES[relation] if value.search(row)])

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
        """ Computes features related to the first tokens pair & last tokens pair
            for the left and right discourse units.
            These are first-pair-POS, last-pair-POS, first-pair-connector, last-pair-connector;
            one column = one feature (first_VERB_x, first_because_x, etc.)

            Args:
                row (pd.Series): row in the data DataFrame
            Returns:
                (pd.DataFrame): the extended DataFrame with all the computed features
             """

        def get_features_for_snippet(first_pair_text, first_pair_morph, last_pair_text, last_pair_morph, mark='_x'):
            result = {}

            for pos_combination in self._FPOS_COMBINATIONS:
                result['first_' + pos_combination + mark] = int(pos_combination == first_pair_morph)
                result['last_' + pos_combination + mark] = int(pos_combination == last_pair_morph)

            for key in self._PAIRS_MARKERS:
                if mark == key[-2:]:
                    if key[:-2] == 'first_pair':
                        for word in self._PAIRS_MARKERS[key]:
                            try:
                                result[key[:-1] + word.pattern + mark] = int(bool(word.search(first_pair_text)))
                            except:
                                if self._verbose == 2:
                                    print("Broken marker:", word.pattern)
                    else:
                        for word in self._PAIRS_MARKERS[key]:
                            try:
                                result[key[:-1] + word.pattern + mark] = int(bool(word.search(last_pair_text)))
                            except:
                                if self._verbose == 2:
                                    print("Broken marker:", word.pattern)

            return result

        # snippet X
        first_pair_text_x = ' '.join([token.text for token in row.snippet_x_tokens[:2]]).lower()
        first_pair_morph_x = '_'.join(self._first_postags(row.snippet_x_locs))

        if len(row.snippet_x_tokens) > 2:
            last_pair_text_x = ' '.join([token.text for token in row.snippet_x_tokens[-2:]]).lower()
            last_pair_morph_x = '_'.join(self._last_postags(row.snippet_x_locs))
        else:
            last_pair_text_x = ' '
            last_pair_morph_x = 'X'

        features_of_snippet_x = get_features_for_snippet(first_pair_text_x, first_pair_morph_x,
                                                         last_pair_text_x, last_pair_morph_x,
                                                         '_x')

        # snippet Y
        first_pair_text_y = ' '.join([token.text for token in row.snippet_y_tokens[:2]]).lower()
        first_pair_morph_y = '_'.join(self._first_postags(row.snippet_y_locs))

        if len(row.snippet_y_tokens) > 2:
            last_pair_text_y = ' '.join([token.text for token in row.snippet_y_tokens[-2:]]).lower()
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
        embeddings['cos_embed_dist'] = paired_distances(df_embed_x, df_embed_y, metric='cosine')
        embeddings['eucl_embed_dist'] = paired_distances(df_embed_x, df_embed_y, metric='euclidian')
        df = pd.concat([df.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)

        return df

    def _get_sentiments_dostoevsky(self, df):
        """ Use dostoevsky library for fast sentiment prediction in Russian

        Args:
            df (pandas.DataFrame): the whole dataframe

        Returns:
            df (pandas.DataFrame): the dataframe with additional columns:
                                   sm_x_positive, sm_x_negative, sm_y_positive, sm_y_negative
        """
        for part in ('x', 'y'):
            try:
                temp = df[f'snippet_{part}'].map(lambda row: self._dost_sentiment_model.predict([row]))
            except:
                temp = df[f'snippet_{part}'].map(lambda row: [{}])

            for key in ['positive', 'negative']:
                df[f'sm_{part}_' + key] = temp.map(lambda row: row[0].get(key, 0.))

        return df

    def _get_sentiment_textblob(self, df):
        """ Use textblob library for fast sentiment prediction in English

        Args:
            df (pandas.DataFrame): the whole dataframe

        Returns:
            df (pandas.DataFrame): the dataframe with additional columns:
                                   sm_x_positive, sm_x_negative, sm_y_positive, sm_y_negative
        """

        x_blob = df.snippet_x_tokens.map(lambda row: TextBlob(' '.join([token.text for token in row])).sentiment)
        y_blob = df.snippet_y_tokens.map(lambda row: TextBlob(' '.join([token.text for token in row])).sentiment)

        df['x_polarity'] = x_blob.map(lambda row: row.polarity)
        df['x_subjectivity'] = x_blob.map(lambda row: row.subjectivity)

        df['y_polarity'] = y_blob.map(lambda row: row.polarity)
        df['y_subjectivity'] = y_blob.map(lambda row: row.subjectivity)

        return df

    def _vectorize_with_use(self, df):

        x_encoded = self._universal_encoder(df.snippet_x.values.tolist())
        y_encoded = self._universal_encoder(df.snippet_y.values.tolist())
        df['use_cossim'] = 1. - paired_distances(x_encoded, y_encoded, metric='cosine')
        df['use_eucl'] = paired_distances(x_encoded, y_encoded, metric=spatial.distance.euclidean)
        df['use_canberra'] = paired_distances(x_encoded, y_encoded, metric=spatial.distance.canberra)

        return pd.concat([df.reset_index(), pd.DataFrame(np.hstack([x_encoded, y_encoded])).add_prefix('use_')], axis=1)


class FeaturesProcessorTokenizer(FeaturesProcessor):
    def __init__(self, verbose=0):
        super(FeaturesProcessorTokenizer, self).__init__(verbose=verbose,
                                                         use_markers=False,
                                                         use_morphology=False,
                                                         use_use=False,
                                                         use_sentiment=False)

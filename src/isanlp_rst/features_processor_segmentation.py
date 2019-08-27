import os
import pickle

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from text_reader import TextReader


class FeaturesProcessorSegmentation:
    def __init__(self, model_path):
        self.directory = 'segmentation'
        self.TextReader = TextReader()
        self.columns = ['left_lemma', 'left_pos', 'left_link', 'lemma', 'pos', 'link', 'right_lemma', 'right_pos',
                        'right_link']

        with open(os.path.join(model_path, self.directory, 'category_features.pckl'), 'rb') as f:
            self.category_features = pickle.load(f)

        with open(os.path.join(model_path, self.directory, 'vectorizer.pckl'), 'rb') as f:
            self.vectorizer = pickle.load(f)

        self._embedder = Word2Vec.load(os.path.join(model_path, 'w2v', 'segmentator', 'model2_tokenized'))

    def _get_embeddings(self, word):
        try:
            return self._embedder[word.lower()]
        except KeyError:
            return np.zeros(self._embedder.vector_size)

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_postag, annot_lemma, annot_syntax_dep_tree):
        triplets = self._extract_triplets(self.TextReader(annot_text),
                                          annot_tokens,
                                          annot_sentences,
                                          annot_lemma,
                                          annot_postag,
                                          annot_syntax_dep_tree)
        features = pd.DataFrame(triplets, columns=self.columns)

        one_hot_features = self.vectorizer.transform(features[self.category_features].to_dict(orient='records'))
        embed_left = np.stack(features.left_lemma.map(self._get_embeddings).values)
        embed_lemma = np.stack(features.lemma.map(self._get_embeddings).values)
        embed_right = np.stack(features.right_lemma.map(self._get_embeddings).values)
        return [embed_lemma, embed_left, embed_right, one_hot_features]

    def _extract_triplets(self, annot_text, annot_tokens, annot_sentences, annot_postag, annot_lemma,
                          annot_syntax_dep_tree):
        triplets = []

        for sentence in range(len(annot_sentences)):
            for token in range(annot_sentences[sentence].begin, annot_sentences[sentence].end):
                if token == annot_sentences[sentence].begin:
                    if token > 0:
                        left_neighbour = (  # annot_tokens[token-1].text,
                            annot_lemma[sentence - 1][-1],
                            annot_postag[sentence - 1][-1],
                            annot_syntax_dep_tree[sentence - 1][-1].link_name)
                        original_text = annot_text[annot_tokens[token - 1].begin:annot_tokens[token].end]
                    else:
                        left_neighbour = ('', '', '')
                        original_text = annot_text[annot_tokens[token].begin:annot_tokens[token].end]
                else:
                    left_neighbour = (  # annot_tokens[token-1].text,
                        annot_lemma[sentence][token - 1 - annot_sentences[sentence].begin],
                        annot_postag[sentence][token - 1 - annot_sentences[sentence].begin],
                        annot_syntax_dep_tree[sentence][
                            token - 1 - annot_sentences[sentence].begin].link_name)
                    original_text = annot_text[annot_tokens[token - 1].begin:annot_tokens[token].end]

                token_itself = (  # annot_tokens[token].text,
                    annot_lemma[sentence][token - annot_sentences[sentence].begin],
                    annot_postag[sentence][token - annot_sentences[sentence].begin],
                    annot_syntax_dep_tree[sentence][token - annot_sentences[sentence].begin].link_name)

                if token == annot_sentences[sentence].end - 1:
                    if token + 1 < len(annot_tokens):
                        right_neighbour = (  # annot_tokens[token+1].text,
                            annot_lemma[sentence + 1][0],
                            annot_postag[sentence + 1][0],
                            annot_syntax_dep_tree[sentence + 1][0].link_name)
                        original_text += annot_text[annot_tokens[token].end:annot_tokens[token + 1].end]
                    else:
                        right_neighbour = ('', '', '')
                else:
                    right_neighbour = (  # annot_tokens[token+1].text,
                        annot_lemma[sentence][token + 1 - annot_sentences[sentence].begin],
                        annot_postag[sentence][token + 1 - annot_sentences[sentence].begin],
                        annot_syntax_dep_tree[sentence][
                            token + 1 - annot_sentences[sentence].begin].link_name)
                    original_text += annot_text[annot_tokens[token].end:annot_tokens[token + 1].end]

                triplets.append(left_neighbour + token_itself + right_neighbour)

        return triplets

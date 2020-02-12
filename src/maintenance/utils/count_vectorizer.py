from sklearn.feature_extraction.text import CountVectorizer

class MyCountVectorizer(CountVectorizer):
    @staticmethod
    def dummy(x):
        return x

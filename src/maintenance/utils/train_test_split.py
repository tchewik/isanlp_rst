import glob
import os
import random


def split_rstreebank(path, dev_ratio=.15, test_ratio=.1, seed=45):
    files = glob.glob(os.path.join(path, '*.edus'))

    def get_genre(files, substring):
        genre = [file for file in files if substring in file]
        random.Random(seed).shuffle(genre)
        dev = genre[::int(1. / dev_ratio)]
        test = [file for file in genre if not file in dev][::int((1. - dev_ratio) / test_ratio)]
        train = [file for file in genre if not file in dev + test]
        print(
            f'{substring} in train: {len(train) / int(len(files) * (1. - dev_ratio - test_ratio))},\tin dev: {len(dev) / int((len(files) * dev_ratio))},\tin test: {len(test) / int((len(files) * test_ratio))}')
        return train, dev, test

    train, dev, test = [], [], []

    for genre in ["news", "ling", "comp", "blog"]:
        samples = get_genre(files, genre)
        train += samples[0]
        dev += samples[1]
        test += samples[2]

    return train, dev, test


def split_essays(path, dev_ratio=.15, test_ratio=.1, seed=45):
    files = glob.glob(os.path.join(path, '*.edus'))

    def get_genre(files, substring):
        genre = [file for file in files if substring in file]
        random.Random(seed).shuffle(genre)
        dev = genre[::int(1. / dev_ratio)]
        test = [file for file in genre if not file in dev][::int((1. - dev_ratio) / test_ratio)]
        train = [file for file in genre if not file in dev + test]
        print(
            f'{substring} in train: {len(train) / int(len(files) * (1. - dev_ratio - test_ratio))},\tin dev: {len(dev) / int((len(files) * dev_ratio))},\tin test: {len(test) / int((len(files) * test_ratio))}')
        return train, dev, test

    train, dev, test = [], [], []

    for genre in ["depression.", "healthy."]:
        samples = get_genre(files, genre)
        train += samples[0]
        dev += samples[1]
        test += samples[2]

    return train, dev, test

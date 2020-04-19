import os
import glob
import random


def split_data(path, ratio=0.2, seed=45):
    #files = sorted(glob.glob(os.path.join(path, '*.edus')), key=lambda s: int(s.split('.')[-2][-1]))
    files = glob.glob(os.path.join(path, '*.edus'))
    
    def get_genre(files, substring):
        genre = [file for file in files if substring in file]
        random.Random(seed).shuffle(genre)
        test = genre[::int(1./ratio)]
        train = [file for file in genre if not file in test]
        print(f'{substring} in train: {len(train) / int(len(files) * (1. - ratio))},\tin test: {len(test) / int((len(files) * ratio))}')
        return train, test
    
    news = get_genre(files, "news")
    ling = get_genre(files, "ling")
    comp = get_genre(files, "comp")
    blog = get_genre(files, "blog")
    
    train = news[0] + ling[0] + comp[0] + blog[0]
    test = news[1] + ling[1] + comp[1] + blog[1]

    return train, test

def split_train_dev_test(path, dev_ratio=.15, test_ratio=.1, seed=45):
    files = glob.glob(os.path.join(path, '*.edus'))
    
    def get_genre(files, substring):
        genre = [file for file in files if substring in file]
        random.Random(seed).shuffle(genre)
        dev = genre[::int(1./dev_ratio)]
        test = [file for file in genre if not file in dev][::int((1. - dev_ratio)/test_ratio)]
        train = [file for file in genre if not file in dev + test]
        print(f'{substring} in train: {len(train) / int(len(files) * (1. - dev_ratio - test_ratio))},\tin dev: {len(dev) / int((len(files) * dev_ratio))},\tin test: {len(test) / int((len(files) * test_ratio))}')
        return train, dev, test
    
    train, dev, test = [], [], []
    
    for genre in ["news", "ling", "comp", "blog"]:
        samples = get_genre(files, genre)
        train += samples[0]
        dev += samples[1]
        test += samples[2]

    return train, dev, test

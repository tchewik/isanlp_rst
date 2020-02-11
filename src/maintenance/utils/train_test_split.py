import os
import glob
import random


def split_data(path, ratio=0.2, seed=42):
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

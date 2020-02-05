import os
import glob


def split_data(path, ratio=0.2):
    files = sorted(glob.glob(os.path.join(path, '*.edus')), key=lambda s: int(s.split('.')[-2][-1]))
    test = files[::int(1./ratio)]
    train = [file for file in files if not file in test]
    
    print(f'news in train: {len([file for file in train if "news" in file]) / len(train)},\tin test: {len([file for file in test if "news" in file]) / len(test)}')
    print(f'ling in train: {len([file for file in train if "ling" in file]) / len(train)},\tin test: {len([file for file in test if "ling" in file]) / len(test)}')
    print(f'comp in train: {len([file for file in train if "comp" in file]) / len(train)},\tin test: {len([file for file in test if "comp" in file]) / len(test)}')

    return train, test

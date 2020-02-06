import pandas as pd


text_html_map = {
    r'\n': r' ',
    r'&gt;': r'>',
    r'&lt;': r'<',
    r'&amp;': r'&',
    r'&quot;': r'"',
    r'&ndash;': r'–',
    r'##### ': r'',
    r'\\\\\\\\': r'\\',
    r'  ': r' ',
    r'——': r'-',
    r'—': r'-',
    r'/': r'',
    r'\^': r'',
    r'^': r'',
    r'±': r'+'
}

def read_edus(filename):
    edus = []
    with open(filename + '.edus', 'r') as f:
        for line in f.readlines():
            edu = str(line.strip())
            for key, value in text_html_map.items():
                edu = edu.replace(key, value)
            edus.append(edu)
    return edus

def read_annotation(filename):
    annot = pd.read_pickle(filename + '.annot.pkl')
    for key, value in text_html_map.items():
        annot['text'] = annot['text'].replace(key, value)
    return annot

def read_gold(filename, features=False):
    if features:
        return pd.read_pickle(filename + '.gold.pkl')
    else:   
        df = pd.read_json(filename + '.json')
        for key in text_html_map.keys():
            df['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
            df['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)

        return df

def read_negative(filename, features=False):
    if features:
        return pd.read_pickle(filename + '.neg.features')
    return pd.read_json(filename + '.json.neg')
    

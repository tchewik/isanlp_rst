import pandas as pd


text_html_map = {
    r'\n': r' ',
    #r'&gt;': r'>',
    #r'&lt;': r'<',
    r'<': r' менее ',
    r'&lt;': r' менее ',
    r'>': r' более ',
    r'&gt;': r' более ',
    r'„': '"',
    r'&amp;': r'&',
    r'&quot;': r'"',
    r'&ndash;': r'–',
    r' & ': r' and ',  #
    r'&id=': r'_id=',
    r'&': r'_',
    r'<->': r'↔',
    r'##### ': r'',
    r'\\\\\\\\': r'\\',
    r'  ': r' ',
    r'——': r'-',
    r'—': r'-',
    #r'/': r'',
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
            
        df = df[df['snippet_x'].map(len) > 0]
        df = df[df['snippet_y'].map(len) > 0]

        return df

def read_negative(filename, features=False):
    if features:
        return pd.read_pickle(filename + '.neg.features')
    return pd.read_json(filename + '.json.neg')
    
def prepare_text(text):
    text = text.replace('  \n', '#####')
    text = text.replace(' \n', '#####')
    text = text + '#####'
    text = text.replace('#####', '\n')
    text_html_map = {
        '\n': r' ',
        '&gt;': r'>',
        '&lt;': r'<',
        '&amp;': r'&',
        '&quot;': r'"',
        '&ndash;': r'–',
        '##### ': r'',
        '\\\\\\\\': r'\\',
        '<': ' менее ',
        '&lt;': ' менее ',
        r'>': r' более ',
        r'&gt;': r' более ',
        r'„': '"',
        r'&amp;': r'&',
        r'&quot;': r'"',
        r'&ndash;': r'–',
        ' & ': ' and ',  #
        '&id=': r'_id=',
        '&': '_',
        '——': r'-',
        '—': r'-',
        #'/': r'',
        '\^': r'',
        '^': r'',
        '±': r'+',
        'y': r'у',
        'xc': r'хс',
        'x': r'х',
    }
    for key in text_html_map.keys():
        text = text.replace(key, text_html_map[key])
        
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text    


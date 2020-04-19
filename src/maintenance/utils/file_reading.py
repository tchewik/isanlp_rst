import pandas as pd


text_html_map = {
    r'\n': r' ',
    r'<': r' менее ',
    r'&lt;': r' менее ',
    r'>': r' более ',
    r'&gt;': r' более ',
    r'„': '"',
    r'&amp;': r'&',
    r'&quot;': r'"',
    r'&ndash;': r'–',
    r'&ouml;': r'o',
    r'&hellip;': r'...',
    r'&eacute;': r'e',
    r'&aacute;': r'a',
    r'&rsquo;': r"'",
    r'&lsquo;': r"'",
    r' & ': r' and ',  #
    r'&id=': r'_id=',
    r'<->': r'↔',
    r'##### ': r'',
    r'\\\\\\\\': r'\\',
    r'  ': r' ',
    r'——': r'-',
    r'—': r'-',
    r'\^': r'',
    r'^': r'',
    r'±': r'+',
    r'x': r'х',
    r'y': r'у',
    r'companу': r'company',
    r'kasperskу': r'kaspersky',
}

second_map = {
    r'&': r'_',
}

def split_bibliography(text_edus):
    for value in (' 1\\.', ' 2\\.', ' 3\\.', ' 4\\.', ' 5\\.', ' 6\\.', ' 7\\.', ' 8\\.', ' 9\\.', ' 10\\.',
                  ' 11\\.', ' 12\\.', ' 13\\.', ' 14\\.', ' 15\\.', ' 16\\.', ' 17\\.', ' 18\\.', ' 19\\.', ' 20\\.',
                  ' Исследование ', ' В статье', ' В результате', ' Статья посвящена',
                  ' Отмечено,', ' Показано, ', ' Выявлено,', ' Ключевые', ' Аннотация', ' На примере',
                  ' В частности,', ' В работе', ' Особое', ' Автор ', ' В отношении', ' Было', ' Среди', ' Выбор', ' В качестве', ' Такие',
                 ):
        text_edus = text_edus.replace(value, '\n' + value[1:])
        
    while '\n\n' in text_edus:
        text_edus = text_edus.replace('\n\n', '\n')
        
    return text_edus.strip()

def read_edus(filename):
    edus = []
    with open(filename + '.edus', 'r') as f:
        edus_text = f.read()
        edus_text = edus_text.replace('#####', '\n')
        edus_text = split_bibliography(edus_text)
    
        for line in edus_text.split('\n'):
            edu = str(line.strip())
            for key, value in text_html_map.items():
                edu = edu.replace(key, value)
            for key, value in second_map.items():
                edu = edu.replace(key, value)
            edus.append(edu)
    return edus

def read_annotation(filename):
    annot = pd.read_pickle(filename + '.annot.pkl')
    annot['text'] = _prepare_text(annot['text'])
    return annot

def read_gold(filename, features=False):
    if features:
        return pd.read_pickle(filename + '.gold.pkl')
    else:   
        df = pd.read_json(filename + '.json')
        for key in text_html_map.keys():
            df['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
            df['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)
            
        for key in second_map.keys():
            df['snippet_x'].replace(key, second_map[key], regex=True, inplace=True)
            df['snippet_y'].replace(key, second_map[key], regex=True, inplace=True)
            
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
        r'&ouml;': r'o',
        r'&hellip;': r'...',
        r'&eacute;': r'e',
        r'&aacute;': r'a',
        r'&rsquo;': r"'",
        r'&lsquo;': r"'",
        ' & ': ' and ',  #
        '&id=': r'_id=',
#         '&': '_',
        '——': r'-',
        '—': r'-',
        #'/': r'',
        '\^': r'',
        '^': r'',
        '±': r'+',
        'y': r'у',
        'xc': r'хс',
        'x': r'х',
         r'companу': r'company',
         r'kasperskу': r'kaspersky',
    }
    
    for key in text_html_map.keys():
        text = text.replace(key, text_html_map[key])
        
    for key in second_map.keys():
        text = text.replace(key, second_map[key])
        
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text    


def _prepare_text(text):
    text = text.replace('  \n', '#####')
    text = text.replace(' \n', '#####')
    text = text + '#####'
    text = text.replace('#####', '\n')
    text_html_map = {
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
        r'&ouml;': r'o',
        r'&hellip;': r'...',
        r'&eacute;': r'e',
        r'&aacute;': r'a',
        r'&rsquo;': r"'",
        r'&lsquo;': r"'",
        ' & ': ' and ',  #
        '&id=': r'_id=',
#         '&': '_',
        '——': r'-',
        '—': r'-',
        #'/': r'',
        '\^': r'',
        '^': r'',
        '±': r'+',
        'y': r'у',
        'xc': r'хс',
        'x': r'х',
         r'companу': r'company',
         r'kasperskу': r'kaspersky',
    }
    
    for key in text_html_map.keys():
        text = text.replace(key, text_html_map[key])
        
    for key in second_map.keys():
        text = text.replace(key, second_map[key])
        
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text

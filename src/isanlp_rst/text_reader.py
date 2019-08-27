class TextReader:
    def __init__(self):
        self.text_html_map = {
            r'\n': r' ',
            r'&gt;': r'>',
            r'&lt;': r'<',
            r'&amp;': r'&',
            r'&quot;': r'"',
            r'&ndash;': r'–',
            r'##### ': r'',
            '\\': r' ',
            r'  ': r' ',
            r'——': r'-',
            r'—': r'-',
            r'/': r'',
            r'\^': r'',
            r'^': r'',
            r'±': r'+'
        }

    def __call__(self, text):
        for key, value in self.text_html_map.items():
            while key in text:
                text = text.replace(key, value)
        return text

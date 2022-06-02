from bs4 import BeautifulSoup


def add_newline_markers(annotation: str, text: str, marker='##### '):
    """ rs3 files can be missing newline markers,
        the function is for their recovery from source texts (green color).
        It will also show the mismatch between source and annotation (red color)"""
    text = text.replace('\xa0', '')
    soup = BeautifulSoup(annotation, 'xml')
    for element in soup.findAll():
        if element.name == 'segment':
            element.string = element.string.replace('\xa0', '')
            idx = text.find(element.string)
            if idx < 0:
                print(f'\x1b[31m{element}\x1b[0m')  # print in red
            elif idx > 2:
                if '\n' in text[idx - 3:idx]:
                    element.string = marker + element.string
                    print(f'\x1b[32m{element}\x1b[0m')
                else:
                    print(element)
            else:
                element.string = marker + element.string
                print(f'\x1b[32m{element}\x1b[0m')

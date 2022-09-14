import os

from isanlp import PipelineCommon
from isanlp.processor_razdel import ProcessorRazdel
from isanlp.processor_remote import ProcessorRemote
from isanlp.annotation_rst import ForestExporter
import sys
import random
import string


class Parser:
    def __init__(self, syntax_address, rst_address):
        self.syntax_address, self.rst_address = syntax_address, rst_address

        self.ppl = PipelineCommon([
            (ProcessorRazdel(), ['text'],
             {'tokens': 'tokens',
              'sentences': 'sentences'}),
            (ProcessorRemote(self.syntax_address[0], self.syntax_address[1], '0'),
             ['tokens', 'sentences'],
             {'lemma': 'lemma',
              'morph': 'morph',
              'syntax_dep_tree': 'syntax_dep_tree',
              'postag': 'postag'}),
            (ProcessorRemote(self.rst_address[0], self.rst_address[1], 'default'),
             ['text', 'tokens', 'sentences', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
             {'rst': 'rst'})
        ])

        self.placeholder = open('output_file.txt', 'r').read()
        self.rs3_exporter = ForestExporter("utf8")

    def __call__(self, text, *args, **kwargs):
        def prepare_text(text):
            text = text.strip()
            while '\n\n' in text:
                text = text.replace('\n\n', '\n')

        result = self.ppl(prepare_text(text))['rst']
        print(result, file=sys.stderr)
        return self.wrap(result)

    def wrap(self, result):
        tempfilename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) + '.rs3'
        self.rs3_exporter(result, tempfilename)
        result_str = open(tempfilename, 'r').read()
        os.remove(tempfilename)
        return result_str

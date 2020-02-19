from isanlp import PipelineCommon
from isanlp.processor_remote import ProcessorRemote
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd


class PipelineDefault:
    def __init__(self, address_morph, address_syntax, address_rst):
        self._ppl = PipelineCommon([(ProcessorRemote(address_morph[0], address_morph[1], 'default'),
                                     ['text'],
                                     {'sentences': 'sentences',
                                      'tokens': 'tokens',
                                      'postag': 'postag',
                                      'lemma': 'lemma'}),
                                    (ConverterMystemToUd(),
                                     ['postag'],
                                     {'morph': 'morph',
                                      'postag': 'postag'}),
                                    (ProcessorRemote(address_syntax[0], address_syntax[1], '0'),
                                     ['tokens', 'sentences'],
                                     {'syntax_dep_tree': 'syntax_dep_tree',
                                      'postag': 'ud_postag'}),
                                    (ProcessorRemote(address_rst[0], address_rst[1], 'default'),
                                     ['text', 'tokens', 'sentences', 'lemma', 'morph', 'postag', 'syntax_dep_tree'],
                                     {'rst': 'rst'})])
        self._name = 'default'

    def __call__(self, *args, **kwargs):
        return self._ppl(*args, **kwargs)

    def get_processors(self):
        return self._ppl.get_processors()

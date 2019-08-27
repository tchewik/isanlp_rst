from processor_rst import ProcessorRST
from isanlp import PipelineCommon

PPL_RST = PipelineCommon([(ProcessorRST('/models'),
                                     ['text', 'tokens', 'sentences', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
                                     {0: 'rst'})
                                    ],
                                   name='default')

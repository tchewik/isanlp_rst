from processor_rst import ProcessorRST
from isanlp import PipelineCommon

PPL_RST = PipelineCommon([(ProcessorRST('/models'),
                                     ['text', 'tokens', 'sentences', 'lemma', 'morph', 'postag', 'syntax_dep_tree'],
                                     {0: 'rst'})
                                    ],
                                   name='default')

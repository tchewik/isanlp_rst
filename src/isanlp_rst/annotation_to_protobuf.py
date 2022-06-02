"""Functions for convertion of basic linguistic annotations to protobuf structures."""

from . import annotation_pb2 as pb
from . import annotation as ann
from . import annotation_rst_pb2 as pb_rst
from . import annotation_rst as ann_rst
import sys


def convert_map(mp):
    pb_annots = pb.AnnotationMap()
    
    for name, annot in mp.items():
        pb_ann_placeholder = pb_annots.data.add()
        pb_ann_placeholder.key = name
        pb_ann_placeholder.value.Pack(convert_annotation(annot))
    
    return pb_annots
        
        
def convert_list(lst):
    pb_annots = pb.AnnotationList()
    
    for annot in lst:
        pb_ann_placeholder = pb_annots.data.add()
        pb_ann_placeholder.Pack(convert_annotation(annot))
        
    return pb_annots


def convert_tuple(tpl):
    return pb.AnnotationTuple(data = convert_list(list(tpl)))


def convert_token(token):
    return pb.Token(span = pb.Span(begin = token.begin, end = token.end), text = token.text)


def convert_sentence(sentence):
    return pb.Sentence(begin = sentence.begin, end = sentence.end)


def convert_wordsynt(wordsynt):
    return pb.WordSynt(parent = wordsynt.parent, link_name = wordsynt.link_name)


def convert_event(event):
    return pb.Event(pred = convert_tuple(event.pred), 
                    args = convert_list(event.args))

def convert_discourse_unit(du):
    if not du.left:
        return pb_rst.DiscourseUnit(id=du.id,
                                    left=pb_rst.DiscourseUnit(),
                                    right=pb_rst.DiscourseUnit(),
                                    text=du.text,
                                    start=du.start,
                                    end=du.end,
                                    relation=du.relation,
                                    nuclearity=du.nuclearity,
                                    proba=du.proba
                                   )
    
    return pb_rst.DiscourseUnit(id=du.id, 
                            left=convert_discourse_unit(du.left),
                            right=convert_discourse_unit(du.right),
                            text=du.text,
                            start=du.start,
                            end=du.end,
                            relation=du.relation,
                            nuclearity=du.nuclearity,
                            proba=du.proba
                           )

def convert_annotation(ling_ann):
    """Converts basic annotations to protobuf structures."""
    
    if type(ling_ann) is dict:
        return convert_map(ling_ann)
    elif type(ling_ann) is list:
        return convert_list(ling_ann)
    elif type(ling_ann) is tuple:
        return convert_tuple(ling_ann)
    elif type(ling_ann) is ann.Token:
        return convert_token(ling_ann)
    elif type(ling_ann) is ann.Sentence:
        return convert_sentence(ling_ann)
    elif type(ling_ann) is ann.WordSynt:
        return convert_wordsynt(ling_ann)
    elif type(ling_ann) is str:
        return pb.LngString(value = ling_ann)
    elif type(ling_ann) is int:
        return pb.LngInt(value = ling_ann)
    elif type(ling_ann) is ann.TaggedSpan:
        return pb.TaggedSpan(span = pb.Span(begin = ling_ann.begin, 
                                            end = ling_ann.end), 
                             tag = ling_ann.tag)
    elif type(ling_ann) is ann.TaggedRelation:
        return pb.TaggedRelation(head = ling_ann.head, 
                                 dep = ling_ann.dep,
                                 tag = ling_ann.tag)
    elif type(ling_ann) is ann.Event:
        return convert_event(ling_ann)
    elif type(ling_ann) is ann_rst.DiscourseUnit:
        return convert_discourse_unit(ling_ann)
    else:
        print(f"WRONG TYPE: {type(ling_ann)}; EXPECTED {ann_rst.DiscourseUnit}", file=sys.stderr)
        raise TypeError()

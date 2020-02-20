from . import annotation_pb2 as pb
from . import annotation_rst_pb2 as pb_rst
from . import annotation as ann
from . import annotation_rst as ann_rst
from google.protobuf.any_pb2 import Any


def get_pb_type(pb_ann):    
    if pb_ann.Is(pb.AnnotationList.DESCRIPTOR):
        return pb.AnnotationList
    elif pb_ann.Is(pb.AnnotationMap.DESCRIPTOR):
        return pb.AnnotationMap
    elif pb_ann.Is(pb.AnnotationTuple.DESCRIPTOR):
        return pb.AnnotationTuple
    elif pb_ann.Is(pb.Sentence.DESCRIPTOR):
        return pb.Sentence
    elif pb_ann.Is(pb.Span.DESCRIPTOR):
        return pb.Span
    elif pb_ann.Is(pb.Token.DESCRIPTOR):
        return pb.Token
    elif pb_ann.Is(pb.WordSynt.DESCRIPTOR):
        return pb.WordSynt
    elif pb_ann.Is(pb.LngInt.DESCRIPTOR):
        return pb.LngInt
    elif pb_ann.Is(pb.LngString.DESCRIPTOR):
        return pb.LngString
    elif pb_ann.Is(pb.Event.DESCRIPTOR):
        return pb.Event
    elif pb_ann.Is(pb.TaggedSpan.DESCRIPTOR):
        return pb.TaggedSpan
    elif pb_ann.Is(pb_rst.DiscourseUnit.DESCRIPTOR):
        return pb_rst.DiscourseUnit
    else:
        raise TypeError()


def convert_tuple(obj):
    return tuple([convert_annotation(item) for item in obj.data.data])


def convert_list(obj):
    return [convert_annotation(item) for item in obj.data]
    

def convert_annotation(pb_ann):
    if type(pb_ann) is Any:
        tp = get_pb_type(pb_ann)
        obj = tp()
        pb_ann.Unpack(obj)
    else:
        obj = pb_ann
    
    #TODO: optimize somehow
    if type(obj) is pb.AnnotationList:
        return convert_list(obj)
    elif obj.DESCRIPTOR == pb.AnnotationMap.DESCRIPTOR:
        return {item.key : convert_annotation(item.value) for item in obj.data}
    elif obj.DESCRIPTOR == pb.AnnotationTuple.DESCRIPTOR:
        return convert_tuple(obj)
    elif obj.DESCRIPTOR == pb.Sentence.DESCRIPTOR:
        return ann.Sentence(obj.begin, obj.end)
    elif obj.DESCRIPTOR == pb.Span.DESCRIPTOR:
        return ann.Span(obj.begin, obj.end)
    elif obj.DESCRIPTOR == pb.Token.DESCRIPTOR:
        return ann.Token(obj.text, obj.span.begin, obj.span.end)
    elif obj.DESCRIPTOR == pb.WordSynt.DESCRIPTOR:
        return ann.WordSynt(obj.parent, obj.link_name)
    elif obj.DESCRIPTOR == pb.LngInt.DESCRIPTOR:
        return obj.value
    elif obj.DESCRIPTOR == pb.LngString.DESCRIPTOR:
        return obj.value
    elif obj.DESCRIPTOR == pb.TaggedSpan.DESCRIPTOR:
        return ann.TaggedSpan(tag = obj.tag, 
                              begin = obj.span.begin, 
                              end = obj.span.end)
    elif obj.DESCRIPTOR == pb.Event.DESCRIPTOR:
        return ann.Event(pred = convert_tuple(obj.pred), 
                         args = convert_list(obj.args))
    elif obj.DESCRIPTOR == pb_rst.DiscourseUnit.DESCRIPTOR:
        if not obj.relation:
            return None
        return ann_rst.DiscourseUnit(id=obj.id, left=convert_annotation(obj.left), right=convert_annotation(obj.right), text=obj.text, start=obj.start, end=obj.end, relation=obj.relation, nuclearity=obj.nuclearity, proba=float(obj.proba))
    else:
        raise TypeError()

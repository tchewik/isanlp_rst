# -*- coding: utf-8 -*-

from collections import Counter
import re

class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class BracketMetric(Metric):

    def __init__(self, eps=1e-8):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


class SpanMetric(Metric):

    def __init__(self, eps=1e-5):
        super(SpanMetric, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            pred, gold = set(pred), set(gold)
            self.tp += len(pred & gold)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.p * self.r / (self.p + self.r + self.eps)

class DiscourseMetric(Metric):

    def __init__(self, eps=1e-8):
        super().__init__()

        self.n = 0.0
        self.us_tp = 0.0
        self.r_tp = 0.0
        self.n_tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            span_pred=self.get_span_label(pred)
            span_gold=self.get_span_label(gold)

            #unlabeled span
            us_pred = Counter([(i, j) for i, j, relation, nuclearity in span_pred])
            us_gold = Counter([(i, j) for i, j, relation, nuclearity in span_gold])
            us_tp = list((us_pred & us_gold).elements())

            #relation span
            r_pred = Counter([(i, j, relation) for i, j, relation, nuclearity in span_pred])
            r_gold = Counter([(i, j, relation) for i, j, relation, nuclearity in span_gold])
            r_tp = list((r_pred & r_gold).elements())

            # nuclearity span
            n_pred = Counter([(i, j, nuclearity) for i, j, relation, nuclearity in span_pred])
            n_gold = Counter([(i, j, nuclearity) for i, j, relation, nuclearity in span_gold])
            n_tp = list((n_pred & n_gold).elements())
            self.n += 1
            # self.n_ucm += len(utp) == len(pred) == len(gold)
            # self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.us_tp += len(us_tp)
            self.r_tp += len(r_tp)
            self.n_tp += len(n_tp)
            self.pred += len(span_pred)
            self.gold += len(span_gold)

    def __repr__(self):
        # s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        # s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        # s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"
        # s = f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        # s += f"RP: {self.rp:6.2%} RR: {self.rr:6.2%} RF: {self.rf:6.2%} "
        # s += f"NP: {self.np:6.2%} NR: {self.nr:6.2%} NF: {self.nf:6.2%}"
        s = f"UF: {self.uf:6.2%} "
        s += f" NF: {self.nf:6.2%} "
        s += f" RF: {self.rf:6.2%}"


        return s

    def get_span_label(self, text_tree):
        if text_tree=='NONE':
            return []
        else:
            metric_token_split = re.split(' ', text_tree)
            span_label=[]
            for each_split in metric_token_split:
                left_start, Nuclearity_left, Relation_left, left_end, \
                right_start, Nuclearity_right, Relation_right, right_end = re.split(':|=|,', each_split[1:-1])
                span_label.append((int(left_start),int(left_end), Relation_left, Nuclearity_left))
                span_label.append((int(right_start),int(right_end), Relation_right, Nuclearity_right))
            return span_label

    @property
    def score(self):
        return self.rf

    # @property
    # def ucm(self):
    #     return self.n_ucm / (self.n + self.eps)
    #
    # @property
    # def lcm(self):
    #     return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.us_tp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.us_tp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.us_tp / (self.pred + self.gold + self.eps)

    @property
    def rp(self):
        return self.r_tp / (self.pred + self.eps)

    @property
    def rr(self):
        return self.r_tp / (self.gold + self.eps)

    @property
    def rf(self):
        return 2 * self.r_tp / (self.pred + self.gold + self.eps)

    @property
    def np(self):
        return self.n_tp / (self.pred + self.eps)

    @property
    def nr(self):
        return self.n_tp / (self.gold + self.eps)

    @property
    def nf(self):
        return 2 * self.n_tp / (self.pred + self.gold + self.eps)


class SPMRL_BracketMetric(Metric):

    def __init__(self, eps=1e-8):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):

            adjust_pred=[(i, j, label.split('-')[0]) for i, j, label in pred]
            adjust_gold=[(i, j, label.split('-')[0]) for i, j, label in gold]
            upred = Counter([(i, j) for i, j, label in adjust_pred])
            ugold = Counter([(i, j) for i, j, label in adjust_gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(adjust_pred)
            lgold = Counter(adjust_gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)

class SPMRL_external_Metric(object):
    def __init__(self, ur, up, uf, lr, lp, lf):
        self.ur = ur
        self.up = up
        self.uf = uf
        self.lr=lr
        self.lp=lp
        self.lf=lf

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other
    def __repr__(self):
        s = f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%}; "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"
        return s
    @property
    def score(self):
        return self.lf

class DiscourseMetricDoc(Metric):

    def __init__(self, eps=1e-8):
        super().__init__()

        self.n = 0.0
        self.us_tp = 0.0
        self.r_tp = 0.0
        self.n_tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.seg_tp = 0.0
        self.rn_tp = 0.0
        self.eps = eps


    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            # print(pred)
            # print(gold)
            # input()
            span_pred, segment_pred=self.get_span_label(pred)
            span_gold, segment_gold=self.get_span_label(gold)

            #unlabeled span
            us_pred = Counter([(i, j) for i, j, relation, nuclearity in span_pred])
            us_gold = Counter([(i, j) for i, j, relation, nuclearity in span_gold])
            us_tp = list((us_pred & us_gold).elements())

            #relation span
            r_pred = Counter([(i, j, relation) for i, j, relation, nuclearity in span_pred])
            r_gold = Counter([(i, j, relation) for i, j, relation, nuclearity in span_gold])
            r_tp = list((r_pred & r_gold).elements())

            # nuclearity span
            n_pred = Counter([(i, j, nuclearity) for i, j, relation, nuclearity in span_pred])
            n_gold = Counter([(i, j, nuclearity) for i, j, relation, nuclearity in span_gold])
            n_tp = list((n_pred & n_gold).elements())

            # relation & nuclearity
            rn_pred = Counter([(i, j, relation, nuclearity) for i, j, relation, nuclearity in span_pred])
            rn_gold = Counter([(i, j, relation, nuclearity) for i, j, relation, nuclearity in span_gold])
            rn_tp = list((rn_pred & rn_gold).elements())

            # segment correct
            seg_pred = Counter(segment_pred)
            seg_gold = Counter(segment_gold)
            seg_tp = list((seg_pred & seg_gold).elements())


            self.n += 1
            # self.n_ucm += len(utp) == len(pred) == len(gold)
            # self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.us_tp += len(us_tp)
            self.r_tp += len(r_tp)
            self.n_tp += len(n_tp)
            self.rn_tp += len(rn_tp)
            self.seg_tp += len(seg_tp)
            self.pred += len(span_pred)
            self.gold += len(span_gold)


    def __repr__(self):
        # s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        # s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        # s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"
        # s = f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        # s += f"RP: {self.rp:6.2%} RR: {self.rr:6.2%} RF: {self.rf:6.2%} "
        # s += f"NP: {self.np:6.2%} NR: {self.nr:6.2%} NF: {self.nf:6.2%}"
        s = f"SegF: {self.segf:6.2%} "
        s += f"UF: {self.uf:6.2%} NF: {self.nf:6.2%} RF: {self.rf:6.2%} "
        s += f"Full RNF: {self.rnf:6.2%} "

        return s

    def get_span_label(self, text_tree):
        if text_tree=='NONE':
            return [[],[]]
        else:
            metric_token_split = re.split(' ', text_tree)
            span_label=[]
            segmentpoints=[]
            for each_split in metric_token_split:
                if each_split:
                    left_start, Nuclearity_left, Relation_left, left_end, \
                    right_start, Nuclearity_right, Relation_right, right_end = re.split(':|=|,', each_split[1:-1])
                    # span_label.append((int(left_start),int(left_end), Relation_left, Nuclearity_left))
                    # span_label.append((int(right_start),int(right_end), Relation_right, Nuclearity_right))
                    Nuclearity=Nuclearity_left+Nuclearity_right
                    if Relation_left=='span':
                        Relation= Relation_right
                    else:
                        Relation =Relation_left
                    span_label.append((int(left_start),int(right_end), Relation, Nuclearity_left+Nuclearity_right))
                    segmentpoints.append(left_end)
    #             assert len(set(segmentpoints))==len(span_label), f"something wrong segmentpoints:{segmentpoints}; span_label:{span_label}"
            return span_label, segmentpoints

    @property
    def score(self):
        # return self.rf
        return self.rnf

    # @property
    # def ucm(self):
    #     return self.n_ucm / (self.n + self.eps)
    #
    # @property
    # def lcm(self):
    #     return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.us_tp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.us_tp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.us_tp / (self.pred + self.gold + self.eps)

    @property
    def segf(self):
        return 2 * self.seg_tp / (self.pred + self.gold + self.eps)

    @property
    def rnf(self):
        return 2 * self.rn_tp / (self.pred + self.gold + self.eps)

    @property
    def rp(self):
        return self.r_tp / (self.pred + self.eps)

    @property
    def rr(self):
        return self.r_tp / (self.gold + self.eps)

    @property
    def rf(self):
        return 2 * self.r_tp / (self.pred + self.gold + self.eps)

    @property
    def np(self):
        return self.n_tp / (self.pred + self.eps)

    @property
    def nr(self):
        return self.n_tp / (self.gold + self.eps)

    @property
    def nf(self):
        return 2 * self.n_tp / (self.pred + self.gold + self.eps)
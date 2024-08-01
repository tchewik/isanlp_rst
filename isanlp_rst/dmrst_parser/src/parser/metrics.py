import re


def get_eval_data_rst_parseval(sen, edus):
    b = re.findall(r'\d+', sen)
    b = [str(edus[int(i) - 1]) for i in b]
    cur_new = []
    x = 0
    while x < len(b):
        cur_new.append(b[x] + '-' + b[x + 1])
        x = x + 2
    span = re.split(r' ', sen)
    # print(span)
    dic = {}
    for i in range(len(span)):
        temp = span[i]
        IDK = re.split(r'[:,=]', temp)
        Nuclearity1 = IDK[1]
        relation1 = IDK[2]
        Nuclearity2 = IDK[5]
        relation2 = IDK[6]
        dic[cur_new[2 * i]] = [relation1, Nuclearity1]
        dic[cur_new[2 * i + 1]] = [relation2, Nuclearity2]
    return dic


def get_eval_data_parseval(tree_spans: str, edus: list):
    span_list = tree_spans.strip().split()
    dic = {}
    for i in range(len(span_list)):
        temp = span_list[i]
        IDK = re.split(r'[:,=]', temp)
        nuclearity = IDK[1][0] + IDK[5][0]
        relation1 = IDK[2]
        relation2 = IDK[6]
        relation = relation1 if relation1 != 'span' else relation2
        start = str(edus[int(IDK[0].strip('(')) - 1])
        end = str(edus[int(IDK[-1].strip(')')) - 1])
        span = start + '-' + end
        dic[span] = [relation, nuclearity]
    return dic


def get_measurement(tree1_spans, tree2_spans, tree1_edus, tree2_edus, use_org_parseval):
    if use_org_parseval:
        dic1 = get_eval_data_parseval(tree1_spans, tree1_edus)
        dic2 = get_eval_data_parseval(tree2_spans, tree2_edus)
    else:
        dic1 = get_eval_data_rst_parseval(tree1_spans, tree1_edus)
        dic2 = get_eval_data_rst_parseval(tree2_spans, tree2_edus)

    n_ns = 0
    n_relation = 0
    n_full = 0

    # number of right spans
    right_span = list(set(dic1.keys()).intersection(set(dic2.keys())))
    n_spans = len(right_span)

    # Right Number of relations and nuclearity
    for span in right_span:
        if dic1[span][0] == dic2[span][0]:
            n_relation = n_relation + 1
        if dic1[span][1] == dic2[span][1]:
            n_ns = n_ns + 1
        if dic1[span][0] == dic2[span][0] and dic1[span][1] == dic2[span][1]:
            n_full += 1

    correct_span = n_spans
    correct_relation = n_relation
    correct_nuclearity = n_ns
    correct_full = n_full
    no_system = len(dic1.keys())
    no_golden = len(dic2.keys())

    return correct_span, correct_relation, correct_nuclearity, correct_full, no_system, no_golden


def get_seg_measure(pred_seg, gold_seg):
    num_gold = len(gold_seg)
    num_pred = len(pred_seg)
    correct = len(set(pred_seg) & set(gold_seg))

    return num_gold, num_pred, correct


def get_batch_metrics(pred_spans_batch, gold_spans_batch, pred_edu_breaks_batch, gold_edu_breaks_batch,
                      use_org_parseval):
    correct_span = 0
    correct_relation = 0
    correct_nuclearity = 0
    correct_full = 0
    n_system = 0
    n_golden = 0
    n_gold_seg = 0
    n_pred_seg = 0
    n_correct_seg = 0

    correct_span_batch_list = []
    correct_relation_batch_list = []
    correct_nuclearity_batch_list = []
    correct_full_batch_list = []
    no_system_batch_list = []
    no_golden_batch_list = []

    for i in range(len(pred_spans_batch)):

        cur_pred_spans = pred_spans_batch[i][0]
        cur_gold_spans = gold_spans_batch[i]
        cur_pred_edus = pred_edu_breaks_batch[i]
        cur_gold_edus = gold_edu_breaks_batch[i]

        cur_span_n = 0
        cur_relation_n = 0
        cur_ns_n = 0
        cur_sys_n = 0
        cur_golden_n = 0
        cur_full = 0

        num_gold_seg, num_pred_seg, num_correct_seg = get_seg_measure(cur_pred_edus, cur_gold_edus)
        n_gold_seg += num_gold_seg
        n_pred_seg += num_pred_seg
        n_correct_seg += num_correct_seg

        if cur_pred_spans != 'NONE' and cur_gold_spans != 'NONE':

            cur_span_n, cur_relation_n, cur_ns_n, cur_full, cur_sys_n, cur_golden_n = get_measurement(cur_pred_spans,
                                                                                                      cur_gold_spans,
                                                                                                      cur_pred_edus,
                                                                                                      cur_gold_edus,
                                                                                                      use_org_parseval)

            correct_span += cur_span_n
            correct_relation += cur_relation_n
            correct_nuclearity += cur_ns_n
            correct_full += cur_full
            n_system += cur_sys_n
            n_golden += cur_golden_n

        elif cur_pred_spans != 'NONE' and cur_gold_spans == 'NONE':
            _, _, _, _, cur_sys_n, _ = get_measurement(cur_pred_spans, cur_pred_spans, cur_pred_edus, cur_pred_edus,
                                                       use_org_parseval)
            n_system += cur_sys_n

        elif cur_pred_spans == 'NONE' and cur_gold_spans != 'NONE':
            _, _, _, _, _, cur_goldenno = get_measurement(cur_gold_spans, cur_gold_spans, cur_gold_edus, cur_gold_edus,
                                                          use_org_parseval)
            n_golden += cur_goldenno

        correct_span_batch_list.append(cur_span_n)
        correct_relation_batch_list.append(cur_relation_n)
        correct_nuclearity_batch_list.append(cur_ns_n)
        correct_full_batch_list.append(cur_full)
        no_system_batch_list.append(cur_sys_n)
        no_golden_batch_list.append(cur_golden_n)

    return (correct_span, correct_relation, correct_nuclearity, correct_full, n_system, n_golden,
            correct_span_batch_list, correct_relation_batch_list, correct_nuclearity_batch_list,
            correct_full_batch_list,
            no_system_batch_list, no_golden_batch_list, (n_gold_seg, n_pred_seg, n_correct_seg))


def get_micro_metrics(correct_span, correct_relation, correct_nuclearity, correct_full, n_sys, n_gold,
                      n_gold_seg, n_pred_seg, n_correct_seg):
    n_sys = 1 if n_sys == 0 else n_sys

    # segmentation
    precision_seg = n_correct_seg / n_pred_seg
    recall_seg = n_correct_seg / n_gold_seg
    f1_seg = (2 * n_correct_seg) / (n_gold_seg + n_pred_seg)

    # Span
    precision_span = correct_span / n_sys
    recall_span = correct_span / n_gold
    f1_span = (2 * correct_span) / (n_gold + n_sys)

    # Relation
    precision_relation = correct_relation / n_sys
    recall_relation = correct_relation / n_gold
    f1_relation = (2 * correct_relation) / (n_gold + n_sys)

    # Nuclearity
    precision_nuclearity = correct_nuclearity / n_sys
    recall_nuclearity = correct_nuclearity / n_gold
    f1_nuclearity = (2 * correct_nuclearity) / (n_gold + n_sys)

    # Full
    f1_Full = (2 * correct_full) / (n_gold + n_sys)

    return (precision_span, recall_span, f1_span), (precision_relation, recall_relation, f1_relation), \
        (precision_nuclearity, recall_nuclearity, f1_nuclearity), f1_Full, (precision_seg, recall_seg, f1_seg)


def calc_metrics(n_correct, n_pred, n_gold):
    pr = n_correct / n_pred
    re = n_correct / n_gold
    f1 = (2 * n_correct) / (n_gold + n_pred)

    return pr, re, f1


def get_macro_metrics(correct_span_list, correct_nuclearity_list, correct_relation_list, correct_full_list,
                      no_system_list, no_golden_list):
    precision_span_list = []
    precision_relation_list = []
    precision_nuclearity_list = []
    precision_full_list = []

    recall_span_list = []
    recall_relation_list = []
    recall_nuclearity_list = []
    recall_full_list = []

    f1_span_list = []
    f1_relation_list = []
    f1_nuclearity_list = []
    f1_full_list = []

    for i in range(len(correct_span_list)):
        correct_span = correct_span_list[i]
        correct_relation = correct_relation_list[i]
        correct_nuclearity = correct_nuclearity_list[i]
        correct_full = correct_full_list[i]
        no_system = no_system_list[i]
        no_golden = no_golden_list[i]

        # span
        precision_span, recall_span, f1_span = calc_metrics(correct_span, no_system, no_golden)
        precision_span_list.append(precision_span)
        recall_span_list.append(recall_span)
        f1_span_list.append(f1_span)

        # Nuclearity
        precision_nuclearity, recall_nuclearity, f1_nuclearity = calc_metrics(correct_nuclearity, no_system, no_golden)
        precision_nuclearity_list.append(precision_nuclearity)
        recall_nuclearity_list.append(recall_nuclearity)
        f1_nuclearity_list.append(f1_nuclearity)

        # Relation
        precision_relation, recall_relation, f1_relation = calc_metrics(correct_relation, no_system, no_golden)
        precision_relation_list.append(precision_relation)
        recall_relation_list.append(recall_relation)
        f1_relation_list.append(f1_relation)

        # Full
        precision_full, recall_full, f1_full = calc_metrics(correct_full, no_system, no_golden)
        precision_full_list.append(precision_full)
        recall_full_list.append(recall_full)
        f1_full_list.append(f1_full)

    precision_span_avg = sum(precision_span_list) / len(precision_span_list)
    recall_span_avg = sum(recall_span_list) / len(recall_span_list)
    f1_span_avg = sum(f1_span_list) / len(f1_span_list)

    precision_nuclearity_avg = sum(precision_nuclearity_list) / len(precision_nuclearity_list)
    recall_nuclearity_avg = sum(recall_nuclearity_list) / len(recall_nuclearity_list)
    f1_nuclearity_avg = sum(f1_nuclearity_list) / len(f1_nuclearity_list)

    precision_relation_avg = sum(precision_relation_list) / len(precision_relation_list)
    recall_relation_avg = sum(recall_relation_list) / len(recall_relation_list)
    f1_relation_avg = sum(f1_relation_list) / len(f1_relation_list)

    precision_full_avg = sum(precision_full_list) / len(precision_full_list)
    recall_full_avg = sum(recall_full_list) / len(recall_full_list)
    f1_full_avg = sum(f1_full_list) / len(f1_full_list)

    return ((precision_span_avg, recall_span_avg, f1_span_avg),
            (precision_nuclearity_avg, recall_nuclearity_avg, f1_nuclearity_avg),
            (precision_relation_avg, recall_relation_avg, f1_relation_avg),
            (precision_full_avg, recall_full_avg, f1_full_avg))

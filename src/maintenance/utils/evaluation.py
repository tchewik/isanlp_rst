
import pandas as pd
from utils.file_reading import text_html_map


def prepare_string(string):
    for key, value in text_html_map.items():
        string = string.replace(key, value).strip()
                
    if '-' in string:
        string = string.replace('-', ' ').strip()

    while '  ' in string:
        string = string.replace('  ', ' ')
        
    return string

def metric_parseval(parsed_pairs, gold, span=True, labeled=False, nuc=False):
    parsed_strings = []
    for row in parsed_pairs:
        if span:
            x, y = prepare_string(row[0]), prepare_string(row[1])

        else:
            x, y = '', ''

        label = ' ' + row[2].split('_')[0] if labeled else ''

        if nuc and row[3]:
            nuclearity = ' ' + row[3]
        else:
            nuclearity = ''

        parsed_strings.append(x + ' ' + y + label + nuclearity)

    parsed_strings = set(parsed_strings)

    gold_strings = []
    for i in gold.index:
        if span:
            x, y = prepare_string(gold.loc[i, 'snippet_x']), prepare_string(gold.loc[i, 'snippet_y'])

        else:
            x, y = '', ''

        label = ' ' + gold.loc[i, 'category_id'].split('_')[0] if labeled else ''
        nuclearity = ' ' + gold.loc[i, 'order'] if nuc else ''
        gold_strings.append(x + ' ' + y + label + nuclearity)

    gold_strings = set(gold_strings)

    true_pos = len(gold_strings & parsed_strings)
    all_parsed = len(parsed_strings)
    all_gold = len(gold_strings)
    return true_pos, all_parsed, all_gold


def metric_parseval_df(parsed_pairs, gold, span=True, labeled=False, nuc=False):
    parsed_strings = []
    
    parsed_strings = []
    for i in parsed_pairs.index:
        if span:
            x, y = prepare_string(parsed_pairs.loc[i, 'snippet_x']), prepare_string(parsed_pairs.loc[i, 'snippet_y'])

        else:
            x, y = '', ''

        label = ' ' + parsed_pairs.loc[i, 'category_id'].split('_')[0] if labeled else ''
        nuclearity = ' ' + parsed_pairs.loc[i, 'order'] if nuc else ''
        parsed_strings.append(x + ' ' + y + label + nuclearity)

    parsed_strings = set(parsed_strings)

    gold_strings = []
    for i in gold.index:
        if span:
            x, y = prepare_string(gold.loc[i, 'snippet_x']), prepare_string(gold.loc[i, 'snippet_y'])

        else:
            x, y = '', ''

        label = ' ' + gold.loc[i, 'category_id'].split('_')[0] if labeled else ''
        nuclearity = ' ' + gold.loc[i, 'order'] if nuc else ''
        gold_strings.append(x + ' ' + y + label + nuclearity)

    gold_strings = set(gold_strings)

    true_pos = len(gold_strings & parsed_strings)
    all_parsed = len(parsed_strings)
    all_gold = len(gold_strings)
    return true_pos, all_parsed, all_gold


def extr_pairs(tree):
    pp = []
    if tree.left:
        pp.append([tree.left.text, tree.right.text, tree.relation])
        pp += extr_pairs(tree.left)
        pp += extr_pairs(tree.right)
    return pp


def extr_pairs(tree, text):
    pp = []
    if tree.left:
        pp.append([text[tree.left.start:tree.left.end], text[tree.right.start:tree.right.end], tree.relation,
                   tree.nuclearity])
        pp += extr_pairs(tree.left, text)
        pp += extr_pairs(tree.right, text)
    return pp


def extr_pairs_forest(forest, text):
    pp = []
    for tree in forest:
        pp += extr_pairs(tree, text)
    return pp


def _check_snippet_pair_in_dataset(left_snippet, right_snippet):
    left_snippet = left_snippet.strip()
    right_snippet = right_snippet.strip()
    return ((((gold.snippet_x == left_snippet) & (gold.snippet_y == right_snippet)).sum(axis=0) != 0)
            or ((gold.snippet_y == left_snippet) & (gold.snippet_x == right_snippet)).sum(axis=0) != 0)


def _not_parsed_as_in_gold(parsed_pairs: pd.DataFrame, gold: pd.DataFrame, labeled=False):
    for key in text_html_map.keys():
        parsed_pairs['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
        parsed_pairs['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)

    for key in text_html_map.keys():
        gold['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
        gold['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)

    tmp = pd.merge(gold, parsed_pairs, on=['snippet_x', 'snippet_y'], how='left', suffixes=('_gold', '_parsed'))
    if labeled:
        tmp = tmp.fillna(0)
        tmp = tmp[tmp.category_id_parsed != 0]
        #tmp.category_id_gold = tmp.category_id_gold.map(lambda row: row[:-2])
        return tmp[tmp.category_id_gold != tmp.category_id_parsed]
    else:
        return tmp[pd.isnull(tmp.category_id_parsed)]

def extr_edus(tree):
    edus = []
    if tree.left:
        edus += extr_edus(tree.left)
        edus += extr_edus(tree.right)
    else:
        edus.append(tree.text)
    return edus


def eval_segmentation(trees, _gold_edus):
    true_predictions = 0
    all_predicted = 0
    
    gold_edus = []
    
    for gold_edu in _gold_edus:
        gold_edus.append(prepare_string(gold_edu))

    for tree in trees:
        pred_edus = extr_edus(tree)
        all_predicted += len(pred_edus)

        for pred_edu in pred_edus:
            for key, value in text_html_map.items():
                pred_edu = pred_edu.replace(key, value)

            if prepare_string(pred_edu) in gold_edus:
                true_predictions += 1
                
            else:
                print(pred_edu)

    return true_predictions, all_predicted, len(gold_edus)


def eval_pipeline(trees, gold_edus, gold_pairs, text):
    parsed_pairs = extr_pairs_forest(trees, text)
    result = {}
    result['seg_true_pred'], result['seg_all_pred'], result['seg_all_true'] = eval_segmentation(trees, gold_edus)
    result['unlab_true_pred'], result['unlab_all_pred'], result['unlab_all_true'] = metric_parseval(parsed_pairs,
                                                                                                    gold_pairs)
    result['lab_true_pred'], result['lab_all_pred'], result['lab_all_true'] = metric_parseval(parsed_pairs, gold_pairs,
                                                                                              labeled=True, nuc=False)
    result['nuc_true_pred'], result['nuc_all_pred'], result['nuc_all_true'] = metric_parseval(parsed_pairs, gold_pairs,
                                                                                              labeled=False, nuc=True)
    result['full_true_pred'], result['full_all_pred'], result['full_all_true'] = metric_parseval(parsed_pairs, gold_pairs,
                                                                                                labeled=True, nuc=True)
    return result

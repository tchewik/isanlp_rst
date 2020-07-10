
import pandas as pd
from utils.file_reading import text_html_map

labels = ['condition_NS',
     'concession_NS',
     'elaboration_NS',
     'preparation_SN',
     'background_SN',
     'condition_SN',
     'purpose_NS',
     'cause-effect_NS',
     'background_NS',
     'interpretation-evaluation_NS',
     'evidence_NS',
     'same-unit_NN',
     'joint_NN',
     'attribution_SN',
     'contrast_NN',
     'restatement_NN',
     'comparison_NN',
     'cause-effect_SN',
     'solutionhood_SN',
     'purpose_SN',
     'sequence_NN',
     'attribution_NS',
     'interpretation-evaluation_SN']

top_classes = [
    'attribution_NS',
    'attribution_SN',
    'purpose_NS',
    'purpose_SN',
    'condition_SN',
    'contrast_NN',
    'condition_NS',
    'joint_NN',
    'concession_NS',
    'same-unit_NN',
    'elaboration_NS',
    'cause-effect_NS',
    #'solutionhood_SN',
    #'cause-effect_SN'
]

class_mapper = {weird_class: 'other' + weird_class[-3:] for weird_class in labels if not weird_class in top_classes}

pred_mapper = {
    'other_NN': 'joint_NN',
    'other_NS': 'joint_NN',
    'other_SN': 'joint_NN'
}

def prepare_gold_pairs(gold_pairs):
    TARGET = 'category_id'

    gold_pairs[TARGET] = gold_pairs[TARGET].map(lambda row: row.split('_')[0])
    gold_pairs[TARGET] = gold_pairs[TARGET].replace([0.0], 'same-unit')
    gold_pairs['order'] = gold_pairs['order'].replace([0.0], 'NN')
    gold_pairs[TARGET] = gold_pairs[TARGET].replace(['relation',], 'joint')
    gold_pairs[TARGET] = gold_pairs[TARGET].replace(['antithesis',], 'contrast')
    gold_pairs[TARGET] = gold_pairs[TARGET].replace(['cause', 'effect'], 'cause-effect')
    gold_pairs[TARGET] = gold_pairs[TARGET].replace(['conclusion',], 'restatement')
    gold_pairs[TARGET] = gold_pairs[TARGET].replace(['evaluation',], 'interpretation-evaluation')
    gold_pairs[TARGET] = gold_pairs[TARGET].replace(['motivation',], 'condition')
    gold_pairs['relation'] = gold_pairs[TARGET].map(lambda row: row.split('_')[0]) + '_' + gold_pairs['order']
    gold_pairs['relation'] = gold_pairs['relation'].replace(['restatement_SN', 'restatement_NS'], 'restatement_NN')
    gold_pairs['relation'] = gold_pairs['relation'].replace(['contrast_SN', 'contrast_NS'], 'contrast_NN')
    gold_pairs['relation'] = gold_pairs['relation'].replace(['solutionhood_NS', 'preparation_NS'], 'elaboration_NS')
    gold_pairs['relation'] = gold_pairs['relation'].replace(['concession_SN', 'evaluation_SN', 
                                                                   'elaboration_SN', 'evidence_SN'], 'preparation_SN')

    for key, value in class_mapper.items():
        gold_pairs['relation'] = gold_pairs['relation'].replace(key, value)
        
    gold_pairs['order'] = gold_pairs['relation'].map(lambda row: row.split('_')[1])
    gold_pairs[TARGET] = gold_pairs['relation'].map(lambda row: row.split('_')[0])
        
    return gold_pairs

def prepare_string(string):
    for key, value in text_html_map.items():
        string = string.replace(key, value).strip()
                
    if '-' in string:
        string = string.replace('-', ' ').strip()

    while '  ' in string:
        string = string.replace('  ', ' ')
        
    return string.strip()

def metric_parseval(parsed_pairs, gold, span=True, labeled=False, nuc=False):
    
    parsed_strings = []
    for i in parsed_pairs.index:
        if span:
            x, y = prepare_string(parsed_pairs.loc[i, 'snippet_x']), prepare_string(parsed_pairs.loc[i, 'snippet_y'])

        else:
            x, y = '', ''
            
        label = parsed_pairs.loc[i, 'category_id'].split('_')[0]
        nuclearity = parsed_pairs.loc[i, 'order']
        merged_label = '_'.join([label, nuclearity])
        
        if labeled or nuc:
            replacement_cand = class_mapper.get(merged_label)
            if replacement_cand:
                if 'other' in replacement_cand:
                    label, nuclearity = pred_mapper.get(replacement_cand).split('_')
                else:
                    label, nuclearity = replacement_cand.split('_')
            
        label = label if labeled else ''
        nuclearity = nuclearity if nuc else ''
        
        result = '&'.join([x, y, label, nuclearity])
        parsed_strings.append(result)

    parsed_strings = list(set(parsed_strings))

    gold_strings = []
    for i in gold.index:
        if span:
            x, y = prepare_string(gold.loc[i, 'snippet_x']), prepare_string(gold.loc[i, 'snippet_y'])

        else:
            x, y = '', ''

        label = gold.loc[i, 'category_id'].split('_')[0] if labeled else ''
        nuclearity = gold.loc[i, 'order'] if nuc else ''
        merged_label = '_'.join([label, nuclearity])
        
        if labeled or nuc:
            if class_mapper.get(merged_label):
                label = class_mapper.get(merged_label).split('_')[0] if labeled else ''
                nuclearity = class_mapper.get(merged_label).split('_')[1] if nuc else ''
            
        result = '&'.join([x, y, label, nuclearity])
        gold_strings.append(result)

    gold_strings = set(gold_strings)
    
    _to_exclude = [string.split('other')[0] for string in gold_strings if 'other' in string]
    gold_strings = set([string for string in gold_strings if not 'other' in string])
    
    _remove_from_parsed_strings = []
    for i, parsed_string in enumerate(parsed_strings):
        for excluding_pair in _to_exclude:
            if excluding_pair in parsed_string:
                _remove_from_parsed_strings.append(i)
    
    parsed_strings = set([parsed_strings[i] for i in range(len(parsed_strings)) if not i in _remove_from_parsed_strings])

    true_pos = len(gold_strings & parsed_strings)
#     if labeled:
#         print('>>', gold_strings & parsed_strings)
        
    all_parsed = len(parsed_strings)
    all_gold = len(gold_strings)
    
    return true_pos, all_parsed, all_gold


def metric_parseval_df(parsed_pairs, gold, span=True, labeled=False, nuc=False):
    parsed_strings = []

    for i in parsed_pairs.index:
        if span:
            x, y = prepare_string(parsed_pairs.loc[i, 'snippet_x']), prepare_string(parsed_pairs.loc[i, 'snippet_y'])

        else:
            x, y = '', ''

        label = ' ' + parsed_pairs.loc[i, 'category_id'].split('_')[0] if labeled else ''
        nuclearity = ' ' + parsed_pairs.loc[i, 'order'] if nuc else ''
        parsed_strings.append(x + ' ' + y + label + nuclearity)

    parsed_strings = list(set(parsed_strings))

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
    
    _to_exclude = [string.split('other')[0] for string in gold_strings if 'other' in string]
    gold_strings = set([string for string in gold_strings if not 'other' in string])
    
    _remove_from_parsed_strings = []
    for i, parsed_string in enumerate(parsed_strings):
        for excluding_pair in _to_exclude:
            if excluding_pair in parsed_string:
                _remove_from_parsed_strings.append(i)
        
    #all_parsed = [string for string in all_parsed if not 'other' in string]
    parsed_strings = set([parsed_strings[i] for i in range(len(parsed_strings)) if not i in _remove_from_parsed_strings])

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


def extr_pairs(tree, text, locations=False):
    pp = []
    if tree.left:
        pp.append([text[tree.left.start:tree.left.end], text[tree.right.start:tree.right.end], tree.relation,
                   tree.nuclearity] + [tree.left.start, tree.right.start] * locations)
        pp += extr_pairs(tree.left, text, locations)
        pp += extr_pairs(tree.right, text, locations)
    return pp


def extr_pairs_forest(forest, text, locations=False):
    pp = []
    for tree in forest:
        pp += extr_pairs(tree, text, locations=locations)
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


def eval_segmentation(trees, _gold_edus, verbose=False):
    true_predictions = 0
    all_predicted = 0
    
    gold_edus = []
    
    for gold_edu in _gold_edus:
        gold_edus.append(prepare_string(gold_edu))

    for tree in trees:
        pred_edus = extr_edus(tree)
        all_predicted += len(pred_edus)

        for pred_edu in pred_edus:
            pred_edu = prepare_string(pred_edu)

            if prepare_string(pred_edu) in gold_edus:
                true_predictions += 1
                
            elif verbose:
                print(pred_edu)

    return true_predictions, all_predicted, len(gold_edus)


def eval_pipeline(trees=None, gold_edus=[], gold_pairs=pd.DataFrame([]), text="", parsed_pairs=pd.DataFrame([])):
    if parsed_pairs.empty:
        parsed_pairs = extr_pairs_forest(trees, text)
    
    result = {}
    result['seg_true_pred'], result['seg_all_pred'], result['seg_all_true'] = eval_segmentation(trees, gold_edus,
                                                                                                verbose=False)
    result['unlab_true_pred'], result['unlab_all_pred'], result['unlab_all_true'] = metric_parseval(parsed_pairs,
                                                                                                    gold_pairs)
    result['lab_true_pred'], result['lab_all_pred'], result['lab_all_true'] = metric_parseval(parsed_pairs, gold_pairs,
                                                                                              labeled=True, nuc=False)
    result['nuc_true_pred'], result['nuc_all_pred'], result['nuc_all_true'] = metric_parseval(parsed_pairs, gold_pairs,
                                                                                              labeled=False, nuc=True)
    result['full_true_pred'], result['full_all_pred'], result['full_all_true'] = metric_parseval(parsed_pairs, gold_pairs,
                                                                                                labeled=True, nuc=True)
    return result

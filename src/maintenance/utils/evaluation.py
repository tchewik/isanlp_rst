import pandas as pd
from utils.file_reading import text_html_map


def metric_parseval(parsed_pairs, gold, labeled=False):
    parsed_strings = []
    for row in parsed_pairs:
        x, y = row[0], row[1]
        
        for key, value in text_html_map.items():
            x = x.replace(key, value).strip()
            y = y.replace(key, value).strip()
            
        label = ' ' + row[2] if labeled else ''
        parsed_strings.append(x + ' ' + y + label)
        
    parsed_strings = set(parsed_strings)
    
    gold_strings = []
    for i in gold.index:
        x, y = gold.loc[i, 'snippet_x'], gold.loc[i, 'snippet_y']
        
        for key, value in text_html_map.items():
            x = x.replace(key, value).strip()
            y = y.replace(key, value).strip()

        label = ' ' + gold.loc[i, 'category_id'] if labeled else ''
        gold_strings.append(x + ' ' + y + label)
        
    gold_strings = set(gold_strings)
    
    true_pos = len(gold_strings & parsed_strings)
    all_parsed = len(parsed_strings)
    all_gold = len(gold_strings)
    
    pr = true_pos / all_parsed
    re = true_pos / all_gold
    f1 = 2 * pr * re / (pr + re + 1e-5)
    
    return {
        'pr': pr,
        're': re,
        'f1': f1
    }
    
def extr_pairs(tree):
    pp = []
    if tree.left:
        pp.append([tree.left.text, tree.right.text, tree.relation])
        pp += extr_pairs(tree.left)
        pp += extr_pairs(tree.right)
    return pp

def extr_pairs_forest(forest):
    pp = []
    for tree in forest:
        pp += extr_pairs(tree)
    return pp

def _check_snippet_pair_in_dataset(left_snippet, right_snippet):
    left_snippet = left_snippet.strip()
    right_snippet = right_snippet.strip()
    return ((((gold.snippet_x == left_snippet) & (gold.snippet_y == right_snippet)).sum(axis=0) != 0) 
            or ((gold.snippet_y == left_snippet) & (gold.snippet_x == right_snippet)).sum(axis=0) != 0)

def _not_parsed_as_in_gold(parsed_pairs: pd.DataFrame, gold: pd.DataFrame):
    for key in text_html_map.keys():
            parsed_pairs['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
            parsed_pairs['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)
            
    for key in text_html_map.keys():
            gold['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
            gold['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)
    
    tmp = pd.merge(gold, parsed_pairs, on=['snippet_x', 'snippet_y'], how='left', suffixes=('_gold', '_parsed'))
    return tmp[pd.isnull(tmp.category_id_parsed)]

def extr_edus(tree):
    edus = []
    if tree.left:
        edus += extr_edus(tree.left)
        edus += extr_edus(tree.right)
    else:
        edus.append(tree.text)
    return edus

def eval_segmentation(trees, gold_edus):
    true_predictions = 0
    all_predicted = 0
    
    for tree in trees:
        pred_edus = extr_edus(tree)
        all_predicted += len(pred_edus)
    
        for pred_edu in pred_edus:
            for key, value in text_html_map.items():
                pred_edu = pred_edu.replace(key, value)
                
            if pred_edu.strip() in gold_edus:
                true_predictions += 1
            
    pr = true_predictions / all_predicted
    re = true_predictions / len(gold_edus)
    f1 = 2 * pr * re / (pr + re + 1e-5)
    return {'pr': pr, 
            're': re,
            'f1': f1}

def eval_pipeline(trees, gold_edus, gold_pairs):
    parsed_pairs = extr_pairs_forest(trees)
    return {
        'segmentation': eval_segmentation(trees, gold_edus),
        'unlabeled_tree_building': metric_parseval(parsed_pairs, gold_pairs),
        'labeled_tree_building': metric_parseval(parsed_pairs, gold_pairs, labeled=True)
    }

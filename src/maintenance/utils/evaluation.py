import pandas as pd


def metric_parseval(parsed_pairs, gold):
    parsed_strings = []
    for i in parsed_pairs.index:
        parsed_strings.append(parsed_pairs.loc[i, 'snippet_x'] + ' ' + parsed_pairs.loc[i, 'snippet_y'])
    parsed_strings = set(parsed_strings)
    
    gold_strings_1 = []
    for i in gold.index:
        gold_strings_1.append(gold.loc[i, 'snippet_x'] + ' ' + gold.loc[i, 'snippet_y'])
    gold_strings_1 = set(gold_strings_1)
    
    gold_strings_2 = []
    for i in gold.index:
        gold_strings_2.append(gold.loc[i, 'snippet_y'] + ' ' + gold.loc[i, 'snippet_x'])
    gold_strings_2 = set(gold_strings_2)
    
    true_pos = len(gold_strings_1 & parsed_strings) + len(gold_strings_2 & parsed_strings)
    all_parsed = len(parsed_strings)
    all_gold = len(gold_strings_1)
    
    return true_pos, all_parsed, all_gold
    
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
    tmp = pd.merge(gold, parsed_pairs, on=['snippet_x', 'snippet_y'], how='left', suffixes=('_gold', '_parsed'))
    return tmp[pd.isnull(tmp.category_id_parsed)]

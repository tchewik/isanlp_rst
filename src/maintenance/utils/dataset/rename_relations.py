def rename_relations(data):
    target_map = {
        'antithesis': 'attribution',
        'cause': 'cause-effect',
        'conclusion': 'restatement',
        'interpretation': 'interpretation-evaluation',
        'evaluation': 'interpretation-evaluation',
        'motivation': 'condition',
    }

    relation_map = {
        'effect_SN': 'cause-effect_NS',  # In essays, they are reversed
        'effect_NS': 'cause-effect_SN',
        'evidence_SN': 'preparation_SN',
        'restatement_SN': 'condition_SN',
        'restatement_NS': 'elaboration_NS',
        'solutionhood_NS': 'elaboration_NS',
        'preparation_NS': 'elaboration_NS',
        'concession_SN': 'preparation_SN',
        'evaluation_SN': 'preparation_SN',
        'elaboration_SN': 'preparation_SN',
        'background_SN': 'preparation_SN',
    }

    data = data[data.snippet_x.map(len) > 0]
    data = data[data.snippet_y.map(len) > 0]

    data['category_id'] = data['category_id'].replace(target_map, regex=False)

    data['relation'] = data['category_id'].map(lambda row: row) + '_' + data['order']
    data['relation'] = data['relation'].replace(relation_map, regex=False)

    return data

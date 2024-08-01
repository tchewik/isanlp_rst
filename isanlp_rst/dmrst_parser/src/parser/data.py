import numpy as np

# Fine-grained labels (51)
# RelationTable = ['adversative-antithesis_NS', 'adversative-antithesis_SN', 'adversative-concession_NS',
#                  'adversative-concession_SN', 'adversative-contrast_NN', 'attribution-negative_NS',
#                  'attribution-negative_SN', 'attribution-positive_NS', 'attribution-positive_SN', 'causal-cause_NS',
#                  'causal-cause_SN', 'causal-result_NS', 'causal-result_SN', 'context-background_NS',
#                  'context-background_SN', 'context-circumstance_NS', 'context-circumstance_SN',
#                  'contingency-condition_NS', 'contingency-condition_SN', 'elaboration-additional_NS',
#                  'elaboration-attribute_NS', 'evaluation-comment_NS', 'evaluation-comment_SN',
#                  'explanation-evidence_NS', 'explanation-evidence_SN', 'explanation-justify_NS',
#                  'explanation-justify_SN', 'explanation-motivation_NS', 'explanation-motivation_SN',
#                  'joint-disjunction_NN', 'joint-list_NN', 'joint-other_NN', 'joint-sequence_NN', 'mode-manner_NS',
#                  'mode-manner_SN', 'mode-means_NS', 'mode-means_SN', 'organization-heading_SN',
#                  'organization-phatic_NS', 'organization-phatic_SN', 'organization-preparation_SN',
#                  'purpose-attribute_NS', 'purpose-goal_NS', 'purpose-goal_SN', 'restatement-partial_NS',
#                  'restatement-partial_SN', 'restatement-repetition_NN', 'same-unit_NN', 'topic-question_SN',
#                  'topic-solutionhood_NS', 'topic-solutionhood_SN']

# Coarse-grained labels (29)
RelationTableGUM = [
    'adversative_NN', 'adversative_NS', 'adversative_SN', 'attribution_NS', 'attribution_SN', 'causal_NS',
    'causal_SN', 'context_NS', 'context_SN', 'contingency_NS', 'contingency_SN', 'elaboration_NS',
    'evaluation_NS', 'evaluation_SN', 'explanation_NS', 'explanation_SN', 'joint_NN', 'mode_NS', 'mode_SN',
    'organization_NS', 'organization_SN', 'purpose_NS', 'purpose_SN', 'restatement_NN', 'restatement_NS',
    'same-unit_NN', 'topic_SN']

# Coarse-grained labels (42)
RelationTableRSTDT = ['Elaboration_NS', 'Attribution_SN', 'Joint_NN', 'same-unit_NN',
       'Attribution_NS', 'Explanation_NS', 'Enablement_NS', 'Background_NS',
       'Evaluation_NS', 'Cause_NS', 'Contrast_SN', 'Contrast_NN',
       'Background_SN', 'Temporal_NN', 'Comparison_NN', 'Contrast_NS',
       'Topic-Change_NN', 'Manner-Means_NS', 'textual-organization_NN',
       'Temporal_NS', 'Condition_NS', 'Condition_SN', 'Cause_SN', 'Summary_NS',
       'Topic-Comment_NN', 'Cause_NN', 'Summary_NN', 'Evaluation_SN',
       'Temporal_SN', 'Explanation_SN', 'Enablement_SN', 'Topic-Comment_NS',
       'Comparison_NS', 'Elaboration_SN', 'Manner-Means_SN', 'Comparison_SN',
       'Summary_SN', 'Condition_NN', 'Topic-Comment_SN', 'Topic-Change_NS',
       'Evaluation_NN', 'Explanation_NN']

RelationTableRuRSTB = ['Joint_NN', 'Elaboration_NS', 'Contrast_NN', 'Attribution_SN', 'Interpretation-evaluation_NS',
                       'Preparation_SN', 'Cause-effect_SN', 'Sequence_NN', 'Cause-effect_NS',
                       'same-unit_NN', 'Condition_SN', 'Purpose_NS', 'Attribution_NS', 'Condition_NS',
                       'Comparison_NN', 'Concession_SN', 'Background_SN', 'Solutionhood_SN', 'Evidence_NS',
                       'Concession_NS', 'Interpretation-evaluation_SN', 'Restatement_NN', 'Concession_SN',
                       'Evidence_SN', 'Purpose_SN']

def getLabelOrdered(Original_Order):
    '''
    Get the right order of lable for stacks manner.
    E.g.
    [8,3,9,2,6,10,1,5,7,11,4] to [8,3,2,1,6,5,4,7,9,10,11]
    '''
    Original_Order = np.array(Original_Order)
    target = []
    stacks = ['root', Original_Order]
    while stacks[-1] != 'root':
        head = stacks[-1]
        if len(head) < 3:
            target.extend(head.tolist())
            del stacks[-1]
        else:
            target.append(head[0])
            temp = np.arange(len(head))
            top = head[temp[head < head[0]]]
            down = head[temp[head > head[0]]]
            del stacks[-1]
            if down.size > 0:
                stacks.append(down)
            if top.size > 0:
                stacks.append(top)

    return [x for x in target]


def nucs_and_rels(label_index, relation_table):
    relation = relation_table[label_index]
    label, nuclearities = relation.split('_')

    nuc_left, nuc_right = 'Nucleus', 'Nucleus'
    rel_left, rel_right = label, label

    if nuclearities == 'NS':
        nuc_right = 'Satellite'
        rel_left = 'span'

    elif nuclearities == 'SN':
        nuc_left = 'Satellite'
        rel_right = 'span'

    return nuc_left, nuc_right, rel_left, rel_right


class Data:
    def __init__(self, input_sentences, edu_breaks, decoder_input, relation_label,
                 parsing_breaks, golden_metric,
                 entity_ids=None, entity_position_ids=None,
                 sent_breaks=None, parents_index=None, sibling=None):
        """
        input_sentences (list[list[str]]): Subtokens for each document.
            ex.: [['▁A', 'e', 'sthetic', '▁Ap', 'preci', 'ation', '▁and', '▁Spanish', '▁Art', '▁:', ...] ...]
        edu_breaks (list[list[[int]]): Positions of each right EDU border.
            ex.: [[9, ...] ...] for example above
        decoder_input (list[list[int]])): ???
            ex.: [[0, 0, 2, 2, 3, 4, 6, ...] ...] for example above
        relation_label (list[list[int]]): Relation labels indices
            ex.: [[14, 7, 36, 36, 36, 14, ...] ...] for example above
        parsing_breaks (list[list[int]]): ???
            ex.: [[1, 0, 5, 2, 3, 4, ...] ...] for example above
        golden_metric (list[str]): document trees in the string format
            ex.: ['(1:Satellite=Background:2,3:Nucleus=span:74) (1:Nucleus=span:1,2:Satellite=Elaboration:2) ...', ...]
        parents_index (list[list[int]]): ???
            ex.: [[0, 73, 73, 73, 5, 5, ...] ...] for example above
        sibling (list[list[int]]): ???
            ex.: [[99, 99, 1, 99, 2, 3, ...] ...] for example above
        """
        self.input_sentences = input_sentences
        self.sent_breaks = sent_breaks
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.edu_breaks = edu_breaks
        self.decoder_input = decoder_input
        self.relation_label = relation_label
        self.parsing_breaks = parsing_breaks
        self.golden_metric = golden_metric
        self.parents_index = parents_index
        self.sibling = sibling

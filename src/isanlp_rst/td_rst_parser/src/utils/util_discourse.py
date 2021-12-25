#NOTE Our objective is to generate the new parent index and  sibling index
# because there is some problem in the original process
# new parent and sibling indices name: Training_ParentsIndex_new.pickle, Training_SiblingIndex_new.pickle
import os
import pickle
import random
import re
from copy import deepcopy
# from DataHandler import get_RelationAndNucleus
curr_path='/mnt/StorageDevice/Projects/Data/Parsing_data/pickle_data'
data_path_pc='/mnt/StorageDevice/Projects/Discourse_parsing/Parsing_data/pickle_data'
data_path_ntu2='/data/thomas/Projects/Data/Parsing_data/pickle_data'
if os.path.exists(curr_path):
	DATA_PATH=curr_path
elif os.path.exists(data_path_pc):
	DATA_PATH=data_path_pc
else:
	DATA_PATH=data_path_ntu2
RelationTable = ['attribution_NS', 'attribution_SN', 'background_NS',
       'cause-effect_NS', 'cause-effect_SN',
       'comparison_NN', 'concession_NS', 'condition_NS', 'condition_SN',
       'contrast_NN', 'elaboration_NS', 'evidence_NS',
       'interpretation-evaluation_NS', 'interpretation-evaluation_SN',
       'joint_NN', 'preparation_SN', 'purpose_NS', 'purpose_SN',
       'restatement_NN', 'same-unit_NN', 'sequence_NN', 'solutionhood_SN']


def Label2RelationAndNucleus(relation):
	'''
	'''

	# 39 relations

	temp = re.split(r'_', relation)
	if len(temp)==1:
		print(relation)
		input()
	sub1 = temp[0]
	sub2 = temp[1]

	if sub2 == 'NN':
		Nuclearity_left = 'Nucleus'
		Nuclearity_right = 'Nucleus'
		Relation_left = sub1
		Relation_right = sub1

	elif sub2 == 'NS':
		Nuclearity_left = 'Nucleus'
		Nuclearity_right = 'Satellite'
		Relation_left = 'span'
		Relation_right = sub1

	elif sub2 == 'SN':
		Nuclearity_left = 'Satellite'
		Nuclearity_right = 'Nucleus'
		Relation_left = sub1
		Relation_right = 'span'
	else:
		raise NotImplementedError
	return Nuclearity_left, Nuclearity_right, Relation_left, Relation_right

def RelationAndNucleus2Label(Nuclearity_left, Nuclearity_right, Relation_left, Relation_right):
	'''
	'''

	# 39 relations
	if Nuclearity_left == 'Nucleus' and Nuclearity_right == 'Nucleus':
		sub2= 'NN'
		assert Relation_left==Relation_right
		label= Relation_left+'_'+sub2
	elif Nuclearity_left == 'Nucleus' and Nuclearity_right == 'Satellite':
		sub2='NS'
		assert Relation_left == 'span'
		label= Relation_right +'_'+sub2
	elif Nuclearity_left == 'Satellite' and Nuclearity_right == 'Nucleus':
		sub2 = 'SN'
		assert Relation_right == 'span'
		label = Relation_left + '_' + sub2
	else:
		raise NotImplementedError
	return label
def parsing2goldmetric(parsing_order_self_pointing_token, parsing_label_self_pointing):
	goldmetric_list=[]
	for i in range(len(parsing_order_self_pointing_token)):
		parsing_order_i=parsing_order_self_pointing_token[i]
		parsing_label_i=parsing_label_self_pointing[i]
		if parsing_order_i[1]==parsing_order_i[2]==parsing_order_i[3]:
			assert parsing_label_i=='None'
		else:
			Nuclearity_left_i, Nuclearity_right_i, Relation_left_i, Relation_right_i=Label2RelationAndNucleus(parsing_label_i)
			goldmetric_token_i='('+ str(parsing_order_i[0]+1) + ':' + Nuclearity_left_i + '=' + \
			                        Relation_left_i +':'+ str(parsing_order_i[1]+1) + ',' + \
			                        str(parsing_order_i[2]+1) + ':' + Nuclearity_right_i + '=' + \
			                        Relation_right_i +':'+ str(parsing_order_i[3]+1) + ')'
			goldmetric_list.append(goldmetric_token_i)
	if len(goldmetric_list)==0:
		return 'NONE'
	else:
		return ' '.join(goldmetric_list)
def edu2token(golden_metric_edu, edu_break, sibling_relationship='left'):
	# stack = [(0,len(parsing_order))]
	assert sibling_relationship in ['left','both'], 'we only support left sibling relationship if existed or both side relationship'
	#NOTE
	#This part to generate parsing order and parsing label in term of edu
	# if golden_metric_edu is not 'NONE':
	if not(golden_metric_edu == 'NONE'):
		parsing_order_edu=[]
		parsing_label=[]
		golden_metric_edu_split=re.split(' ',golden_metric_edu)
		for each_split in golden_metric_edu_split:
			left_start, Nuclearity_left, Relation_left, left_end, \
			right_start, Nuclearity_right , Relation_right, right_end = re.split(':|=|,', each_split[1:-1])
			left_start=int(left_start)-1
			left_end=int(left_end)-1
			right_start=int(right_start)-1
			right_end=int(right_end)-1
			relation_label=RelationAndNucleus2Label(Nuclearity_left, Nuclearity_right, Relation_left, Relation_right)
			parsing_order_edu.append((left_start, left_end, right_start, right_end))
			parsing_label.append(relation_label)

		#Now we add to the parsing edu the part that corresponding to edu detection component (or when the case all the values
		# in parsing order for edu are the same
		parsing_order_self_pointing_edu = []
		stacks = ['__StackRoot__', parsing_order_edu[0]]
		while stacks[-1] is not '__StackRoot__':
			stack_head = stacks[-1]
			assert (len(stack_head) == 4)
			parsing_order_self_pointing_edu.append(stack_head)
			if stack_head[0] == stack_head[1] and stack_head[2] == stack_head[3] and stack_head[2] == stack_head[1]:
				del stacks[-1]
			elif stack_head[0] == stack_head[1] and stack_head[2] == stack_head[3]:
				stack_top = (stack_head[0], stack_head[0], stack_head[0], stack_head[0])
				stack_down = (stack_head[2], stack_head[2], stack_head[2], stack_head[2])
				del stacks[-1]
				stacks.append(stack_down)
				stacks.append(stack_top)
			elif stack_head[0] == stack_head[1]:
				stack_top = (stack_head[0], stack_head[0], stack_head[0], stack_head[0])
				stack_down = [x for x in parsing_order_edu if x[0] == stack_head[2] and x[3] == stack_head[3]]
				assert len(stack_down) == 1
				stack_down = stack_down[0]
				del stacks[-1]
				stacks.append(stack_down)
				stacks.append(stack_top)
			elif stack_head[2] == stack_head[3]:
				stack_top = [x for x in parsing_order_edu if x[0] == stack_head[0] and x[3] == stack_head[1]]
				stack_down = (stack_head[2], stack_head[2], stack_head[2], stack_head[2])
				assert len(stack_top) == 1
				stack_top = stack_top[0]
				del stacks[-1]
				stacks.append(stack_down)
				stacks.append(stack_top)
			else:
				stack_top = [x for x in parsing_order_edu if x[0] == stack_head[0] and x[3] == stack_head[1]]
				stack_down = [x for x in parsing_order_edu if x[0] == stack_head[2] and x[3] == stack_head[3]]
				assert len(stack_top) == 1 and len(stack_down) == 1
				stack_top = stack_top[0]
				stack_down = stack_down[0]
				del stacks[-1]
				stacks.append(stack_down)
				stacks.append(stack_top)
		parsing_label_self_pointing=[]
		for x in parsing_order_self_pointing_edu:
			if x in parsing_order_edu:
				parsing_label_self_pointing.append(parsing_label[parsing_order_edu.index(x)])
			else:
				parsing_label_self_pointing.append('None')
		edu_span=[]
		for i in range(len(edu_break)):
			if i==0:
				edu_span.append((0, edu_break[0]))
			elif i <len(edu_break):
				edu_span.append((edu_break[i-1]+1, edu_break[i]))
		parsing_order_self_pointing_token=[]
		for x in parsing_order_self_pointing_edu:
			if x[0]==x[1]==x[2]==x[3]:
				start_span=edu_span[x[0]][0]
				end_span=edu_span[x[0]][1]
				parsing_order_self_pointing_token.append((start_span, end_span, end_span, end_span))
			else:
				start_leftspan=edu_span[x[0]][0]
				end_leftspan=edu_span[x[1]][1]
				start_rightspan = edu_span[x[2]][0]
				end_rightspan = edu_span[x[3]][1]
				parsing_order_self_pointing_token.append((start_leftspan, end_leftspan, start_rightspan, end_rightspan))
		parsing_order_token=[]
		for x in parsing_order_edu:
			start_leftspan = edu_span[x[0]][0]
			end_leftspan = edu_span[x[1]][1]
			start_rightspan = edu_span[x[2]][0]
			end_rightspan = edu_span[x[3]][1]
			parsing_order_token.append((start_leftspan, end_leftspan, start_rightspan, end_rightspan))
	else:
		parsing_order_self_pointing_edu=[(0,0,0,0)]
		edu_span = [(0, edu_break[0])]
		parsing_order_edu=[]
		parsing_order_self_pointing_token=[(0, edu_break[0], edu_break[0], edu_break[0])]
		parsing_order_token=[]
		parsing_label_self_pointing=['None']
		parsing_label=['None']

	return {'parsing_order_self_pointing_edu': parsing_order_self_pointing_edu,
	        'parsing_order_edu': parsing_order_edu,
	        'parsing_order_self_pointing_token': parsing_order_self_pointing_token,
	        'parsing_order_token': parsing_order_token,
			'parsing_label_self_pointing': parsing_label_self_pointing,
			'parsing_label': parsing_label,
	        'edu_span': edu_span
	        }
def prepare_data(data_path=DATA_PATH):
	#process train data
	train_data={'parsing_order_self_pointing_edu': [],'parsing_order_edu': [],
	        'parsing_order_self_pointing_token': [],'parsing_order_token': [],
			'parsing_label_self_pointing': [], 'parsing_label': [],
	        'edu_span': [], 'sent': []
	        }
	Tr_InputSentences = pickle.load(open(os.path.join(data_path, "Training_InputSentences.pickle"), "rb"))
	Tr_EDUBreaks = pickle.load(open(os.path.join(data_path, "Training_EDUBreaks.pickle"), "rb"))
	Tr_GoldenMetric = pickle.load(open(os.path.join(data_path, "Training_GoldenLabelforMetric.pickle"), "rb"))
	for i in range(len(Tr_GoldenMetric)):
		Tr_golden_metric_edu_i=Tr_GoldenMetric[i][0]
		Tr_EDUBreaks_i=Tr_EDUBreaks[i]
		Tr_data_i=edu2token(golden_metric_edu=Tr_golden_metric_edu_i, edu_break=Tr_EDUBreaks_i)
		train_data['parsing_order_self_pointing_edu'].append(Tr_data_i['parsing_order_self_pointing_edu'])
		train_data['parsing_order_edu'].append(Tr_data_i['parsing_order_edu'])
		train_data['parsing_order_self_pointing_token'].append(Tr_data_i['parsing_order_self_pointing_token'])
		train_data['parsing_order_token'].append(Tr_data_i['parsing_order_token'])
		train_data['parsing_label_self_pointing'].append(Tr_data_i['parsing_label_self_pointing'])
		train_data['parsing_label'].append(Tr_data_i['parsing_label'])
		train_data['edu_span'].append(Tr_data_i['edu_span'])
		train_data['sent'].append(Tr_InputSentences[i])

	test_data = {'parsing_order_self_pointing_edu': [], 'parsing_order_edu': [],
	              'parsing_order_self_pointing_token': [], 'parsing_order_token': [],
	              'parsing_label_self_pointing': [], 'parsing_label': [],
	              'edu_span': [], 'sent': []
	              }
	Test_InputSentences = pickle.load(open(os.path.join(data_path, "Testing_InputSentences.pickle"), "rb"))
	Test_EDUBreaks = pickle.load(open(os.path.join(data_path, "Testing_EDUBreaks.pickle"), "rb"))
	Test_GoldenMetric = pickle.load(open(os.path.join(data_path, "Testing_GoldenLabelforMetric.pickle"), "rb"))
	Test_golden_metric_edu = [x[0] for x in Test_GoldenMetric]
	for i in range(len(Test_GoldenMetric)):
		Test_golden_metric_edu_i=Test_GoldenMetric[i][0]
		Test_EDUBreaks_i=Test_EDUBreaks[i]
		Test_data_i=edu2token(golden_metric_edu=Test_golden_metric_edu_i, edu_break=Test_EDUBreaks_i)
		test_data['parsing_order_self_pointing_edu'].append(Test_data_i['parsing_order_self_pointing_edu'])
		test_data['parsing_order_edu'].append(Test_data_i['parsing_order_edu'])
		test_data['parsing_order_self_pointing_token'].append(Test_data_i['parsing_order_self_pointing_token'])
		test_data['parsing_order_token'].append(Test_data_i['parsing_order_token'])
		test_data['parsing_label_self_pointing'].append(Test_data_i['parsing_label_self_pointing'])
		test_data['parsing_label'].append(Test_data_i['parsing_label'])
		test_data['edu_span'].append(Test_data_i['edu_span'])
		test_data['sent'].append(Test_InputSentences[i])
	return train_data, test_data
def convert_data(data_raw, vocab_dict):
	transform_data=deepcopy(data_raw)
	return transform_data
	label_self_pointing_data=transform_data['parsing_label_self_pointing']
	label_vocabulary=vocab_dict['label_vocab']
	transform_data['parsing_label_indices_self_pointing']=[]
	for label_data_i in label_self_pointing_data:
		transform_data_i=label_vocabulary.convert2idx([x for x in label_data_i])
		transform_data['parsing_label_indices_self_pointing'].append(transform_data_i)
	example_goldmetric_edu='(1:Satellite=Attribution:1,2:Nucleus=span:6) (2:Nucleus=span:2,3:Satellite=Enablement:6) (3:Nucleus=span:3,4:Satellite=Elaboration:6) (4:Nucleus=Temporal:4,5:Nucleus=Temporal:6) (5:Nucleus=span:5,6:Satellite=Enablement:6)'

if __name__ == '__main__':
	example_edu_break=[11, 15, 23, 29, 32, 40]
	example_goldmetric_edu='(1:Nucleus=span:3,4:Satellite=Attribution:4) (1:Nucleus=Joint:1,2:Nucleus=Joint:3) (2:Satellite=Attribution:2,3:Nucleus=span:3)'
	# example_edu_break=[6, 9, 22, 32]
	# example_goldmetric_edu='NONE'
	# example_edu_break=[7]
	# example_goldmetric_edu='NONE'
	# example_edu_break=[32]
	# example_goldmetric_edu='(1:Nucleus=Same-Unit:3,4:Nucleus=Same-Unit:5) (1:Nucleus=span:1,2:Satellite=Elaboration:3) (4:Satellite=Attribution:4,5:Nucleus=span:5) (2:Nucleus=span:2,3:Satellite=Enablement:3)'
	# example_edu_break=[13, 16, 25, 26, 37]
	# example_goldmetric_edu='(1:Satellite=Contrast:1,2:Nucleus=span:4) (2:Nucleus=span:2,3:Satellite=Explanation:4) (3:Nucleus=span:3,4:Satellite=Elaboration:4)'
	# example_edu_break=[4, 10, 18, 29]
	example_token_edu_data=edu2token(example_goldmetric_edu, example_edu_break)
	print(example_goldmetric_edu)
	print(example_edu_break)
	print('==================================================')
	print(example_token_edu_data['parsing_order_edu'])
	print(example_token_edu_data['parsing_order_self_pointing_edu'])
	print(example_token_edu_data['parsing_order_token'])
	print(example_token_edu_data['parsing_order_self_pointing_token'])
	print(example_token_edu_data['parsing_label'])
	print(example_token_edu_data['parsing_label_self_pointing'])
	print('==================================================')
	example_reconstruct_goldmetric_edu=parsing2goldmetric(parsing_order_self_pointing_token=example_token_edu_data['parsing_order_self_pointing_edu'],
	                                                      parsing_label_self_pointing=example_token_edu_data['parsing_label_self_pointing'])
	example_reconstruct_goldmetric_token=parsing2goldmetric(parsing_order_self_pointing_token=example_token_edu_data['parsing_order_self_pointing_token'],
	                                                      parsing_label_self_pointing=example_token_edu_data['parsing_label_self_pointing'])
	print(example_reconstruct_goldmetric_edu)
	print(example_reconstruct_goldmetric_token)
	print(example_goldmetric_edu==example_reconstruct_goldmetric_edu)
	train_data, test_data=prepare_data()


	# Tr_data = {'Tr_InputSentences': Tr_InputSentences, 'Tr_EDUBreaks': Tr_EDUBreaks, 'Tr_DecoderInput': Tr_DecoderInput,
	#            'Tr_RelationLabel': Tr_RelationLabel, 'Tr_ParsingBreaks': Tr_ParsingBreaks,
	#            'Tr_GoldenMetric': Tr_GoldenMetric}
	# random_check_list= random.sample(range(len(Tr_ParsingBreaks)),10)
	# print(random_check_list)
	# for i in range(len(Tr_ParsingBreaks)):
	# 	# check_input=['For', 'Ms.', 'Bogart', ',', 'who', 'initially', 'studied', 'and', 'directed', 'in', 'Germany', '(', 'and', 'cites', 'such', 'European', 'directors', 'as', 'Peter', 'Stein', ',', 'Giorgio', 'Strehler', 'and', 'Ariane', 'Mnouchkine', 'as', 'influences', ')', 'tends', 'to', 'stage', 'her', 'productions', 'with', 'a', 'Brechtian', 'rigor', '--', 'whether', 'the', 'text', 'demands', 'it', 'or', 'not', '.']
	# 	# if Tr_InputSentences[i]==check_input:
	# 	if i in random_check_list:
	# 		print(Tr_GoldenMetric[i])
	# 		print(Tr_ParsingBreaks[i])
	# 		print(Tr_EDUBreaks[i])
	# 		print(Tr_DecoderInput[i])
	# 		print(Tr_RelationLabel[i])
	# 		dict_info=edu2token(Tr_ParsingBreaks[i],Tr_RelationLabel[i],Tr_EDUBreaks[i])
	# 		# parsing_span=[(x[0],x[3]) for x in list_goldenmetric]
	# 		parsing_edu=dict_info['parsing_edu']
	# 		parent_dict=dict_info['parent_dict']
	# 		sibling_dict=dict_info['sibling_dict']
	# 		transform_goldenmetric=dict_info['transform_goldenmetric']
	# 		list_goldenmetric=dict_info['list_goldenmetric']
	# 		transform_parsing_span=dict_info['transform_parsing_span_edu']
	# 		transform_parsing_span_token=dict_info['transform_parsing_span_token']
	# 		ParsingIndex_Token=dict_info['ParsingIndex_Token']
	# 		DecoderInputIndex_Token=dict_info['DecoderInputIndex_Token']
	# 		ParentsIndex_Token=dict_info['ParentsIndex_Token']
	# 		SiblingIndex_Token=dict_info['SiblingIndex_Token']
	# 		RelationLabel_Token=dict_info['RelationLabel_Token']
	# 		transform_relation_nucleus_edulabel=dict_info['transform_relation_nucleus_edulabel']
	# 		transform_goldenmetric_edulabel_token=dict_info['transform_goldenmetric_edulabel_token']
	# 		list_goldenmetric_edulabel=dict_info['list_goldenmetric_edulabel']
	# 		RelationLabel_Token_string=dict_info['RelationLabel_Token_string']
	# 		transform_goldenmetric_token=dict_info['transform_goldenmetric_token']
	#
	# 		print('=====================================')
	# 		# print(parsing_edu)
	# 		# print([parent_dict[x] for x in parsing_edu])
	# 		# print([sibling_dict[x] for x in parsing_edu])
	# 		# print(list_goldenmetric)
	# 		# print(list_goldenmetric_edulabel)
	# 		# print(transform_goldenmetric)
	# 		# print(transform_parsing_span)
	# 		# print(transform_parsing_span_token)
	# 		print(ParsingIndex_Token)
	# 		print(DecoderInputIndex_Token)
	# 		print(ParentsIndex_Token)
	# 		print(SiblingIndex_Token)
	# 		print(RelationLabel_Token)
	# 		print(RelationLabel_Token_string)
	# 		print(transform_goldenmetric_edulabel_token)
	# 		print(transform_goldenmetric_token)
	# 		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

# relation_label=[21, 31, 7, 36]
# list_goldenmetric=[(0, 3, 4, 4), (0, 2, 3, 3), (0, 0, 1, 2), (1, 1, 2, 2)]
# stack=[(0,4)]
# # relation_label=[21, 31, 7]
# # list_goldenmetric=[(0, 1, 2, 3), (0, 0, 1, 1), (2, 2, 3, 3)]
# transform_goldenmetric = deepcopy(list_goldenmetric)
# trasform_relation_label=deepcopy(relation_label)
# count = 0
# for i_x, x in enumerate(list_goldenmetric):
# # for i_x, x in reversed(list(enumerate(list_goldenmetric))):
# 	if x[0] == x[1] and x[2] == x[3] and x[2] == x[1] + 1:
# 		transform_goldenmetric.insert(i_x + count + 1, (x[0], x[0], x[0], x[0]))
# 		transform_goldenmetric.insert(i_x + count + 2, (x[2], x[2], x[2], x[2]))
# 		count += 2
# 		# transform_goldenmetric.append( (x[0], x[0], x[0], x[0]))
# 		# transform_goldenmetric.append( (x[2], x[2], x[2], x[2]))
# 		# trasform_relation_label.append(-1)
# 		# trasform_relation_label.append(-1)
#
# 	elif x[0] == x[1]:
# 		transform_goldenmetric.insert(i_x + count + 1, (x[0], x[0], x[0], x[0]))
# 		count += 1
# 		# transform_goldenmetric.append( (x[0], x[0], x[0], x[0]))
# 		# trasform_relation_label.append(-1)
# 	elif x[2] == x[3]:
# 		transform_goldenmetric.insert(i_x + count + 1 + x[1]-x[0], (x[2], x[2], x[2], x[2]))
# 		count += 1
# 		# transform_goldenmetric.append((x[2], x[2], x[2], x[2]))
# 		# trasform_relation_label.append(-1)
# print(transform_goldenmetric)
# print(trasform_relation_label)



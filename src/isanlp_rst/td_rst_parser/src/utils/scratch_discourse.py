import os

import torch
import torch.nn as nn
from src.models import PointingDiscourseModel
from src.parsers.parser import Parser
from src.utils import Config, Dataset, Embedding
from src.utils.common import bos, eos, pad, unk
from src.utils.field import ChartDiscourseField, Field, RawField, SubwordField, ParsingOrderField
from src.utils.logging import get_logger, progress_bar
from src.utils.metric import BracketMetric
from src.utils.transform import DiscourseTree

path=''
min_freq=20
fix_len=20
data_path='/mnt/StorageDevice/Projects/Discourse_parsing/Parsing_data/pickle_data'
train={'sentences':os.path.join(data_path, "Training_InputSentences.pickle"),
       'edu_break':os.path.join(data_path, "Training_EDUBreaks.pickle"),
       'golden_metric':os.path.join(data_path, "Training_GoldenLabelforMetric.pickle")}
args = Config(**locals())
logger = get_logger(__name__)
logger.info("Build the fields")
WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
EDU_BREAK = RawField('edu_break')
GOLD_METRIC = RawField('golden_metric')
CHART = ChartDiscourseField('charts_discourse', pad=pad)
PARSINGORDER = ParsingOrderField('parsingorder')
transform = DiscourseTree(WORD=(WORD, FEAT), EDU_BREAK=EDU_BREAK, GOLD_METRIC=GOLD_METRIC, CHART=CHART, PARSINGORDER=PARSINGORDER)

train=Dataset(transform, train)
WORD.build(train, args.min_freq)
FEAT.build(train)
CHART.build(train)
train.build(5000,32, True)
bar = progress_bar(train.loader)
for words, feats, edu_break, golden_metric, (spans, labels), parsing_order in bar:
	break
print(edu_break[0])
print(words[0])
print(spans[0])
print(labels[0])
print(parsing_order[0])
# print(DiscourseTree.build_gold(edu_break[0], golden_metric[0]))
# check=[DiscourseTree.build_gold(edu_break_i, golden_metric_i) for edu_break_i, golden_metric_i in zip(edu_break, golden_metric)]
pred_spans=[_parsing_order[:,_parsing_order[2]!=_parsing_order[1]].tolist() for _parsing_order in parsing_order]
adjust_spans=[[(i,k,j , CHART.vocab.itos[label])
               for (i,k,j), label in zip(_span[:,_span[2]!=0].transpose(0,1).tolist(), _label[_span[2]!=0].tolist())]
              for _span,_label  in zip(spans, labels)]
# print(adjust_spans[0])
print(DiscourseTree.build(adjust_spans[0]))
print(DiscourseTree.build_gold(edu_break[0], golden_metric[0]))
# for i in range(len(adjust_spans)):
# 	if pred_spans[i]!=adjust_spans[i]:
# 		print(pred_spans[i])
# 		print(adjust_spans[i])
# 		break
######################
#decoding

#this one is to perform beam search the decoder in discourse parsing
import torch
import torch.nn as nn
import torch.nn.functional as F
beam_size=1
# sequence_length=[x+ 5 for x in range(64)]
sequence_length=[3,4,1,6,7]
hidden_size=6
mb_size=len(sequence_length)
enc_len=max(sequence_length)
num_hyp=1
example_embedding=nn.Embedding(num_embeddings=enc_len, embedding_dim=hidden_size)
# example_rnn=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=True)
# Last_Hiddenstates=(torch.zeros(2,mb_size, hidden_size),torch.zeros(2,mb_size, hidden_size))
example_rnn=nn.GRU(input_size=hidden_size,hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=True)
Last_Hiddenstates=torch.zeros(2,mb_size, hidden_size)
# print(example_embedding.weight.size())

all_scores=[]
all_rnn_state=[]
decoder_length=(torch.as_tensor(sequence_length,dtype=torch.int64)-1)
stacked_inputspan= torch.zeros(mb_size, num_hyp , 2 , enc_len, dtype=torch.int64)
stacked_parsing_order = torch.zeros(mb_size, num_hyp, 4, enc_len, dtype=torch.int64)
stacked_parsing_label = torch.zeros(mb_size, num_hyp, 1, enc_len, dtype=torch.int64)

for i in range(len(sequence_length)):
	stacked_inputspan[i,:, 1, 0] = sequence_length[i] - 1
hypothesis_scores=torch.zeros(mb_size, num_hyp)
# num_steps = 2
num_steps = enc_len - 1
# t=-1
for t in range(num_steps):
	print('step',t)
	# t=t+1
	# assert t<num_steps
	curr_inputspan = stacked_inputspan[:, :, :, t]
	curr_input_startpoint = curr_inputspan[:, :, 0]
	curr_input_endpoint = curr_inputspan[:, :, 1]
	curr_input_presentation=example_embedding(curr_input_startpoint)+ example_embedding(curr_input_endpoint)
	DecoderOutputs, New_Last_Hiddenstates = example_rnn(curr_input_presentation.view(mb_size * num_hyp, 1, -1),Last_Hiddenstates)
	range_hyp = torch.arange(enc_len, dtype=torch.int64).view(1, 1, enc_len).expand(mb_size,num_hyp,enc_len)
	mask_hyp = ~((range_hyp >= curr_input_startpoint.unsqueeze(-1).expand(range_hyp.size())) * (
				range_hyp <= curr_input_endpoint.unsqueeze(-1).expand(range_hyp.size())))
	mask_hyp_length = (t >= decoder_length).view(mb_size,1,1).expand(mb_size,num_hyp,enc_len)
	mask_hyp_no_parsing= ((0 == curr_input_startpoint.unsqueeze(-1).expand(range_hyp.size())) * (
				0 == curr_input_endpoint.unsqueeze(-1).expand(range_hyp.size())))
	#NOTE synthetic_data
	pointing_prod=torch.Tensor(mb_size, num_hyp, enc_len).uniform_()
	# label_prob=torch.Tensor(mb_size, num_hyp, enc_len, enc_len, label_size).uniform_()
	all_scores.append(pointing_prod)
	pointing_scores = F.log_softmax(pointing_prod, dim=-1)
	# label_scores = F.log_softmax(label_prob)
	# pointing_scores= pointing_scores.masked_fill(mask_hyp, 2*pointing_scores.min())
	pointing_scores= pointing_scores.masked_fill(mask_hyp, float('-inf'))
	pointing_scores= pointing_scores.masked_fill(mask_hyp_length, 0)
	pointing_scores= pointing_scores.masked_fill(mask_hyp_no_parsing, 0)
	#just fill the mask with very negative number
	#NOTE ###################
	hypothesis_scores = hypothesis_scores.unsqueeze(2) + pointing_scores
	hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(mb_size, -1), dim=1, descending=True)
	prev_num_hyp=num_hyp
	num_hyp= (~mask_hyp).view(mb_size, -1).sum(dim=1).max().clamp(max=beam_size).item()
	hypothesis_scores = hypothesis_scores[:, :num_hyp]
	hyp_index = hyp_index[:, :num_hyp]
	base_index = hyp_index / enc_len
	split_index = hyp_index % enc_len
	hyp_startpoint = curr_input_startpoint.gather(dim=1, index=base_index)
	hyp_endpoint = curr_input_endpoint.gather(dim=1, index=base_index)
	based_index_expand=base_index.unsqueeze(-1).unsqueeze(-1).expand(mb_size, num_hyp, 2, enc_len)
	stacked_inputspan=stacked_inputspan.gather(dim=1, index=based_index_expand)
	based_index_parsing_order=base_index.unsqueeze(-1).unsqueeze(-1).expand(mb_size, num_hyp, 4, enc_len)
	stacked_parsing_order = stacked_parsing_order.gather(dim=1, index=based_index_parsing_order)
	stacked_parsing_order[:, :, 0, t]=hyp_startpoint
	stacked_parsing_order[:, :, 1, t]=torch.where((split_index >= hyp_startpoint) & (split_index <= hyp_endpoint),split_index,hyp_startpoint)
	stacked_parsing_order[:, :, 2, t]=torch.where((split_index >= hyp_startpoint) & (split_index < hyp_endpoint),split_index+1,hyp_endpoint)
	stacked_parsing_order[:, :, 3, t]=hyp_endpoint

	candidate_leftspan=stacked_inputspan[:, :, :, t+1]
	candidate_leftspan[:, :, 0] = torch.where((split_index > hyp_startpoint) & (split_index < hyp_endpoint),hyp_startpoint,candidate_leftspan[:,:,0])
	candidate_leftspan[:, :, 1] = torch.where((split_index > hyp_startpoint) & (split_index < hyp_endpoint),split_index, candidate_leftspan[:, :, 1])
	stacked_inputspan[:, :, :, t + 1] =candidate_leftspan

	position_rightspan = (t+split_index+1-hyp_startpoint).clamp(max=enc_len-1)
	position_rightspan_expand=position_rightspan.unsqueeze(-1).unsqueeze(-1).expand(mb_size, num_hyp,2,1)
	candidate_rightspan= stacked_inputspan.gather(dim=3, index=position_rightspan_expand).squeeze(-1)
	candidate_rightspan[:,:,0]=torch.where(1+split_index<hyp_endpoint, 1+split_index,candidate_rightspan[:,:,0])
	candidate_rightspan[:,:,1]=torch.where(1+split_index<hyp_endpoint, hyp_endpoint,candidate_rightspan[:,:,1])
	stacked_inputspan.scatter_(dim=3, index=position_rightspan_expand,src=candidate_rightspan.unsqueeze(-1))
	# print(stacked_inputspan)
	# print(stacked_parsing_order)
	# print(hypothesis_scores)
	batch_index = torch.arange(mb_size, dtype=torch.int64).view(mb_size, 1)
	hx_index = (base_index + batch_index * prev_num_hyp).view(mb_size * num_hyp)
	if isinstance(New_Last_Hiddenstates, tuple):
		new_hx, new_cx = New_Last_Hiddenstates
		new_hx = new_hx[:, hx_index]
		new_cx = new_cx[:, hx_index]
		hx, cx = Last_Hiddenstates
		last_hx=torch.where(hyp_endpoint==0, hx, new_hx)
		last_cx=torch.where(hyp_endpoint==0, cx, new_cx)
		Last_Hiddenstates = (last_hx, last_cx)
	else:
		New_Last_Hiddenstates = New_Last_Hiddenstates[:, hx_index]
		Last_Hiddenstates = torch.where(hyp_endpoint == 0, Last_Hiddenstates, New_Last_Hiddenstates)
	all_rnn_state.append(Last_Hiddenstates)

final_stacked_inputspan=stacked_inputspan[:,0,:,:-1]
final_stacked_parsing_order=stacked_parsing_order[:,0,:,:-1]
# padding_nonparsing_mask=~(final_stacked_parsing_order[:,3,:]==0).to(torch.long)
# padding_nonparsing_mask=final_stacked_parsing_order[:,3,:].ne(0).to(torch.long)
padding_nonparsing_mask=final_stacked_parsing_order[:,3,:].ne(final_stacked_parsing_order[:, 2, :]).to(torch.long)
_, padding_nonparsing_mask_index = torch.sort(padding_nonparsing_mask, dim=1, descending=True)
final_parsing_order=final_stacked_parsing_order.gather(dim=-1, index= padding_nonparsing_mask_index.unsqueeze(1).expand(mb_size, 4, num_steps))
final_parsing_length=padding_nonparsing_mask.sum(dim=-1)
max_parsing_length = int(final_parsing_length.max())
final_parsing_order = final_parsing_order[:,:,:max_parsing_length]
# print(final_stacked_inputspan)
print(final_stacked_parsing_order)
print(final_parsing_order)
# print(final_parsing_length)
print(hypothesis_scores)

# for t in range(num_steps):
# 	print(F.log_softmax(all_scores[t], dim=-1))
# padded_vector=torch.zeros((1,1,1), dtype=final_stacked_parsing_order.dtype)
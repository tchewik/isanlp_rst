import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
# from src.models import PointingConstituencyZhModel
from src.parsers.parser import Parser
from src.utils import Config, Dataset, Embedding
from src.utils.common import bos, eos, pad, unk
from src.utils.field import ChartField, Field, RawField, SubwordField, ParsingOrderField
from src.utils.logging import get_logger, progress_bar, init_logger
from src.utils.metric import BracketMetric
from src.utils.transform import TreeZh


train='/mnt/StorageDevice/Projects/Data/Treebank/ctb_clean/train.clean.txt'
# train='/mnt/StorageDevice/Projects/Data/WSJ_parsing_clean/23.auto.clean'
dev='/mnt/StorageDevice/Projects/Data/Treebank/ctb_clean/dev.clean.txt'
test='/mnt/StorageDevice/Projects/Data/Treebank/ctb_clean/test.clean.txt'
fix_len=1000000
batch_size=5000
buckets=8
min_freq=2
args = Config(**locals())
WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
TREE = RawField('trees')
CHART = ChartField('charts')
PARSINGORDER = ParsingOrderField('parsingorder')
transform = TreeZh(WORD=(WORD, FEAT), TREE=TREE, CHART=CHART, PARSINGORDER=PARSINGORDER)
train = Dataset(transform, args.train)
WORD.build(train, args.min_freq)
FEAT.build(train)
CHART.build(train)
train.build(args.batch_size, args.buckets, True, dist.is_initialized())

for words, feats, trees, (spans, labels), parsingorders in progress_bar(train.loader):
	break
print([WORD.vocab[x] for x in words[0].tolist() if x!=0])
print(len([WORD.vocab[x] for x in words[0].tolist() if x!=0]))

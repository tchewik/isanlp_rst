from nltk import Tree
file_1='/mnt/StorageDevice/Projects/Data/Treebank/ctb_clean/train.txt'
file_2='/mnt/StorageDevice/Projects/Data/Treebank/ctb_clean/train.clean.txt'
with open(file_1) as f:
	data_1=f.readlines()
data_1=[x.strip() for x in data_1]

with open(file_2) as f:
	data_2=f.readlines()
data_2=[x.strip() for x in data_2]
for i in range(len(data_1)):
	tree_1=Tree.fromstring(data_1[i])
	tree_2=Tree.fromstring(data_2[i])
	if tree_1!=tree_2:
		break
import pickle
a=pickle.load(open("/mnt/StorageDevice/Projects/Data/Treebank/ctb_clean/parsed.pkl","rb"))
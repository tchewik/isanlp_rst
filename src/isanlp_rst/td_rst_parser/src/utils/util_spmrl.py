import nltk
from nltk import Tree
def custom_chomsky_normal_form(
    tree, factor="right",dummy_label_manipulating="parent", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^"
):
	# assume all subtrees have homogeneous children
	# assume all terminals have no siblings

	# A semi-hack to have elegant looking code below.  As a result,
	# any subtree with a branching factor greater than 999 will be incorrectly truncated.
	if horzMarkov is None:
		horzMarkov = 999

	# Traverse the tree depth-first keeping a list of ancestor nodes to the root.
	# I chose not to use the tree.treepositions() method since it requires
	# two traversals of the tree (one to get the positions, one to iterate
	# over them) and node access time is proportional to the height of the node.
	# This method is 7x faster which helps when parsing 40,000 sentences.
	tree = tree.copy(True)

	nodeList = [(tree, [tree.label()])]
	while nodeList != []:
		node, parent = nodeList.pop()
		if isinstance(node, Tree):

			# parent annotation
			parentString = ""
			originalNode = node.label()
			if vertMarkov != 0 and node != tree and isinstance(node[0], Tree):
				parentString = "%s<%s>" % (parentChar, "-".join(parent))
				node.set_label(node.label() + parentString)
				parent = [originalNode] + parent[: vertMarkov - 1]

			# add children to the agenda before we mess with them
			for child in node:
				nodeList.append((child, parent))

			# chomsky normal form factorization
			if len(node) > 2:
				childNodes = [child.label() for child in node]
				nodeCopy = node.copy()
				node[0:] = []  # delete the children

				curNode = node
				numChildren = len(nodeCopy)
				for i in range(1, numChildren - 1):
					if factor == "right":
						if dummy_label_manipulating=="parent":
							newHead = "%s%s<%s>%s" % (
								originalNode,
								childChar,
								"-".join(
									childNodes[i: min([i + horzMarkov, numChildren])]
								),
								parentString,
							)  # create new head
						elif dummy_label_manipulating=="universal":
							newHead='|<>'
						elif dummy_label_manipulating=="universal_node_unary":
							newHead='NODE|<>'
						newNode = Tree(newHead, [])
						curNode[0:] = [nodeCopy.pop(0), newNode]
					else:
						if dummy_label_manipulating=="parent":
							newHead = "%s%s<%s>%s" % (
								originalNode,
								childChar,
								"-".join(
									childNodes[max([numChildren - i - horzMarkov, 0]): -i]
								),
								parentString,
							)
						elif dummy_label_manipulating=="universal":
							newHead='|<>'
						elif dummy_label_manipulating=="universal_node_unary":
							newHead='NODE|<>'

						newNode = Tree(newHead, [])
						curNode[0:] = [newNode, nodeCopy.pop()]

					curNode = newNode

				curNode[0:] = [child for child in nodeCopy]
	return tree

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
    }
# def preprocess_spmrl(text):
# 	if text in BERT_TOKEN_MAPPING:
# 		return BERT_TOKEN_MAPPING[text]
# 	return text
def preprocess_spmrl(word):
	word = BERT_TOKEN_MAPPING.get(word, word)
	# This un-escaping for / and * was not yet added for the
	# parser version in https://arxiv.org/abs/1812.11760v1
	# and related model releases (e.g. benepar_en2)
	word = word.replace('\\/', '/').replace('\\*', '*')
	# Mid-token punctuation occurs in biomedical text
	word = word.replace('-LSB-', '[').replace('-RSB-', ']')
	word = word.replace('-LRB-', '(').replace('-RRB-', ')')
	if word == "n't":
		word = "'t"
	return word


def clean_leaves(tree):
	tree=tree.copy()
	for i in range(len(tree.leaves())):
		leaf=tree[tree.leaf_treeposition(i)]
		if leaf in BERT_TOKEN_MAPPING:
			tree[tree.leaf_treeposition(i)]=BERT_TOKEN_MAPPING[leaf]
	return tree




import nltk
from nltk import Tree
#Redefine tree method spmrl
def binarize(tree, binarize_direction='left',dummy_label_manipulating='parent'):
	assert binarize_direction in ['left', 'right'],f"We only support left/right direction here"
	assert dummy_label_manipulating in ['parent', 'universal', 'universal_node_unary'],f"We only support parent/universal direction here"
	tree = tree.copy(True)
	nodes = [tree]
	while nodes:
		node = nodes.pop()
		if isinstance(node, nltk.Tree):
			nodes.extend([child for child in node])
			if len(node) > 1:
				for i, child in enumerate(node):
					if not isinstance(child[0], nltk.Tree):
						if dummy_label_manipulating=='parent':
							node[i] = nltk.Tree(f"{node.label()}|<>", [child])
						elif dummy_label_manipulating=='universal':
							node[i] = nltk.Tree(f"|<>", [child])
						elif dummy_label_manipulating=='universal_node_unary':
							node[i] = nltk.Tree(f"UNARY|<>", [child])
	tree=custom_chomsky_normal_form(tree, binarize_direction,dummy_label_manipulating, 0, 0)
	tree.collapse_unary()
	return tree
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
if __name__ == "__main__":
	example_tree_string = "(TOP (S (NP (_ She)) (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)) (ADJ (_ alone))))) (_ .)))"
	# example_tree_string="(TOP (S (NP (_ She)) (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis))))) (_ .)))"
	example_tree = nltk.Tree.fromstring(example_tree_string)
	binarize_direction='left'
	example_binarized_tree_universal= binarize(example_tree,binarize_direction=binarize_direction,dummy_label_manipulating='universal')
	example_binarized_tree_parent= binarize(example_tree,binarize_direction=binarize_direction,dummy_label_manipulating='parent')
	example_binarized_tree_universal_node_unary= binarize(example_tree,binarize_direction=binarize_direction,dummy_label_manipulating='universal_node_unary')
	example_tree.pretty_print()
	example_binarized_tree_parent.pretty_print()
	example_binarized_tree_universal.pretty_print()
	example_binarized_tree_universal_node_unary.pretty_print()

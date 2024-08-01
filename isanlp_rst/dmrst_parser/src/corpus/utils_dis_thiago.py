import os

import numpy as np
# from nltk.tokenize import treebank

from . import data
from .utils_rs3 import CustomTokenizer

TOKENIZER = CustomTokenizer()

# Modify the name of the RST DT files if in the list, for match with PTB
file_mapping = {'file1': 'wsj_0764',
                'file2': 'wsj_0430',
                'file3': 'wsj_0766',
                'file4': 'wsj_0778',
                'file5': 'wsj_2172'}


# ----------------------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------------------
def convert_parens_in_rst_tree_str(rst_tree_str):
    '''
    Deal with the parenthesis present in the text of some EDUs
    '''
    new_tree = ''
    i = 0
    while i < len(rst_tree_str):
        c = rst_tree_str[i]
        if rst_tree_str[i:i + 13] == "text <s><EDU>":
            end_text = False
            cur_str = rst_tree_str[i:i + 13]
            j = i + 13
            while rst_tree_str[j:j + 6] != "</EDU>":  # </EDU></s>
                cur_str += rst_tree_str[j]
                j += 1
            cur_str = cur_str.replace("(", "-LRB-")
            cur_str = cur_str.replace(")", "-RRB-")
            cur_str = cur_str.replace("[", "-LSB-")
            cur_str = cur_str.replace("]", "-RSB-")
            cur_str = cur_str.replace("{", "-LCB-")
            cur_str = cur_str.replace("}", "-RCB-")
            new_tree += cur_str
            i = j
        elif rst_tree_str[i:i + 10] == "text <EDU>":
            end_text = False
            cur_str = rst_tree_str[i:i + 10]
            j = i + 13
            while rst_tree_str[j:j + 6] != "</EDU>":  # </EDU></s>
                cur_str += rst_tree_str[j]
                j += 1
            cur_str = cur_str.replace("(", "-LRB-")
            cur_str = cur_str.replace(")", "-RRB-")
            cur_str = cur_str.replace("[", "-LSB-")
            cur_str = cur_str.replace("]", "-RSB-")
            cur_str = cur_str.replace("{", "-LCB-")
            cur_str = cur_str.replace("}", "-RCB-")
            new_tree += cur_str
            i = j
        else:
            new_tree += c
            i += 1
    return new_tree


def buildTree(text):
    """
    Build tree from *.dis file (from DPLP, by Yangfeng Ji)

    :type text: string
    :param text: RST tree read from a *.dis file
    """
    text = convert_parens_in_rst_tree_str(text)
    tokens = text.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
    eduIds = []
    queue = processtext(tokens)
    stack = []
    while queue:
        token = queue.pop(0)
        if token == ')':  # If ')', start processing
            content = []  # Content in the stack
            while stack:
                cont = stack.pop()
                if cont == '(':
                    break
                else:
                    content.append(cont)
            content.reverse()  # Reverse to the original order
            # Parse according to the first content word
            if len(content) < 2:
                raise ValueError("content = {}".format(content))
            label = content.pop(0)
            if label == 'Root':
                node = data.SpanNode(prop=label)
                node = createnode(node, content)
                stack.append(node)
            elif label == 'Nucleus':
                node = data.SpanNode(prop=label)
                node = createnode(node, content)
                stack.append(node)
            elif label == 'Satellite':
                node = data.SpanNode(prop=label)
                node = createnode(node, content)
                stack.append(node)
            elif label == 'span':
                # Merge
                beginindex = int(content.pop(0))
                endindex = int(content.pop(0))
                stack.append(('span', beginindex, endindex))
            elif label == 'leaf':
                # Merge
                eduindex = int(content.pop(0))
                checkcontent(label, content)
                stack.append(('leaf', eduindex, eduindex))
                eduIds.append(eduindex)
            elif label == 'rel2par':
                # Merge
                relation = content.pop(0)
                checkcontent(label, content)
                stack.append(('relation', relation))
            elif label == 'text':
                # Merge
                txt = createtext(content)
                stack.append(('text', txt))
            elif label == 'prom':
                # ignore
                continue
            else:
                raise ValueError("Unrecognized parsing label: {} \n\twith content = {}\n".format(label, content))
        else:
            # else, keep push into the stack
            stack.append(token)
    return stack[-1], eduIds


def createnode(node, content):
    """
    Assign value to an SpanNode instance (from DPLP, by Yangfeng Ji)

    :type node: SpanNode instance
    :param node: A new node in an RST tree

    :type content: list
    :param content: content from stack
    """
    for c in content:
        if isinstance(c, data.SpanNode):
            # Sub-node
            node.nodelist.append(c)
            c.pnode = node
        elif c[0] == 'span':
            node.eduspan = (c[1], c[2])
        elif c[0] == 'relation':
            node.relation = c[1]
        elif c[0] == 'leaf':
            node.eduspan = (c[1], c[1])
            node.nucspan = (c[1], c[1])
            node.nucedu = c[1]
        elif c[0] == 'text':
            node.text = c[1]
        else:
            raise ValueError("Unrecognized property: {}".format(c[0]))
    return node


def processtext(tokens):
    """
    Preprocessing token list for filtering '(' and ')' in text
    (from DPLP, by Yangfeng Ji)

    :type tokens: list
    :param tokens: list of tokens
    """
    identifier = '_!'
    within_text = False
    for (idx, tok) in enumerate(tokens):
        if identifier in tok:
            for _ in range(tok.count(identifier)):
                within_text = not within_text
        if ('(' in tok) and (within_text):
            tok = tok.replace('(', '-LB-')
        if (')' in tok) and (within_text):
            tok = tok.replace(')', '-RB-')
        tokens[idx] = tok
    return tokens


def createtext(lst):
    """ Create text from a list of tokens (from DPLP, by Yangfeng Ji)

    :type lst: list
    :param lst: list of tokens
    """
    newlst = []
    for item in lst:
        item = item.replace("_!", "")
        newlst.append(item)
    text = ' '.join(newlst)
    # Lower-casing, why?
    return text  # .lower()


def checkcontent(label, c):
    """ Check whether the content is legal (from DPLP, by Yangfeng Ji)

    :type label: string
    :param label: parsing label, such 'span', 'leaf'

    :type c: list
    :param c: list of tokens
    """
    if len(c) > 0:
        raise ValueError("{} with content={}".format(label, c))


# TODO merge all the binarization fcts, should be the same
def binarizeTreeRight(tree):
    """
    Convert a general RST tree to a binary RST tree (from DPLP, by Yangfeng Ji)

    :type tree: instance of SpanNode
    :param tree: a general RST tree
    """
    queue = [tree]
    while queue:
        node = queue.pop(0)
        queue += node.nodelist
        # Construct binary tree
        if len(node.nodelist) == 2:
            node.lnode = node.nodelist[0]
            node.rnode = node.nodelist[1]
            # Parent node
            node.lnode.pnode = node
            node.rnode.pnode = node
        elif len(node.nodelist) > 2:
            # Remove one node from the nodelist
            node.lnode = node.nodelist.pop(0)
            newnode = data.SpanNode(node.nodelist[0].prop)
            newnode.nodelist += node.nodelist
            # -- ADDED
            newnode.eduspan = tuple([newnode.nodelist[0].eduspan[0], newnode.nodelist[-1].eduspan[1]])
            # Right-branching
            node.rnode = newnode
            # Parent node
            node.lnode.pnode = node
            node.rnode.pnode = node
            # Add to the head of the queue, so the code will keep branching
            # until the nodelist size is 2
            queue.insert(0, newnode)
        # Clear nodelist for the current node
        node.nodelist = []
    return tree


def buildTreeThiago(text):
    """
    Build tree from *.thiago file

    :type text: string
    :param text: RST tree read from a *.dis file
    """
    text = convert_parens_in_rst_tree_str(text)
    tokens = text.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
    eduIds = []
    edus = {}
    allnodes = []
    root = None
    queue = processtext(tokens)
    stack = []
    while queue:
        token = queue.pop(0)
        if token == ')':
            content = []  # Content in the stack
            while stack:
                cont = stack.pop()
                if cont == '(':
                    break
                else:
                    content.append(cont)
            content.reverse()  # Reverse to the original order
            # Parse according to the first content word
            if len(content) < 2:
                raise ValueError("content = {}".format(content))
            label = content.pop(0)
            if label == 'Root':
                node = data.SpanNode(prop=label)
                node = createnodeThiago(node, content)
                root = node
                allnodes.append(node)
                stack.append(node)
            elif label == 'Nucleus':
                node = data.SpanNode(prop=label)
                node = createnodeThiago(node, content)
                allnodes.append(node)
                stack.append(node)
            elif label == 'Satellite':
                node = data.SpanNode(prop=label)
                node = createnodeThiago(node, content)
                allnodes.append(node)
                stack.append(node)
            elif label == 'span':
                # Merge
                beginindex = int(content.pop(0))
                endindex = int(content.pop(0))
                stack.append(('span', beginindex, endindex))
            elif label == 'leaf':
                # Merge
                eduindex = int(content.pop(0))
                checkcontent(label, content)
                stack.append(('leaf', eduindex, eduindex))
                eduIds.append(eduindex)
            elif label == 'rel2par':
                # Merge
                relation = content.pop(0)
                checkcontent(label, content)
                stack.append(('relation', relation))
            elif label == 'text':
                # Merge
                txt = createtext(content)
                stack.append(('text', txt))
                edus[eduindex] = txt
            elif label == 'prom' or label == 'schema':
                # ignore
                continue
            else:
                raise ValueError("Unrecognized parsing label: {} \n\twith content = {}\n".format(label, content))
        else:
            # else, keep push into the stack
            stack.append(token)
    if len(stack) == 0:
        return root, eduIds, allnodes, edus
    else:
        return stack[-1], eduIds, allnodes, edus


def createnodeThiago(node, content):
    """
    Assign value to a SpanNode instance

    :type node: SpanNode instance
    :param node: A new node in an RST tree

    :type content: list
    :param content: content from stack
    """
    for c in content:
        if isinstance(c, data.SpanNode):
            # Sub-node
            node.nodelist.append(c)
            c.pnode = node
        elif c[0] == 'span':
            node.eduspan = (c[1], c[2])
        elif c[0] == 'relation':
            node.relation = c[1]
        elif c[0] == 'leaf':
            node.eduspan = (c[1], c[1])
            node.nucspan = (c[1], c[1])
            node.nucedu = c[1]
        elif c[0] == 'text':
            node.text = c[1]
        else:
            raise ValueError("Unrecognized property: {}".format(c[0]))
    return node


def findNodeT(m, allnodes):
    for node in allnodes:
        if m.eduspan == node.eduspan:
            return node
    return None


def findDuplicate(allnodes, verbose=False):
    remove2kept = {}
    for i, n in enumerate(allnodes):
        for j, m in enumerate(allnodes):
            if i != j and n.eduspan == m.eduspan and not j in remove2kept.keys() and not j in remove2kept.values() and not i in remove2kept.keys() and not i in remove2kept.values():
                if verbose:
                    print("--KEPT", n.relation, n.prop, n.eduspan, n.text)
                    print("--RM", m.relation, m.prop, m.eduspan, m.text)
                # keep n, remove m, transfer all info
                n.nodelist.extend(m.nodelist)  # --- extend nodelist
                remove2kept[j] = i
                if n.relation == "span" and m.relation != "span":  # --- keep relation and nuc
                    n.relation = m.relation
                    n.prop = m.prop
                if n.text == None and m.text != None:  # --- keep text
                    n.text = m.text
                if verbose:
                    print("--Final KEPT", n.relation, n.prop, n.eduspan, n.text)
    # Need to replace all instance of the nodes removed in the nodelist of the other nodes
    for k, n in enumerate(allnodes):
        if i in remove2kept:
            if verbose:
                print("will be removed", n.eduspan)
        else:
            for c in n.nodelist:
                idx = allnodes.index(c)
                if idx in remove2kept:
                    if verbose:
                        print("Modified", n.eduspan, allnodes[idx].eduspan,
                              allnodes[idx].relation, allnodes[idx].prop, allnodes[idx].text)
                    idxc = n.nodelist.index(c)
                    n.nodelist.pop(idxc)
                    n.nodelist.append(allnodes[remove2kept[idx]])
                    if verbose:
                        print("Replaced by", allnodes[remove2kept[idx]].eduspan,
                              allnodes[remove2kept[idx]].relation,
                              allnodes[remove2kept[idx]].prop, allnodes[remove2kept[idx]].text)
    # Remove the nodes
    newnodes = []
    for i, n in enumerate(allnodes):
        if not i in remove2kept:
            n.nodelist = orderNodeList(n.nodelist)
            newnodes.append(n)
        else:
            if verbose:
                print("Removed", i, n.eduspan, n.prop, n.relation)
    allnodes = newnodes
    return newnodes


def cleanChildren(allnodes):
    for n in allnodes:
        for c in n.nodelist:
            if n.eduspan == c.eduspan:
                n.nodelist.remove(c)
                n.nodelist = orderNodeList(n.nodelist)
    return allnodes


def correctThiago(allnodes, verbose=False):
    ''' Deal with some issues when reading the Thiago files '''
    if verbose:
        print('\n', '-' * 30)
        for t in allnodes:
            print(t.eduspan, t.prop, t.relation, [r.eduspan for r in t.nodelist])
        print('\n', '-' * 30)
    # -- Search duplicated nodes, ie nodes with the same eduspan
    allnodes = findDuplicate(allnodes)
    if verbose:
        print('\n', '-' * 30)
        for t in allnodes:
            print(t.eduspan, t.prop, t.relation, [r.eduspan for r in t.nodelist])
        print('\n', '-' * 30)
    # -- Clean: remove children with the same eduspan as their parents
    allnodes = cleanChildren(allnodes)
    if verbose:
        print('\n', '-' * 30)
        for t in allnodes:
            print(t.eduspan, t.prop, t.relation, [r.eduspan for r in t.nodelist])
        print('\n', '-' * 30)

    return allnodes


def findMisplacedChildren(allnodes):
    misplaced_children = []
    for node in allnodes:
        node.nodelist = orderNodeList(node.nodelist)
        eduCovered = sorted(list(set([m.eduspan[0] for m in node.nodelist])))
        eduCovered.extend(list(set([m.eduspan[1] for m in node.nodelist])))
        eduCovered = sorted(list(set(eduCovered)))
        # the span retrieved from the children is not the same as eduspan
        if len(eduCovered) != 0 and tuple([min(eduCovered), max(eduCovered)]) != node.eduspan:
            for m in node.nodelist:
                # a child is outside the scope of the parent
                if m.eduspan[-1] < node.eduspan[0] or m.eduspan[0] > node.eduspan[-1]:
                    cnode = findNodeT(m, allnodes)
                    misplaced_children.append(cnode)
                    node.nodelist.remove(m)
    return misplaced_children


def findLonelyParent(allnodes):
    parents = []
    for node in allnodes:
        eduCovered = sorted(list(set([m.eduspan[0] for m in node.nodelist])))
        eduCovered.extend(list(set([m.eduspan[1] for m in node.nodelist])))
        eduCovered = sorted(list(set(eduCovered)))
        # If the span of the parent is not entirely covered by its children span
        if len(eduCovered) != 0 and tuple([min(eduCovered), max(eduCovered)]) != node.eduspan:
            parents.append(node)
    return parents


def bTree(allnodes, path, verbose=False):
    allnodes = correctThiago(allnodes)
    # Reorganize children: some children have the wrong parent according to their span
    # - Find the children that have wrong parents, rm from the parent nodelist
    misplaced_children = findMisplacedChildren(allnodes)
    misplaced_children.extend(findMisplacedChildren(allnodes))
    misplaced_children.extend(findMisplacedChildren(allnodes))
    # - find the parents missing a child
    parents = findLonelyParent(allnodes)
    if verbose:
        print('misplaced_children', [m.eduspan for m in misplaced_children])
        print('parents', [m.eduspan for m in parents])
    # - associate a misplaced child with its parent
    for node in parents:
        find_missing_eduspan(node, misplaced_children)
    if len(misplaced_children) != 0:
        parents = findLonelyParent(allnodes)
        for node in parents:
            find_missing_eduspan_backup(node, misplaced_children)
    for node in allnodes:
        node.nodelist = orderNodeList(node.nodelist)
    if verbose:
        print('\n', '-' * 30)
        for t in allnodes:
            print(t.eduspan, t.prop, t.relation, [r.eduspan for r in t.nodelist])
        print('\n', '-' * 30)
    root = [n for n in allnodes if n.prop == "Root"][0]
    return root


def find_missing_eduspan_backup(node, misplaced_children, verbose=False):
    if verbose:
        print("\nBACKUP MISSING CHILDREN\n", node.eduspan, [m.eduspan for m in node.nodelist])
        print('misplaced_children', [m.eduspan for m in misplaced_children])
    okChildren = []
    for c in misplaced_children:
        if c.eduspan[0] >= node.eduspan[0] and c.eduspan[1] <= node.eduspan[1]:
            if verbose:
                print("OK child", c.eduspan)
            node.nodelist.append(c)
            okChildren.append(c)
    if verbose:
        print([m.eduspan for m in node.nodelist])
    for c in okChildren:
        misplaced_children.remove(c)


def find_missing_eduspan(node, misplaced_children, verbose=False):
    if verbose:
        print("\nMISSING CHILDREN\n", node.eduspan, [m.eduspan for m in node.nodelist])
    eduCovered = sorted(list(set([m.eduspan[0] for m in node.nodelist])))
    eduCovered.extend(list(set([m.eduspan[1] for m in node.nodelist])))
    eduCovered = sorted(list(set(eduCovered)))
    if len(eduCovered) != 0 and tuple([min(eduCovered), max(eduCovered)]) != node.eduspan:
        if eduCovered[0] != node.eduspan[0]:
            if verbose:
                print("\tMissing", node.eduspan[0], eduCovered[0] - 1)
            child = findChild(node.eduspan[0], eduCovered[0] - 1, misplaced_children)
            if child != None:
                node.nodelist.append(child)
                misplaced_children.remove(child)
        elif len(eduCovered) == 1:
            if verbose:
                print("\tMissing, ", eduCovered[0] + 1, node.eduspan[1])
                child = findChild(eduCovered[0] + 1, node.eduspan[1], misplaced_children)
            if child != None:
                node.nodelist.append(child)
                misplaced_children.remove(child)
        elif eduCovered[1] != node.eduspan[1]:
            if verbose:
                print("\tMissing, ", eduCovered[1] + 1, node.eduspan[1])
            child = findChild(eduCovered[1] + 1, node.eduspan[1], misplaced_children)
            if child != None:
                node.nodelist.append(child)
                misplaced_children.remove(child)


def findChild(beg, end, misplaced_children):
    for c in misplaced_children:
        if c.eduspan[0] == beg and c.eduspan[1] == end:
            return c
    return None


def printThiagoList(tree):
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if len(node.nodelist) != 0:
            queue.extend([n for n in node.nodelist])


def printThiago(tree):
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if node.lnode != None and node.rnode != None:
            queue.append(node.lnode)
            queue.append(node.rnode)


# TODO merge all the binarization fcts, should be the same
def binarizeTreeRightThiago(tree, verbose=False):
    """
    Convert a general RST tree to a binary RST tree

    :type tree: instance of SpanNode
    :param tree: a general RST tree
    """
    queue = [tree]
    while queue:
        node = queue.pop(0)
        queue += node.nodelist
        # Construct binary tree
        if verbose:
            print('\n', node.eduspan, [n.eduspan for n in node.nodelist])
        if len(node.nodelist) == 2:
            if verbose:
                print("BIN", node.__str__())
            node.lnode = node.nodelist[0]
            node.rnode = node.nodelist[1]
            # Parent node
            node.lnode.pnode = node
            node.rnode.pnode = node
        elif len(node.nodelist) > 2:
            childrenRelations = [m.relation for m in node.nodelist]
            childrenNuclearity = [m.prop for m in node.nodelist]
            if verbose:
                print("NOT BIN", node.__str__())
            # Simple rules
            #             if childrenNuclearity[-1].lower() == 'satellite':
            #                 newnode = leftAttach( node )
            #             else: #last node = nucleus or span relation
            #                 newnode = rightAttach( node )
            # RST DT coherent rules
            if childrenNuclearity[-1].lower() == 'nucleus' or childrenRelations[-1].lower() == 'span' or snsPattern(
                    childrenRelations,
                    childrenNuclearity):  # last node = nucleus or span relation or specific pattern 'S N+ S'
                newnode = rightAttach(node)
            else:
                newnode = leftAttach(node)
        if node.lnode != None and node.rnode != None:
            queue.append(node.lnode)
            queue.append(node.rnode)
        node.nodelist = []
    return tree


def snsPattern(relations, nuclearity):
    # At least 3 nodes, a satellite at the beg and at the end, and only nuclei in between or span
    if len(nuclearity) < 3:
        return False
    if nuclearity[0].lower() != "satellite" or nuclearity[-1].lower() != "satellite":
        return False
    if len(np.unique(nuclearity[1:-1])) != 1:
        # should be the case where we have (S N S) S: first LA then come back to RA (N generally span)
        return False
    if np.unique(nuclearity[1:-1])[0].lower() != "nucleus" or not (
            len(np.unique(relations[1:-1])) == 1 and np.unique(relations[1:-1])[0].lower() == "span"):
        return False
    return True


def leftAttach(node):
    node.rnode = node.nodelist.pop(-1)
    newnode = data.SpanNode('Nucleus')
    # has to be a nucleus since we do it when the last/right node is a satellite and we have a NS rel
    newnode.nodelist += node.nodelist
    newnode.eduspan = tuple([newnode.nodelist[0].eduspan[0], newnode.nodelist[-1].eduspan[1]])
    # Left-branching
    node.lnode = newnode
    # Parent node
    node.lnode.pnode = node
    node.rnode.pnode = node
    return newnode


def rightAttach(node):
    node.lnode = node.nodelist.pop(0)
    newnode = data.SpanNode('Nucleus')
    # has to be a nucleus since we do it when the last/right node is a nucleus node.nodelist[0].prop
    newnode.nodelist += node.nodelist
    newnode.eduspan = tuple([newnode.nodelist[0].eduspan[0], newnode.nodelist[-1].eduspan[1]])
    # Right-branching
    node.rnode = newnode
    # Parent node
    node.lnode.pnode = node
    node.rnode.pnode = node
    return newnode


def orderNodeList(nodelist):
    newlist = sorted([n for n in nodelist], key=lambda x: x.eduspan[1])
    return newlist


# ----------------------------------------------------------------------------------
# READ FILES
# ----------------------------------------------------------------------------------
def getDisFiles(tbpath):
    disFiles = [os.path.join(tbpath, fname) for fname in os.listdir(tbpath) if fname.endswith(".dis")]
    eduFiles = [os.path.join(tbpath, fname) for fname in os.listdir(tbpath) if fname.endswith(".edus")]
    return disFiles, eduFiles


def findFile(eduFiles, basename_dis):
    ''' Retrieve the edu file corresponding to the basename_dis '''
    for _file in eduFiles:

        # basename = os.path.basename( _file ).split('.')[0]
        basename = os.path.basename(_file).replace('.out', '').replace('.dis', '').replace('.txt', '').replace('.edus',
                                                                                                               '')
        if basename_dis == basename:
            return _file
    return None


def readEduDoc(fedu, doc):
    """
    Read information from the edu file, and fill the fields tokendict and edudict of the document
    Use the TOKENIZER to get tokens

    :type fedu: string
    :param fedu: edu file name
    """
    if not os.path.isfile(fedu):
        raise IOError("File doesn't exist: {}".format(fedu))
    # EDU ids start at 1
    gidx, eidx, tokendict, edudict = 0, 1, {}, {}
    with open(fedu, 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            eduTxt = line
            edudict[eidx] = []
            # need to be tokenized, here simple nltk tokenization
            tokens = TOKENIZER.tokenize(line)
            for tok in tokens:
                tokendict[gidx] = tok
                edudict[eidx].append(gidx)
                gidx += 1
            eidx += 1
    doc.tokendict = tokendict
    doc.edudict = edudict
    return doc

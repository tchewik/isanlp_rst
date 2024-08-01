import os
import sys

# import matplotlib
import numpy as np

# matplotlib.use("TkAgg")


def backprop(tree, doc):
    """
    Starting from leaf node, propagating node information back to root node

    :type tree: SpanNode instance
    :param tree: an binary RST tree
    """
    treenodes = BFTbin(tree)
    treenodes.reverse()
    for node in treenodes:
        if (node.lnode is not None) and (node.rnode is not None):
            # Non-leaf node
            node.eduspan = __getspaninfo(node.lnode, node.rnode)
            node.text = __gettextinfo(doc.edudict, node.eduspan)
            if node.relation is None:
                # If it is a new node created by binarization
                if node.prop == 'Root':
                    pass
                else:
                    node.relation = __getrelationinfo(node.lnode,
                                                      node.rnode)
            node.form, node.nucspan = __getforminfo(node.lnode,
                                                    node.rnode)
        elif (node.lnode is None) and (node.rnode is not None):
            raise ValueError("Unexpected left node")
        elif (node.lnode is not None) and (node.rnode is None):
            raise ValueError("Unexpected right node")
        else:
            # Leaf node
            node.text = __gettextinfo(doc.edudict, node.eduspan)
    return treenodes[-1]


def __getspaninfo(lnode, rnode):
    """
    Get span size for parent node

    :type lnode,rnode: SpanNode instance
    :param lnode,rnode: Left/Right children nodes
    """
    try:
        eduspan = (lnode.eduspan[0], rnode.eduspan[1])
        return eduspan
    except TypeError:
        print(lnode.prop, rnode.prop)
        print(lnode.nucspan, rnode.nucspan)


def __getforminfo(lnode, rnode):
    """
    Get Nucleus/Satellite form and Nucleus span

    :type lnode,rnode: SpanNode instance
    :param lnode,rnode: Left/Right children nodes
    """
    if (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
        nucspan = lnode.eduspan
        form = 'NS'
    elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
        nucspan = rnode.eduspan
        form = 'SN'
    elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
        nucspan = (lnode.eduspan[0], rnode.eduspan[1])
        form = 'NN'
    else:
        print(lnode.prop, lnode.eduspan, rnode.prop, rnode.eduspan)
        raise ValueError("Form:" + lnode.prop)
    return form, nucspan


def __getrelationinfo(lnode, rnode):
    """
    Get relation information

    :type lnode,rnode: SpanNode instance
    :param lnode,rnode: Left/Right children nodes
    """
    if (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
        relation = lnode.relation
    elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
        relation = lnode.relation
    elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
        relation = rnode.relation
    else:
        print(lnode._id, rnode._id)
        print('lnode.prop = {}, lnode.eduspan = {}'.format(lnode.prop, lnode.eduspan))
        print('rnode.prop = {}, rnode.eduspan = {}'.format(rnode.prop, rnode.eduspan))
        raise ValueError("Error when find relation for new node")
    return relation


def __gettextinfo(edudict, eduspan):
    """
    Get text span for parent node

    :type edudict: dict of list
    :param edudict: EDU from this document

    :type eduspan: tuple with two elements
    :param eduspan: start/end of EDU IN this span
    """
    # text = lnode.text + " " + rnode.text
    text = []
    for idx in range(eduspan[0], eduspan[1] + 1, 1):
        text += edudict[idx]
    # Return: A list of token indices
    return text


def parse(tree):
    """
    Get parse tree in string format

        For visualization, use nltk.tree:
        from nltk.tree import Tree
        t = Tree.fromstring(parse)
        t.draw()
    """
    parse = getParse(tree, "")
    return parse


def getParse(tree, parse):
    """
    Get parse tree

    NOTE:
    - this fct expands the relations from the daughters to the node
    - the original fct extractrelation was not doing the mapping expected, removed

    :type tree: SpanNode instance
    :param tree: an binary RST tree

    :type parse: string
    :param parse: parse tree in string format
    """
    if (tree.lnode is None) and (tree.rnode is None):
        # Leaf node
        parse += " ( EDU " + str(tree.nucedu)
    else:
        parse += " ( " + tree.form
        # get the relation from its satellite node
        if tree.form == 'NN':
            if tree.rnode.relation == "span":
                parse += "-" + tree.lnode.relation
                # parse += "-" + extractrelation(tree.lnode.relation)
            else:
                parse += "-" + tree.rnode.relation
                # parse += "-" + extractrelation(tree.rnode.relation)
        elif tree.form == 'NS':
            parse += "-" + tree.rnode.relation
            # parse += "-" + extractrelation(tree.rnode.relation)
        elif tree.form == 'SN':
            parse += "-" + tree.lnode.relation
            # parse += "-" + extractrelation(tree.lnode.relation)
        else:
            raise ValueError("Unrecognized N-S form")
    if tree.lnode is not None:
        parse = getParse(tree.lnode, parse)
    if tree.rnode is not None:
        parse = getParse(tree.rnode, parse)
    parse += " ) "
    return parse


def getParseNobin(tree, parse):
    """
    Get parse tree

    NOTE: this fct expands the relations from the daughters to the node

    :type tree: SpanNode instance
    :param tree: an binary RST tree

    :type parse: string
    :param parse: parse tree in string format
    """
    parse += " ( " + str(tree._id) + "-" + str(tree.prop) + '-' + str(tree.relation)
    if len(tree.nodelist) != 0:
        for m in tree.nodelist:
            parse = getParseNobin(m, parse)
    parse += " ) "
    return parse


def BFTbin(tree):
    """
    Breadth-first treavsal on binary RST tree

    :type tree: SpanNode instance
    :param tree: an binary RST tree
    """
    queue = [tree]
    bft_nodelist = []
    while queue:
        node = queue.pop(0)
        #         print( "--> ", node, node.lnode )
        bft_nodelist.append(node)
        if node.lnode is not None:
            queue.append(node.lnode)
        if node.rnode is not None:
            queue.append(node.rnode)
    return bft_nodelist


def getRelation(label):
    """
    Get the relation from the label used in the RST DT.
    Could be stg like RELATION-s-e, with -s linked to the nuclearity, -e meaning
    that we have an embedded relation (TODO: check if previous studies keep this -e)
    """
    relation = label
    nuc = label.split('-')[0]
    if nuc.lower() in ["ns", "sn", "nn"]:
        relation = '-'.join(label.split('-')[1:])
    # Order matters: could have REL-n-e
    if relation.split('-')[-1].lower() == 'e':
        relation = '-'.join(relation.split('-')[:-1])
    if relation.split('-')[-1].lower() == 's':
        relation = '-'.join(relation.split('-')[:-1])
    if relation.split('-')[-1].lower() == 'n':
        relation = '-'.join(relation.split('-')[:-1])
    return relation, nuc


# ----------------------------------------------------------------------------------
# MAPPING
# ----------------------------------------------------------------------------------
def getLabelMapping(mappingFile, outputExt):
    labelsMapping = None
    nbClasses = -1
    if mappingFile != None:  # Modify the ext, add map+number of classes
        labelsMapping = readMapping(mappingFile)
        # TODO span seems to be kept as a relation, should have been removed when building the tree
        nbClasses = len(np.unique(labelsMapping.values()))
        outputExt = ".map" + str(nbClasses) + outputExt
    return labelsMapping, outputExt, nbClasses


def readMapping(mappingFile):
    '''
    Read a label mapping file and return a mapping (dict)

    :type mappingFile: file path
    :param mappingFile: the mapping file to read
    '''
    mapping = {}
    with open(mappingFile) as fin:
        _lines = fin.readlines()
        for l in _lines:
            l = l.strip()
            relation, group = l.split(' ')
            # original relation --> class
            mapping[relation] = group
    return mapping


def addLabels(tree, labelSet):
    """
    Fill the label set, used to check which relations exactly are used in the corpus
    """
    if tree == None:
        return
    for st in tree.subtrees():
        label = st.label()
        if not label.lower() == "edu":
            relation = getRelation(label)
            labelSet.add(relation)
            if "span" in relation[0].lower():
                print(relation)
                sys.exit("Still a span relation?? " + " ".join([c.label() for c in st]))


def countLabels(tree, rel2count):
    if tree == None:
        return
    for st in tree.subtrees():
        label = st.label()
        if not label.lower() == "edu":
            if label in rel2count:
                rel2count[label] += 1
            else:
                rel2count[label] = 1


def mapLabels(tree, mappingDict):
    '''
    Modify the labels in the tree according to a predefined mapping.

    :type tree: SpanNode
    :param tree: the RST tree to be modified

    :type mappingDict: dict of String
    :param mappingDict: mapping from the original relation to the mapped relation
    '''
    # Keep original label
    if mappingDict == None:
        return
    for st in tree.subtrees():
        label = st.label()
        if not label.lower() == "edu":
            relation, nuc = getRelation(label)
            if not relation in mappingDict:
                sys.exit("Unknow label: " + label + ", " + relation)
            # Keep nuclearity information
            mappedRelation = mappingDict[relation]
            if nuc.lower() in ["ns", "sn", "nn"]:
                mappedRelation = nuc + '-' + mappingDict[relation]
            st.set_label(mappedRelation)


def performMapping(tree, mappingDict):
    if mappingDict == None:
        print("No mapping found !")
        return
    for st in tree.subtrees():
        label = st.label()
        if not label.lower() == "edu":
            relation, nuc = getRelation(label)
            if not relation.lower() in mappingDict:
                sys.exit("Unknown label: " + label + ", " + relation)
            # Keep nuclearity information
            if nuc.lower() in ["ns", "sn", "nn"]:
                mappedRelation = nuc + '-' + mappingDict[relation.lower()]
            else:
                sys.exit("Unknown nuclearity value:", nuc)
            st.set_label(mappedRelation)


# ----------------------------------------------------------------------------------
# WRITE/DRAW/print
# ----------------------------------------------------------------------------------

def writeEdusFile(doc, ext, pathout):
    """
    Write files similar to the .edus files in the RST DT for the other RST Treebanks.

    doc: Document instance, contains info about the the tokens in each EDU
    forigin: the discourse file, keep the same basename and path
    ext: the original extension (ie .rs3 or .thiago) to be replaced by the new one (ie .edus)
    """
    edufile = os.path.join(pathout, os.path.basename(doc.path).replace(ext, ".edus"))
    f = open(edufile, 'w', encoding="utf8")
    for edu in doc.edudict:
        f.write(doc.edudict[edu].strip() + "\n")
    f.close()


def printBinTree(tree):
    ''' Can only be used after binarize (+backprop ev) but backprop only completed in parse  '''
    queue = [tree]
    while queue:
        n = queue.pop()
        if n.lnode != None:
            print("-->", n._id, n.relation, n.eduspan, n.prop, n.lnode._id, n.rnode._id)
            queue.append(n.lnode)
            queue.append(n.rnode)
        else:
            print("-->", n._id, n.relation, n.eduspan, n.prop)


def checkTree(tree, doc):
    """ Check the final tree (ie Nltk Tree)  """
    idEduOrdered = []
    for st in tree.subtrees():
        label = st.label()
        if label is None or label.lower() == 'none':
            print(doc.path, "\nUnknown label", st.label())
            return False
        if label.lower() == "edu":
            id_edu = [c for c in st][0]
            if id_edu is None or id_edu == 'None':
                print(doc.path, "\nEDU with None id", st.label())
                return False
            idEduOrdered.append(int(id_edu))  # id of the EDU
        else:
            prop = st.label().split('-')[0]
            if prop not in ['NS', 'NN', 'SN']:
                print(doc.path, "\nNode prop unknown", st.label())
                return False
            if len([c for c in st]) == 0:
                print(doc.path, "\nCDU w/o children", st.label())
                return False

    idEduOrdered = sorted(idEduOrdered)
    if idEduOrdered != list(range(1, len(idEduOrdered) + 1)):
        print(doc.path, "\nPb in EDU ids\n", idEduOrdered, "\n != ", list(range(1, len(idEduOrdered) + 1)))
        return False
    return True

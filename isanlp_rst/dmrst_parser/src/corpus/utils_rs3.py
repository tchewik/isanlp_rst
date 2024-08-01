from __future__ import print_function

import os
import sys

import numpy as np
from lxml import etree
# from nltk.tokenize import treebank
# from nltk.tokenize import WhitespaceTokenizer

from . import data

import nltk

class CustomTokenizer:
    def __init__(self):
        self.mwe_tokenizer = nltk.tokenize.MWETokenizer(separator='')
        self.mwe_tokenizer.add_mwe(('<', 'P', '>'))  # End of paragraph token annotated in RST-DT

    def tokenize(self, text):
        return self.mwe_tokenizer.tokenize(nltk.word_tokenize(text))

TOKENIZER = CustomTokenizer()

# ----------------------------------------------------------------------------------
# READ
# ----------------------------------------------------------------------------------
def parseXML(rs3file):
    for enc in ["utf8", "windows-1252"]:
        # ,"iso-8859-3","windows-1250", "gb2312"]:
        # spanish RST DT:"iso-8859-3", basque:None, CST:"windows-1252" !!1250 modifies the accents
        try:
            xml_parser = etree.XMLParser(encoding=enc)
            rs3_xml_tree = etree.parse(rs3file, xml_parser)
            doc_root = rs3_xml_tree.getroot()
            return doc_root, rs3_xml_tree
        except:
            continue
    try:
        xml_parser = etree.XMLParser()
        rs3_xml_tree = etree.parse(rs3file, xml_parser)
        doc_root = rs3_xml_tree.getroot()
        return doc_root, rs3_xml_tree
    except:
        sys.exit("Unable to read file: " + rs3file)
    return None, None


def getRelationsType(rs3_xml_tree):
    relations = {}
    for rel in rs3_xml_tree.iterfind('//header/relations/rel'):
        relName = rel.attrib['name'].replace(' ', '-')  # .encode("utf8")
        if 'type' in rel.attrib:
            if relName in relations:
                relations[relName].add(rel.attrib['type'])
            else:
                relations[relName] = set()
                relations[relName].add(rel.attrib['type'])
        else:
            continue  # Ignore 'schema' see for ex the Basque corpus SENTARG01-A1.rs3 sentiment schema
    for r in relations:
        relations[r] = list(relations[r])
    return relations


def readRS3Annotation(doc_root):
    '''
    a group's ``type`` attribute tells us whether the group
    node represents a span (of RST segments or other groups) OR
    a multinuc relation (i.e. it dominates several RST nucleii).

    a group's ``relname`` gives us the name of the relation between
    the group node and the group's parent node.
    '''
    eduList = []
    groupList = []
    # - EDU
    eduIds = []
    root = None
    potential_root = None  # see pb with i==0 and not parent..
    for i, segment in enumerate(doc_root.iter('segment')):
        # In SUMMIT, EDU 1 root in at least one doc (Parcial-38txts/CIENCIA_2005_6507.rs3)
        if i == 0 and not 'parent' in segment.attrib:  # Ignore the 1st for the Postdam corpus (title)
            potential_root = segment
            continue
        if 'parent' not in segment.attrib:
            continue
        # Get information for the current segment
        id_ = int(segment.attrib['id'])
        parent = int(segment.attrib['parent'])
        relname = segment.attrib['relname'].replace(' ', '-')
        edu = {"id": id_,
               "parent": parent,
               "relname": relname,
               "text": segment.text.strip().replace("\n", " "),
               "position": i}
        if not id_ in eduIds:  # Avoid the repeted segments
            eduList.append(edu)
            eduIds.append(id_)
    # - CDU
    cduIds = []
    for group in doc_root.iter('group'):
        id_ = int(group.attrib['id'])
        if 'parent' in group.attrib:
            cdu = {"id": id_,
                   "parent": int(group.attrib['parent']),
                   "relname": group.attrib['relname'].replace(' ', '-'),
                   "type": group.attrib['type']}
            if not id_ in cduIds:  # Avoid the repeted segments
                groupList.append(cdu)
                cduIds.append(id_)
        else:
            root = {"id": id_,
                    "parent": None,
                    "type": group.attrib['type']}
            if not id_ in cduIds:  # Avoid the repeted segments
                groupList.append(root)
                cduIds.append(id_)
    if root == None:
        # check if we have a potential root
        if potential_root != None:
            all_id = [d for d in eduIds]
            all_id.extend(cduIds)
            # create a fake root (should be removed later)
            root_id = sorted(all_id)[-1] + 1
            root = {"id": root_id,
                    "parent": None,
                    "type": "span"}

            edu = {"id": potential_root.attrib['id'],
                   "parent": root_id,
                   "text": segment.text.strip().replace("\n", " "),
                   "position": 0,
                   "relname": "span"}
            if not potential_root.attrib['id'] in eduIds:  # Avoid the repeted segments
                eduList.append(edu)
                eduIds.append(potential_root.attrib['id'])
    return eduList, groupList, root


# ----------------------------------------------------------------------------------
# EDU TEXT
# ----------------------------------------------------------------------------------
def retrieveEdu(tree, eduIds):
    """
    Read information from the edus
    Use the TOKENIZER to get tokens

    tokendict: each token in the document, id in the doc -> form
    edudict: each edu in the document, id in the tree -> list of token id
    """
    # Parcourir l arbre, recuperer les edu
    gidx, tokendict, edudict = 0, {}, {}
    for eidx in eduIds:  # with this list, we are sure to have ordered Edus
        eduNode = findNodeTree(eidx, tree)
        if eduNode == None:
            sys.exit("EDU not found: " + str(eidx))
        text = eduNode.text
        edudict[eidx] = []
        tokens = TOKENIZER.tokenize(text)
        for tok in tokens:
            # tok = unicode(tok, "utf8")
            tokendict[gidx] = tok
            edudict[eidx].append(gidx)
            gidx += 1
    return tokendict, edudict


# ----------------------------------------------------------------------------------
# TREE UTILS
# ----------------------------------------------------------------------------------
def buildNodes(eduList, groupList, rootDict, relations):
    '''
    Build a tree using the SpanNode class defined in DPLP

    For each DU, retrieve: id, relation, prop (ie nucleus or satellite)
    + for EDU: span
    We add the span for the CDU later, need to be sure to have the EDU covered ordered
    '''
    # Rename EDUs and CDUs to get continuous indexing for EDUs
    renameDus(eduList, groupList, rootDict)
    eduIds = [e["id"] for e in eduList]  # Ordered EDUs
    cduIds = [e["id"] for e in groupList]  # CDUs
    units = [e for e in eduList]
    units.extend(groupList)  # All DU
    root = data.SpanNode("Root")  # Root node
    root._id, root.eduSpan = rootDict["id"], tuple([eduIds[0], eduIds[-1]])  # Set the span for the root
    # Build the other nodes
    allNodes = [root]
    for e in units:
        # Check if the corresponding node has been built already
        # (even an EDU can be a parent for now)
        node = findNode(e["id"], allNodes)
        if node == None:
            newNode = data.SpanNode(None)  # Prop is unknown for now
            newNode._id, newNode.relation = e["id"], e["relname"]
            if e["id"] in eduIds:  # EDU ie isLeave
                newNode.text = e["text"]
                newNode.eduspan, newNode.nucspan = tuple([e["id"], e["id"]]), tuple([e["id"], e["id"]])
                newNode.nucedu = e["id"]
                newNode.position = e["position"]
            allNodes.append(newNode)
    # Update info for the parent
    updateParentNodes(allNodes, units, eduIds)
    updateNuclearityEDU(allNodes, units, relations)
    return root


def updateNuclearityEDU(allNodes, units, nucRelations):
    '''
    - if an EDU is annotated with a multinuc relation, it s a nucleus
    - if an EDU is annotated with a mononuc relation, it s a satellite and its parent a nucleus
    - else, if an edu is annotated with span, it s a nucleus
    '''
    for n in allNodes:
        # dictUnit["type"] == "multinuc" do not always correspond to all children being nucleus
        # when a node have more than 2 children annotated with different relations
        # for ex see Brazilian CST D2-C16
        # This info seems redundant with the types of the relation defined in the list, but somes rel can be both
        # -> Rely on the type of relation
        if n.relation in nucRelations and len(nucRelations[n.relation]) == 1:
            if nucRelations[n.relation][0] == "multinuc":
                n.prop = 'Nucleus'
            else:
                n.prop = 'Satellite'
                # in Discourse Graph, implies that the parent is a nucleus but not sure for the multi children cases
                parent = findParentNode(n, allNodes)
                if parent != None:
                    parent.prop = 'Nucleus'
        if n.prop == None and n.relation == "span":
            n.prop = 'Nucleus'
    for n in allNodes:
        if n.prop == None:
            if n.relation in nucRelations and "multinuc" in nucRelations[n.relation]:
                n.prop = 'Nucleus'
                print("PROP CHOSE NUC", n._id, n.prop, sorted([m._id for m in n.eduCovered]), n.relation)
            else:
                print("PROP UNDEF", n._id, n.prop, sorted([m._id for m in n.eduCovered]), n.relation)


def updateParentNodes(allNodes, units, eduIds):
    ''' Update eduCovered and nodelist '''
    for e in units:
        # retrieve the node
        node = findNode(e["id"], allNodes)
        # find the children
        for f in units:
            if f["parent"] == e["id"]:
                node.nodelist.append(findNode(f["id"], allNodes))
    for n in allNodes:
        # Begin directly with the children to not add the node itself if it s an EDU
        n.eduCovered = getEduCoveredChildren(n, eduIds)


def getEduCoveredChildren(node, eduIds):
    eduCovered = set()
    queue = [m for m in node.nodelist if m._id != node._id]  # seems to happen in the spanish RST DT
    while queue:
        n = queue.pop(0)
        if n._id in eduIds:
            eduCovered.add(n)
        for m in n.nodelist:
            queue.append(m)
    return list(eduCovered)


def renameDus(eduList, groupList, rootDict):
    '''
    Change the ID of the EDUs in order to have a continuous list of ID
    '''
    eduIds = [e["id"] for e in eduList]  # Ordered EDUs wr text
    cduIds = [e["id"] for e in groupList]  # CDU ids
    cduIds.append(rootDict["id"])
    newEdusIds = range(1, len(eduIds) + 1)  # New list of EDU ids
    if len(newEdusIds) != len(eduIds):
        sys.exit("Do not have the same number of old and new ids")
    if eduIds == newEdusIds:
        return eduList, groupList
    mappingEdus = {}  # Mapping depending on the position of the EDU in the text
    for i, e in enumerate(eduIds):
        mappingEdus[e] = newEdusIds[i]
    intersec = list(set(newEdusIds) & set(cduIds))  # Check with CDU ids, may be changed
    mappingCdus = {}  # Mapping CDUs with non existing ids
    if len(intersec) != 0:
        curId = sorted(list(set(eduIds) | set(cduIds)))[-1] + 1  # Be sure to not use the same id
        for i in intersec:
            mappingCdus[i] = curId
            curId += 1
    if len(list(set(mappingEdus.keys()) & set(mappingCdus.keys()))) != 0:  # Check
        print(mappingEdus, file=sys.stderr)
        print(mappingCdus, file=sys.stderr)
        sys.exit("Overlap between EDU Ids and CDU Ids.")
    # Merge the mappings
    mappingDus = mappingEdus.copy()
    mappingDus.update(mappingCdus)
    # Change the id in the structure
    for e in eduList:
        if e["id"] in mappingDus:
            e["id"] = mappingDus[e["id"]]
        if e["parent"] in mappingDus:
            e["parent"] = mappingDus[e["parent"]]
    for e in groupList:
        if e["id"] in mappingDus:
            e["id"] = mappingDus[e["id"]]
        if e["parent"] in mappingDus:
            e["parent"] = mappingDus[e["parent"]]


def orderSpanList(tree, eduIds):
    # First sort node.eduCovered, and ordered eduSpan for each node
    queue = [tree]
    while queue:
        node = queue.pop(0)
        eduCovered = []
        setEduCovered(node, eduIds, eduCovered)
        node.eduCovered = sortEdu(eduCovered, eduIds)
        node.eduspan = tuple([node.eduCovered[0], node.eduCovered[-1]])
        for m in node.nodelist:
            queue.append(m)
    # Then, check order of nodes in node.nodelist
    queue = [tree]
    while queue:
        node = queue.pop(0)
        orderNodeList(node)
        for m in node.nodelist:
            queue.append(m)


# Utils fct when we do not yet have a tree
def findNode(id_, allNodes):
    for n in allNodes:
        if n._id == id_:
            return n
    return None


def findParentNode(child, allNodes):
    for n in allNodes:
        if child in n.nodelist:
            return n
    return None


def getParentDict(allGroup, parent):
    for g in allGroup:
        if g["id"] == parent:
            return g
    return None


def getParentNode(parent, allNodes):
    for n in allNodes:
        if n._id == parent:
            return n
    return None


# Utils fct for tree structure
def findNodeTree(id_, tree):
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if node._id == id_:
            return node
        for m in node.nodelist:
            queue.append(m)
    return None


def getParentTree(child, tree):
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if child._id in [m._id for m in node.nodelist]:
            return node
        for m in node.nodelist:
            queue.append(m)
    return None


# ----------------------------------------------------------------------------------
# TREE: Clean
# ----------------------------------------------------------------------------------

def cleanTree(tree, eduIds, relationSet, doc):
    '''
    DPLP code deals with: retrieving prop info and moving the relation from the
    children to the parent (see backprop()), and binarizing the tree (binarizeTree()).

    But DPLP doesn t like:
    - lonely child: a node with only one child (EDU or CDU)
    - same-unit/embedded: this relation is treated differently at least in the Basque data,
    we modify the tree in order to make it more similar to the RST DT.
    '''
    allId = list(getIdDu(tree))
    # 4 - DU with only one child
    cleanLonelyEDU(tree, eduIds, relationSet, allId)
    cleanEDU(tree, eduIds, relationSet, allId)
    orderSpanList(tree, eduIds)  # seems necessary...
    cleanLonelyCDU(tree, eduIds, relationSet)
    # 5 - Embedded relations
    cleanEmbedded(tree, eduIds, relationSet, allId, doc)
    cleanLonelyCDU(tree, eduIds, relationSet)
    # Check all nodelists are still ordered
    orderSpanList(tree, eduIds)
    tree.prop = "Root"  # in case it has been changed somewhere..


def cleanEmbedded(tree, eduIds, relationSet, allId, doc):
    '''
    Only deals with problematic cases leading to a pb in the order of the units
    - In general, we have Node_X ( Node_Y( e1-SU, e2-Ri, e4-SU ), e3-Rj )
    with in the text the order e1 e2 e3 e4.
    We put the e3 back as a child of Node_X and then binarize normally ( right attach )
    - We have also more stranger cases

    For now we do not annotate as embedded the embedded relations (not done in several corpora)
    since this info is generally removed in the RST DT for parsing it
    '''
    queue = [tree]
    while queue:
        n = queue.pop(0)
        if "same-unit" in [m.relation for m in n.nodelist]:
            if not areAdjacent([m for m in n.nodelist], eduIds):
                # Cas general: un noeud manquant qui est le voisin du parent
                # -> added in n.nodelist
                id_linked = [m._id for m in n.nodelist]
                missing_nodes_id = [i for i in range(min(id_linked), min(id_linked) + len(id_linked) + 1) if
                                    not i in id_linked]
                parent = getParentTree(n, tree)
                # check if these nodes are neighbors of the current node
                neighbors_id = [m._id for m in parent.nodelist if not m._id == n._id]
                check = True
                for i in missing_nodes_id:
                    if not i in neighbors_id:
                        check = False
                if check:
                    new_nodelist = []
                    for m in parent.nodelist:
                        if m._id in missing_nodes_id:
                            n.nodelist.append(m)
                            parent.nodelist.remove(m)
                    orderSpanList(n, eduIds)
        for m in n.nodelist:
            queue.append(m)


def getNodeCovering(tree, eduIds, id_linked):
    queue = [tree]
    while queue:
        n = queue.pop(0)
        eduCovered = getEduCovered(n, eduIds)
        if id_linked == sorted([m._id for m in eduCovered]):
            return n
        for m in n.nodelist:
            queue.append(m)
    return None


def cleanEDU(tree, eduIds, relationSet, allId):
    """
    EDU with more than one child, happen often in Brazilian and German corpora
    e.g. CSTNews corpus doc D5_C11_GPovo): EDU_1 -> 2-4_elab 5_elab with 2-4 a CDU
    (Need to keep adjacency: the parent EDU can be before its children or in between)
    Means that one argument is shared:
        Edu1 -> Elab(2-4) Elab(5)  ==> Elab( 1, 2-4 ) Elab( 1, 5 )
        -> Create a group that will replace Edu1 with relation span,
        Create a group that replace the CDU and has Edu1 and the original Cdu as children
            newCDU_span -> newCDU_span( Edu1_span, Elab(2-4) ) Edu5_elab
            (Edu1_span = nucleus of the relation)
    """
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if node._id in eduIds and len(node.nodelist) > 1:
            newnode = data.SpanNode(node.prop)  # Keep the nuclearity of the group
            newnode.nodelist = [node]
            newnode.nodelist.extend([m for m in node.nodelist])
            node.nodelist = []  # an EDU has no children
            newnode.relation = "span"
            eduCovered = []
            setEduCovered(newnode, eduIds, eduCovered)
            newnode.eduCovered = sortEdu(eduCovered, eduIds)
            newnode.eduspan = tuple([newnode.eduCovered[0], newnode.eduCovered[-1]])
            newnode._id = node._id + 200
            # Put this node in place of node
            parent = getParentTree(node, tree)
            parent.nodelist.remove(node)
            parent.nodelist.append(newnode)
            # left attach
            left_node = data.SpanNode("Nucleus")  # Has to be a nucleus to give the correct interpretation
            left_node.relation = "span"
            left_node.nodelist = newnode.nodelist[:-1]
            newnode.nodelist = [left_node, newnode.nodelist[-1]]
            eduCovered = []
            setEduCovered(left_node, eduIds, eduCovered)
            left_node.eduCovered = sortEdu(eduCovered, eduIds)
            left_node.eduspan = tuple([left_node.eduCovered[0], left_node.eduCovered[-1]])
            left_node._id = node._id + 201
            orderSpanList(tree, eduIds)
            newnode.nodelist = sorted(newnode.nodelist, key=lambda x: x.eduspan[0])
            left_node.nodelist = sorted(left_node.nodelist, key=lambda x: x.eduspan[0])
            parent.nodelist = sorted(parent.nodelist, key=lambda x: x.eduspan[0])
            queue.append(newnode)
        else:
            for m in node.nodelist:
                queue.append(m)


def cleanLonelyEDU(rootNode, eduIds, relationSet, allId):
    """
    (1) EDU with no neighbor: the parent CDU receives this EDU as lnode and its children
    as rnode
    (2) EDU with a neighbor: create an new artificial node with this EDU as lnode and its
    children as rnode
    """
    queue = [rootNode]
    while queue:
        n = queue.pop(0)
        if n._id in eduIds and len(n.nodelist) == 1:  # Lonely child
            child = n.nodelist[0]
            parent = getParentTree(n, rootNode)
            if parent == None:
                sys.exit("No parent found, but we are supposed to deal with an EDU ?!")
            # - EDU: 2 cas, soit l EDU a un voisin soit elle n en n a pas
            if len(parent.nodelist) > 1:  # (2) EDU with neighbor
                # New CDU with the same parent as n and n and its children as children
                newnode = data.SpanNode(n.prop)
                newnode.relation = n.relation
                n.relation = "span"
                newnode.nodelist = [n]
                newnode._id = allId[-1] + 1
                allId.append(newnode._id)
                newnode.nodelist += n.nodelist
                eduCovered = []
                setEduCovered(newnode, eduIds, eduCovered)
                newnode.eduCovered = sortEdu(eduCovered, eduIds)
                newnode.eduspan = tuple([newnode.eduCovered[0], newnode.eduCovered[-1]])
                # Ce nouveau noeud est un nucleus
                newnode.prop = "Nucleus"
                newnode.relation = "span"
                index = parent.nodelist.index(n)
                parent.nodelist.remove(n)
                parent.nodelist.insert(index, newnode)
                queue.append(parent)
            else:  # (1) EDU with no neighbor
                if parent._id in eduIds:
                    print("Node:", n._id, "Parent:", parent._id, "Child:", child._id)
                    sys.exit("Edu with one child and no neighbors but whose parent is also en EDU")
                parent.nodelist.append(n.nodelist[0])
                queue.append(parent)  # just to be sure
            n.nodelist = []
            n.eduspan = tuple([n._id, n._id])
        else:
            for m in n.nodelist:
                queue.append(m)


def cleanLonelyCDU(rootNode, eduIds, relationSet):
    """
    1- CDU with 1 CDU child: the CDU child should have the span relation, can be safely removed
    """
    queue = [rootNode]
    while queue:
        n = queue.pop(0)
        if len(n.nodelist) == 1:  # Lonely child
            child = n.nodelist[0]
            if not n._id in eduIds:  # CDU
                if not child._id in eduIds:
                    # We have parent -> n (a cdu) -> another cdu -> its children
                    # We want parent -> n -> its children
                    n.nodelist = child.nodelist
                else:  # the child is an EDU
                    # We have parent -> a cdu -> an edu
                    # We want parent -> an edu
                    parent = getParentTree(n, rootNode)
                    parent.nodelist.remove(n)
                    if len(child.nodelist) == 0:
                        # only if the EDU has no child?
                        if child.relation.lower() == 'span' and n.relation.lower() != 'span':
                            child.relation = n.relation
                            child.prop = n.prop
                        parent.nodelist.append(child)
                    else:
                        parent.nodelist.extend(child.nodelist)
                for m in n.nodelist:
                    queue.append(m)
        else:
            for m in n.nodelist:
                queue.append(m)


def areAdjacent(sameunitNodes, eduIds):
    eduCovered = []
    for n in sameunitNodes:
        getEduCovered(n, eduIds, eduCovered)
    idEduCovered = [m._id for m in eduCovered]
    if sorted(idEduCovered) == list(range(min(idEduCovered), max(idEduCovered) + 1)):
        return True
    return False


def setEduCovered(n, eduIds, eduCovered):
    if n._id in eduIds:
        eduCovered.append(n)
    for m in n.nodelist:
        setEduCovered(m, eduIds, eduCovered)


def getEduCovered(tree, eduIds, eduCovered):
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if node._id in eduIds:
            eduCovered.append(node)
        else:
            for m in node.nodelist:
                getEduCovered(m, eduIds, eduCovered)
    return eduCovered


def _markEmbed(tree):
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if node.relation != None and node.relation.lower() != "span" and node.relation.lower() != "same-unit" and node.relation.lower()[
                                                                                                                  -2:] != "-e":
            node.relation += "-e"
        for m in node.nodelist:
            queue.append(m)


def sortEdu(eduCovered, eduIds):
    # Only return a sorted list of id
    positions = [eduIds.index(i) for i in [n._id for n in eduCovered]]
    sortedIds = [x for (y, x) in sorted(zip(positions, [n._id for n in eduCovered]))]
    return sortedIds


def orderNodeList(node):
    node.nodelist = sorted(node.nodelist, key=lambda x: x.eduspan[0])


def getIdDu(tree):
    idDu = set()
    queue = [tree]
    while queue:
        n = queue.pop(0)
        idDu.add(n._id)
        if n.nodelist:
            for m in n.nodelist:
                queue.append(m)
        else:
            if n.lnode and n.rnode:
                queue.append(n.lnode)
                queue.append(n.rnode)
    return idDu


# ----------------------------------------------------------------------------------
# TREE: BINARIZE
# ----------------------------------------------------------------------------------
# TODO merge all the binarization fcts, should be the same
def binarizeTreeGeneral(tree, doc, nucRelations=None):
    """
    Convert a general RST tree to a binary RST tree
    < DPLP but modified: no more systematic right branching, depend of the relations
    of the (children) nodes, keeping a right branching by default (ie if all nodes are NN)

    Simple rules:
    - if the last node is a nucleus or holds the span relation: RA
    - if the last node is a satellite: left attach (pb: makes a diff with the RST DT for the S span S cases)

    But to be coherent with the always Right Attach in the Eng-RST DT, we use:
    - if the last node is a nucleus or holds the span relation: RA
    - if we have a sequence S N+ S, we don't care, both RA and LA are ok, so we choose the RA to be consistent with RSTDT
    - else (ie if the last node is a satellite), LA

    :type tree: instance of SpanNode
    :param tree: a general RST tree
    """
    cases = set()
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if len(node.nodelist) == 2:
            node.lnode = node.nodelist[0]
            node.rnode = node.nodelist[1]
            # Parent node
            node.lnode.pnode = node
            node.rnode.pnode = node
        elif len(node.nodelist) > 2:
            childrenRelations = [m.relation for m in node.nodelist]
            childrenNuclearity = [m.prop for m in node.nodelist]
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
                newnode = leftAttach(node)  # last node = satellite except for the specific pattern
        if node.lnode != None and node.rnode != None:
            queue.append(node.lnode)
            queue.append(node.rnode)


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
    newnode.nodelist += node.nodelist
    # ADDED
    newnode.eduspan = tuple([newnode.nodelist[0].eduspan[0], newnode.nodelist[-1].eduspan[1]])
    # Left-branching
    node.lnode = newnode
    # ADDED
    newnode._id = node._id + 100
    # Parent node
    node.lnode.pnode = node
    node.rnode.pnode = node
    return newnode


def rightAttach(node):
    node.lnode = node.nodelist.pop(0)
    newnode = data.SpanNode('Nucleus')
    newnode.nodelist += node.nodelist
    # ADDED
    newnode.eduspan = tuple([newnode.nodelist[0].eduspan[0], newnode.nodelist[-1].eduspan[1]])
    # Right-branching
    node.rnode = newnode
    # ADDED
    newnode._id = node._id + 100
    # Parent node
    node.lnode.pnode = node
    node.rnode.pnode = node
    return newnode


# ----------------------------------------------------------------------------------
# WRITE
# ----------------------------------------------------------------------------------

def writeEdus(doc, ext, pathout):
    """
    Write files similar to the .edus files in the RST DT for the other RST Treebanks.

    doc: Document instance, contains info about the the tokens in each EDU
    forigin: the discourse file, keep the same basename and path
    ext: the original extension (ie .rs3) to be replaced by the new one (ie .edus)
    """
    edufile = os.path.join(pathout, os.path.basename(doc.path.replace(ext, ".edus")))
    f = open(edufile, 'w', encoding="utf8")
    for edu in doc.edudict:
        line = " ".join([doc.tokendict[gidx] for gidx in doc.edudict[edu]])
        f.write(line.strip() + "\n")
    f.close()


def printTreeRS3(tree):
    '''
    Print a tree when a node is associated with a nodelist
    '''
    queue = [tree]
    while queue:
        n = queue.pop(0)
        try:
            print("-->", n._id, n.relation, n.eduspan, n.prop, [m._id for m in n.nodelist])
        except:
            print("-->", n._id, n.relation.encode('utf8'), n.eduspan, n.prop, [m._id for m in n.nodelist])
        for m in n.nodelist:
            queue.append(m)


"""
Script to convert Rhetorical Structure Theory trees from .rs3 format
to relationships examples pairs.
"""

from rs3_feature_extraction import ParsedToken
import re, sys, codecs, os, tempfile, subprocess, ntpath
import xml
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from argparse import ArgumentParser, FileType
import pandas as pd
import glob
import copy
from file_reading import prepare_text, text_html_map


OUT_PATH = 'data'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class NODE:
    def __init__(self, id, left, right, parent, depth, kind, text, relname, relkind):

        self.id = id
        self.parent = parent
        self.left = left
        self.right = right
        self.depth = depth
        self.kind = kind
        self.text = text
        self.relname = relname
        self.relkind = relkind
        self.sortdepth = depth
        self.leftmost_child = ""
        self.children = []
        self.dep_parent = ""
        self.dep_rel = relname

    def to_row(self):
        return [self.id, self.text, self.dep_parent, self.dep_rel, self.kind]

    def __repr__(self):
        return "\t".join([self.id, self.dep_parent, self.dep_rel, self.kind])


def get_left_right(node_id, nodes, min_left, max_right, rel_hash):
    """
    Calculate leftmost and rightmost EDU covered by a NODE object. For EDUs this is the number of the EDU
    itself. For spans and multinucs, the leftmost and rightmost child dominated by the NODE is found recursively.
    """
    if nodes[node_id].parent != "0" and node_id != "0":
        parent = nodes[nodes[node_id].parent]
        if min_left > nodes[node_id].left or min_left == 0:
            if nodes[node_id].left != 0:
                min_left = nodes[node_id].left
        if max_right < nodes[node_id].right or max_right == 0:
            max_right = nodes[node_id].right
        if nodes[node_id].relname == "span":
            if parent.left > min_left or parent.left == 0:
                parent.left = min_left
            if parent.right < max_right:
                parent.right = max_right
        elif nodes[node_id].relname in rel_hash:
            if parent.kind == "multinuc" and rel_hash[nodes[node_id].relname] == "multinuc":
                if parent.left > min_left or parent.left == 0:
                    parent.left = min_left
                if parent.right < max_right:
                    parent.right = max_right
        get_left_right(parent.id, nodes, min_left, max_right, rel_hash)


def get_depth(orig_node, probe_node, nodes):
    if probe_node.parent != "0":
        parent = nodes[probe_node.parent]
        if parent.kind != "edu" and (
                probe_node.relname == "span" or parent.kind == "multinuc" and probe_node.relkind == "multinuc"):
            orig_node.depth += 1
            orig_node.sortdepth += 1
        elif parent.kind == "edu":
            orig_node.sortdepth += 1
        get_depth(orig_node, parent, nodes)


def read_rst(filename, rel_hash):
    f = codecs.open(filename, "r", "utf-8")
    try:
        xmldoc = minidom.parseString(codecs.encode(f.read(), "utf-8"))
    except ExpatError:
        message = "Invalid .rs3 file"
        return message

    nodes = []
    ordered_id = {}
    schemas = []
    default_rst = ""

    # Get relation names and their types, append type suffix to disambiguate
    # relation names that can be both RST and multinuc
    item_list = xmldoc.getElementsByTagName("rel")
    for rel in item_list:
        relname = re.sub(r"[:;,]", "", rel.attributes["name"].value)
        if rel.hasAttribute("type"):
            rel_hash[relname + "_" + rel.attributes["type"].value[0:1]] = rel.attributes["type"].value
            if rel.attributes["type"].value == "rst" and default_rst == "":
                default_rst = relname + "_" + rel.attributes["type"].value[0:1]
        else:  # This is a schema relation
            schemas.append(relname)

    item_list = xmldoc.getElementsByTagName("segment")
    if len(item_list) < 1:
        return '<div class="warn">No segment elements found in .rs3 file</div>'

    id_counter = 0

    # Get hash to reorder EDUs and spans according to the order of appearance in .rs3 file
    for segment in item_list:
        id_counter += 1
        ordered_id[segment.attributes["id"].value] = id_counter
    item_list = xmldoc.getElementsByTagName("group")
    for group in item_list:
        id_counter += 1
        ordered_id[group.attributes["id"].value] = id_counter
    ordered_id["0"] = 0

    element_types = {}
    node_elements = xmldoc.getElementsByTagName("segment")
    for element in node_elements:
        element_types[element.attributes["id"].value] = "edu"
    node_elements = xmldoc.getElementsByTagName("group")
    for element in node_elements:
        element_types[element.attributes["id"].value] = element.attributes["type"].value

    id_counter = 0
    item_list = xmldoc.getElementsByTagName("segment")
    for segment in item_list:
        id_counter += 1
        if segment.hasAttribute("parent"):
            parent = segment.attributes["parent"].value
        else:
            parent = "0"
        if segment.hasAttribute("relname"):
            relname = segment.attributes["relname"].value
        else:
            relname = default_rst

        # Tolerate schemas, but no real support yet:
        if relname in schemas:
            relname = "span"

            relname = re.sub(r"[:;,]", "", relname)  # remove characters used for undo logging, not allowed in rel names
        # Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
        if parent in element_types:
            if element_types[parent] == "multinuc" and relname + "_m" in rel_hash:
                relname = relname + "_m"
            elif relname != "span":
                relname = relname + "_r"
        else:
            if not relname.endswith("_r") and len(relname) > 0:
                relname = relname + "_r"
        edu_id = segment.attributes["id"].value
        if len(segment.childNodes):
            try:
                contents = segment.childNodes[0].data.strip()
                nodes.append(
                    [str(ordered_id[edu_id]), id_counter, id_counter, str(ordered_id[parent]), 0, "edu", contents,
                     relname])
            except KeyError as e:
                print(bcolors.FAIL + 'PARENT ID ERROR: ' + str(e) + bcolors.ENDC)

    item_list = xmldoc.getElementsByTagName("group")
    for group in item_list:
        if group.attributes.length == 4:
            parent = group.attributes["parent"].value
        else:
            parent = "0"
        if group.attributes.length == 4:
            relname = group.attributes["relname"].value
            # Tolerate schemas by treating as spans
            if relname in schemas:
                relname = "span"

            relname = re.sub(r"[:;,]", "", relname)  # remove characters used for undo logging, not allowed in rel names
            # Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
            if parent in element_types:
                if element_types[parent] == "multinuc" and relname + "_m" in rel_hash:
                    relname = relname + "_m"
                elif relname != "span":
                    relname = relname + "_r"
            else:
                relname = ""
        else:
            relname = ""
        group_id = group.attributes["id"].value
        group_type = group.attributes["type"].value
        contents = ""
        nodes.append([str(ordered_id[group_id]), 0, 0, str(ordered_id[parent]), 0, group_type, contents, relname])

    elements = {}
    for row in nodes:
        elements[row[0]] = NODE(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], "")

    for element in elements:
        if elements[element].kind == "edu":
            get_left_right(element, elements, 0, 0, rel_hash)

    for element in elements:
        node = elements[element]
        get_depth(node, node, elements)

    for nid in elements:
        node = elements[nid]
        if node.parent != "0":
            elements[node.parent].children.append(nid)
            if node.left == elements[node.parent].left:
                elements[node.parent].leftmost_child = nid

    # Ensure left most multinuc children are recognized even if there is an rst dependent further to the left
    for nid in elements:
        node = elements[nid]
        if node.kind == "multinuc" and node.leftmost_child == "":
            min_left = node.right
            leftmost = ""
            for child_id in node.children:
                child = elements[child_id]
                if child.relname.endswith("_m"):  # Using _m suffix to recognize multinuc relations

                    if child.left < min_left:
                        min_left = child.left
                        leftmost = child_id
            node.leftmost_child = leftmost

    return elements


def seek_other_edu_child(nodes, source, exclude, block):
    """
    Recursive function to find some child of a node which is an EDU and does not have the excluded ID
    :param nodes: dictionary of IDs to NODE objects
    :param source: the source node from which to traverse
    :param exclude: node ID to exclude as target child
    :param block: list of IDs for which children should not be traversed (multinuc right children)
    :return: the found child ID or None if none match
    """

    if source == "0":
        return None
    else:
        # Check if this is already an EDU
        if nodes[source].kind == "edu" and source != exclude and source not in block:
            return source
        # Loop through children of this node
        children_to_search = [child for child in nodes[source].children if
                              child not in nodes[exclude].children and child not in block]
        if len(children_to_search) > 0:
            if int(exclude) < int(children_to_search[0]):
                children_to_search.sort(key=lambda x: int(x))
            else:
                children_to_search.sort(key=lambda x: int(x), reverse=True)
        for child_id in children_to_search:
            # Found an EDU child which is not the original caller
            if nodes[child_id].kind == "edu" and child_id != exclude and (
                    nodes[source].kind != "span" or nodes[child_id].relname == "span") and \
                    not (nodes[source].kind == "multinuc" and nodes[source].leftmost_child == exclude) and \
                    (nodes[nodes[child_id].parent].kind not in ["span", "multinuc"]):
                return child_id
            # Found a non-terminal child
            elif child_id != exclude:
                # If it's a span, check below it, following only span relation paths
                if nodes[source].kind == "span":
                    if nodes[child_id].relname == "span":
                        candidate = seek_other_edu_child(nodes, child_id, exclude, block)
                        if candidate is not None:
                            return candidate
                # If it's a multinuc, only consider the left most child as representing it topographically
                elif nodes[source].kind == "multinuc" and child_id == nodes[source].leftmost_child:
                    candidate = seek_other_edu_child(nodes, child_id, exclude, block)
                    if candidate is not None:
                        return candidate
    return None


def find_dep_head(nodes, source, exclude, block):
    parent = nodes[source].parent
    if parent != "0":
        if nodes[parent].kind == "multinuc":
            if int(nodes[nodes[source].parent].left) == int(source):
                return None
            if nodes[source].parent == source:
                return None
            for child in nodes[parent].children:
                # Check whether exclude and child are under the same multinuc and exclude is further to the left
                if nodes[child].left > int(exclude) and nodes[child].left >= nodes[parent].left and int(exclude) >= nodes[parent].left:
                    block.append(child)
    else:
        # Prevent EDU children of root from being dep head - only multinuc children possible at this point
        for child in nodes[source].children:
            if nodes[child].kind == "edu":
                block.append(child)
    candidate = seek_other_edu_child(nodes, nodes[source].parent, exclude, block)

    if candidate is not None:
        return candidate
    else:
        if parent == "0":
            return None
        else:
            if parent not in nodes:
                raise IOError("Node with id " + source + " has parent id " + parent + " which is not listed\n")
            return find_dep_head(nodes, parent, exclude, block)


def get_nonspan_rel(nodes, node):
    if node.parent == "0":  # Reached the root
        return "ROOT"
    elif nodes[node.parent].kind == "multinuc" and nodes[node.parent].leftmost_child == node.id:
        return get_nonspan_rel(nodes, nodes[node.parent])
    elif nodes[node.parent].kind == "multinuc" and nodes[node.parent].leftmost_child != node.id:
        return node.relname
    elif nodes[node.parent].relname != "span":
        grandparent = nodes[node.parent].parent
        if grandparent == "0":
            return "ROOT"
        elif not (nodes[grandparent].kind == "multinuc" and nodes[node.parent].left == nodes[grandparent].left):
            return nodes[node.parent].relname
        else:
            return get_nonspan_rel(nodes, nodes[node.parent])
    else:
        if node.relname.endswith("_r"):
            return node.relname
        else:
            return get_nonspan_rel(nodes, nodes[node.parent])


def get_pairs(df, text):
    pd.options.mode.chained_assignment = None
    
#     text = text.replace('  \n', '#####')
#     text = text.replace(' \n', '#####')
#     text = text + '#####'
#     text = text.replace('#####', '\n')
#     text_html_map = {
#         '\n': r' ',
#         '&gt;': r'>',
#         '&lt;': r'<',
#         '&amp;': r'&',
#         '&quot;': r'"',
#         '&ndash;': r'–',
#         '##### ': r'',
#         '\\\\\\\\': r'\\',
#         '<': ' менее ',
#         '&lt;': ' менее ',
#         r'>': r' более ',
#         r'&gt;': r' более ',
#         r'„': '"',
#         r'&amp;': r'&',
#         r'&quot;': r'"',
#         r'&ndash;': r'–',
#         ' & ': ' and ',  #
#         '&id=': r'_id=',
#         '&': '_',
#         '   ': r' ',
#         '  ': r' ',
#         '  ': r' ',
#         '——': r'-',
#         '—': r'-',
#         #'/': r'',
#         '\^': r'',
#         '^': r'',
#         '±': r'+',
#         'y': r'у',
#         'xc': r'хс',
#         'x': r'х'
#     }

#     for key in text_html_map.keys():
#         text = text.replace(key, text_html_map[key])
#         df['snippet'].replace(key, text_html_map[key], regex=True, inplace=True)
    
          
    df['id'] = df.index
    table = df.merge(df, left_on='dep_parent', right_on='id', how='inner', sort=False, right_index=True) \
        .drop(columns=['dep_parent_y', 'dep_rel_y', 'dep_parent_x', 'kind_x', 'kind_y']) \
        .rename(columns={"dep_rel_x": "category_id"})
    del df

    table = table[table.category_id != 'ROOT']
    table = table[table.category_id != 'span']
    
    for key in text_html_map.keys():
        #text = text.replace(key, text_html_map[key])
        table['snippet_x'].replace(key, text_html_map[key], regex=True, inplace=True)
        table['snippet_y'].replace(key, text_html_map[key], regex=True, inplace=True)

    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        if text.endswith(prefix):
            return text[:-len(prefix)]
        return text

    table.snippet_x = table.apply(lambda row: remove_prefix(row.snippet_x.strip(), row.snippet_y.strip()), axis=1)
    table.snippet_y = table.apply(lambda row: remove_prefix(row.snippet_y.strip(), row.snippet_x.strip()), axis=1)
    table['snippet_x'] = table['snippet_x'].apply(lambda row: row.strip())
    table['snippet_y'] = table['snippet_y'].apply(lambda row: row.strip())
    
#     def find_in_text(plain_text, row):
#         cand = plain_text.find(row.strip())
#         if cand == -1:
#             cand = plain_text.find(row.replace('  ', ' ').strip())
#         return cand

    def find_in_text(plain_text, x, y):
        cand_x = plain_text.find(x)
        cand_y = plain_text.find(y, cand_x + len(x))
        if cand_y - cand_x > len(x) + 3:
            cand_x = plain_text.find(x, cand_x)
            cand_y = plain_text.find(y, cand_x + len(x))
        return (cand_x, cand_y)   
    
    locations = table.apply(lambda row: find_in_text(text, row.snippet_x.strip(), row.snippet_y.strip()), axis=1)
    table['loc_x'] = locations.map(lambda row: row[0])
    table['loc_y'] = locations.map(lambda row: row[1])
    #table['loc_y'] = table.snippet_y.apply(lambda row: find_in_text(text, row.strip()))
          
    def exact_order(row):
        
        if 'order' in row.keys():
            order = row.order
            
            if row.category_id[-2:] == '_m':
                order = 'NN'
        else:
            order = ''

            if row.loc_x < row.loc_y:
                order = 'SN'

            if row.loc_x > row.loc_y:
                order = 'NS'
        
            if row.loc_x == -1 and row.category_id == 'elaboration_r':
                order = 'NS'

            if row.loc_x == -1 and row.category_id == 'preparation_r':
                order = 'NS'
        
        return order
        
    table['order'] = table.apply(lambda row: exact_order(row), axis=1)
    
    ns = table[table.order == 'NS']
    sn = table[table.order == 'SN']

    ns = ns.rename(columns={
        'snippet_x': 'snippet_y_',
        'id_x': 'id_y_',
        'id_y': 'id_x_',
        'snippet_y': 'snippet_x_'
    })
    ns = ns.rename(columns={
        'snippet_x_': 'snippet_x',
        'id_x_': 'id_x',
        'id_y_': 'id_y',
        'snippet_y_': 'snippet_y'
    })

    table = pd.concat([sn, ns], ignore_index=True, sort=False)
    
    table.loc[table.category_id.str[-2:] == '_m', 'order'] = 'NN'
    table.snippet_y = table.apply(lambda row: remove_prefix(row.snippet_y.strip(), row.snippet_x.strip()), axis=1)

    locations = table.apply(lambda row: find_in_text(text, row.snippet_x.strip(), row.snippet_y.strip()), axis=1)
    table['loc_x'] = locations.map(lambda row: row[0])
    table['loc_y'] = locations.map(lambda row: row[1])
    
#     table['loc_x'] = table.snippet_x.apply(lambda row: find_in_text(text, row.strip()))
#     table['loc_y'] = table.snippet_y.apply(lambda row: find_in_text(text, row.strip()))
    #table['order'] = table.apply(lambda row: exact_order(row), axis=1)
    table = table[table.loc_x != -1]
    table = table[table.loc_y != -1]
    
    def cut_middle_injections(row):
        if row.loc_x + len(row.snippet_x) > row.loc_y:
            row.snippet_x = row.snippet_x[:row.loc_y - row.loc_x]
        return row
    
    table = table.apply(lambda row: cut_middle_injections(row), axis=1)
    
    table.drop(columns=['id_x', 'id_y',
                        #'loc_x', 'loc_y', 'loc_x+y',
                        #'new_paragraph_x', 'new_paragraph_y',
                        'dep_parent',
                        ], inplace=True)

    table.drop_duplicates(inplace=True)
    
#     edus_list = []
#     for i in range(len(edus)-1, 0, -1):
#         for j in range(i-1, 0, -1):
#             if len(edus[j][0]) > 4:
#                 edus[i][0] = edus[i][0].replace(edus[j][0], '')
#         edus_list.append(edus[i][0].strip())
#     edus_list.append(edus[0][0])
    
    return table

#######################################################################################################

desc = "Usage example:\n\n" + \
       "python rst2dep.py <INFILES>"
parser = ArgumentParser(description=desc)
parser.add_argument('path', nargs='+', help='Path of a file or a folder of files.')
parser.add_argument("-r", "--root", action="store", dest="root", default="",
                    help="optional: path to corpus root folder containing a directory dep/ and \n" +\
                    "a directory xml/ containing additional corpus formats")

options = parser.parse_args()
full_paths = [os.path.join(os.getcwd(), path) for path in options.path]
files = set()
for path in full_paths:
    if os.path.isfile(path):
        files.add(path)
    else:
        files |= set(glob.glob(path + '/*' + '.rs3'))

for rstfile in files:
    print('>>> read file', rstfile)
    
    out_file = rstfile.split('/')[-1]
    if out_file.endswith("rs3"):
        out_file = out_file.replace("rs3", "json")
    else:
        out_file = out_file + ".pkl"
        
    out_file = os.path.join(OUT_PATH, out_file)


    ### 1. save edus in <filename>.edus ##############################################################
    
    try:
        xmldoc = minidom.parse(rstfile)
    except xml.parsers.expat.ExpatError as e:
        original = open(rstfile, 'r').read()
        
        mapping = {
            r'&amp;': r'&',
            r'&quot;': r'"',
            r'&ndash;': r'–',
            r'&ouml;': r'o',
            r'&hellip;': r'...',
            r'&eacute;': r'e',
            r'&aacute;': r'a',
            r'&rsquo;': r"'",
            r'&lsquo;': r"'",
            r' & ': r' and ',  #
            r'&id=': r'_id=',
        }
        
        mapped = original
        for key, value in mapping.items():
            mapped = mapped.replace(key, value)
        
        with open(rstfile, 'w') as buffer:
            buffer.write(mapped)
            
        try:
            xmldoc = minidom.parse(rstfile)
        except xml.parsers.expat.ExpatError as e: 
            with open(rstfile, 'w') as f:
                f.write(original)
                
            print('Error occured in file:', rstfile)
            print(e)

    edus = xmldoc.getElementsByTagName('segment')
    with open(out_file.replace("json", "edus"), 'w') as f:
        for edu in edus:
            if len(edu.childNodes) > 0:
                f.write(edu.childNodes[0].nodeValue + '\n')

    ### 2. save trees in <filename>.json #############################################################
    
    nodes = read_rst(rstfile, {})
    out_graph = []
    dep_root = options.root    

    # Add tokens to terminal nodes
    if nodes == "Invalid .rs3 file":
        print(nodes)
    else:
        edus = list(nodes[nid] for nid in nodes if nodes[nid].kind == "edu")
        edus.sort(key=lambda x: int(x.id))
        token_reached = 0

        # Get each node with 'span' relation its nearest non-span relname
        for nid in nodes:
            node = nodes[nid]
            if nid == "9":
                pass
            new_rel = node.relname
            if node.parent == "0":
                new_rel = "ROOT"
            node.dep_rel = new_rel

        counter = 0
        joint_trees = []

        for nid in nodes:
            node = nodes[nid]

            if node.parent != '0' and nodes[node.parent].kind == "span" and (
                    int(nodes[node.parent].left) - 1 == int(node.id) or int(nodes[node.parent].right) + 1 == int(node.id)):
                dummy_text = ''
                parent = nodes[node.parent]
                for node_id in range(parent.left, parent.right + 1):
                    if nodes.get(str(node_id)):
                        dummy_text += nodes[str(node_id)].text + " "
                if dummy_text:
                    # print('1.', node.id, dummy_text)
                    parent = copy.copy(parent)
                    parent.text = dummy_text
                    parent.children = []
                    parent.dep_parent = '0'
                    parent.dep_rel = "ROOT"
                    node.dep_parent = parent.id
                    #out_graph.append(parent)
                    out_graph.append(node)

            elif nid != '0' and node.kind in ["multinuc", "span"]:
                if node.parent == '0' and node.kind == 'multinuc':
                    dummy_text = ''
                    for node_id in range(node.left, node.right + 1):
                        if nodes.get(str(node_id)):
                            dummy_text += nodes[str(node_id)].text + " "
                    node.text = dummy_text
                    node.children = []
                    out_graph.append(node)

                elif node.parent == '0' and node.kind == 'span':
                    dummy_text = ''
                    for node_id in range(node.left, node.right + 1):
                        if nodes.get(str(node_id)):
                            dummy_text += nodes[str(node_id)].text + " "
                    #print('2.2', node.id, dummy_text)
                    node.text = dummy_text
                    node.children = []
                    out_graph.append(node)

                elif node.parent != '0':
                    dummy_text = ''
                    for node_id in range(node.left, node.right + 1):
                        if nodes.get(str(node_id)):
                            dummy_text += nodes[str(node_id)].text + " "

                    if dummy_text:
                        if node.kind == "multinuc" and (node.left, node.right) != (
                        nodes[node.parent].left, nodes[node.parent].right):
                            node.dep_parent = node.parent
                            node.text = dummy_text
                            #print('3.', node.id, node.text)
                            node.children = []
                            out_graph.append(node)
                        elif nodes[node.parent].kind == 'multinuc':
                            node.dep_parent = node.parent
                            node.text = dummy_text
                            #print('4.', node.id, node.text)

                            if node.dep_rel in ['joint_m', 'same-unit_m']:
                                # if node.dep_rel == 'joint_m':
                                # print('4.', node.id, node.text, node.dep_rel)
                                children = nodes[node.parent].children
                                if len(children) > 2 and children[0] != node.id and not children in joint_trees:
                                    # print('::', children)
                                    joint_trees.append(nodes[node.parent].children)

                            #node.children = []
                            out_graph.append(node)
                        else:
                            node.text = dummy_text
                            # print('5.', node.id, node.text)  #106
                            node.dep_parent = node.parent
                            node.children = []
                            out_graph.append(node)

            elif node.kind == "edu":
                dep_parent = find_dep_head(nodes, nid, nid, [])
                if dep_parent is None:
                    # print('a.')
                    # This is the root
                    node.dep_parent = "0"
                    node.dep_rel = "ROOT"
                elif node.parent != '0' and nodes[node.parent].kind == 'span':
                    # print('b.')
                    node.dep_parent = "0"
                    node.dep_rel = "ROOT"
                else:
                    #print('c.', node.parent, dep_parent, nodes[dep_parent].text, node.dep_rel, node.text)
                    if node.dep_rel in ['joint_m', 'same-unit_m']:
                        # if node.dep_rel == 'joint_m':
                        # print('4.', node.id, node.text, node.dep_rel)
                        children = nodes[node.parent].children
                        if len(children) > 2 and children[0] != node.id and not children in joint_trees:
                            # print('::', children)
                            joint_trees.append(nodes[node.parent].children)
                    node.dep_parent = node.parent
                out_graph.append(node)

            else:
                pass
                #print('>>>', nid, node.kind)

        out_graph.sort(key=lambda x: int(x.id))

        def get_node(id):
            return [i for i, x in enumerate(out_graph) if x.id == id][0]

        for joint_tree in joint_trees:
            nid = [get_node(id) for id in joint_tree]
            #print(nid)

            # news_44/108-115
            for i in range(len(nid)):
                if not out_graph[nid[i]].children:
                    out_graph[nid[i]].children = [out_graph[nid[i]].id]

            for i in range(len(nid)):
                for k in range(0, len(nid)-i-1):
                    if int(out_graph[nid[k]].children[0]) > int(out_graph[nid[k + 1]].children[0]):
                        nid[k], nid[k + 1] = nid[k + 1], nid[k]

            for i in range(1, len(nid)-1):
                out_graph[nid[i]].dep_parent = out_graph[nid[i-1]].id
                out_graph[nid[i]].dep_rel = out_graph[nid[-2]].dep_rel
                out_graph[nid[i]].text = ' '.join([out_graph[nid[k]].text.strip() for k in range(i - 1, i + 1)])
                out_graph[nid[i]].text.replace(out_graph[nid[i+1]].text, '')

            out_graph[nid[-1]].dep_parent = out_graph[nid[-2]].id
            out_graph[nid[-1]].dep_rel = out_graph[nid[-2]].dep_rel

        data = []

        for node in out_graph:
            data.append(node.to_row())

        filename = '.'.join(out_file.split('/')[-1].split('.')[:-1])
        textfile = '/'.join(rstfile.split('/')[:-1]).replace('rs3', 'txt') + '/' + filename + '.txt'

        with open(textfile, 'r') as f:
            text = prepare_text(f.read())
        
        df = pd.DataFrame(data, columns=['id', 'snippet', 'dep_parent', 'dep_rel', 'kind']).set_index('id')
        new_df = get_pairs(df, text)
        new_df['filename'] = filename
        new_df.to_json(out_file)

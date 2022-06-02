import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse.csgraph import connected_components


class RS3ForestSplitter:
    """ Splits a single *.rs3 file with multiple trees
        to multiple *_part_x.rs3 files with single trees """

    def __call__(self, filename: str, output_dir: str):
        output_filename = filename.split('/')[-1]
        output_filename = output_filename.replace(".rst", "").replace(".rs3", "")

        # Save file header and make adjacency matrix
        pairs = []  # [[id1, parent1], [id2, parent2], ...]
        context = ET.iterparse(filename, events=('end',))
        for event, elem in context:
            if elem.tag == 'header':
                header = ET.tostring(elem).decode('utf-8')

                #### For some reason, relation Restatement is duplicating in the markup
                header = header.replace('<rel name="restatement" type="rst"/>', '')
                header = header.replace('<rel name="restatement" type="rst" />', '')

            elif elem.tag == 'body':
                for child in elem:
                    if child.get('parent'):
                        pairs.append(list(map(int, [child.get('id'), child.get('parent')])))
                    else:
                        pairs.append(list(map(int, [child.get('id'), child.get('id')])))

        max_id = np.array(pairs).max()
        adj_matrix = np.zeros((max_id, max_id))
        for pair in pairs:
            adj_matrix[pair[0] - 1, pair[1] - 1] = 1
            adj_matrix[pair[1] - 1, pair[0] - 1] = 1
            adj_matrix[pair[0] - 1, pair[0] - 1] = 1
            adj_matrix[pair[1] - 1, pair[1] - 1] = 1

        # Find separated trees
        n_components, labels = connected_components(adj_matrix)
        trees = dict()
        for _id, tree_number in enumerate(labels):
            if not trees.get(tree_number):
                trees[tree_number] = [str(_id + 1)]
            else:
                trees[tree_number].append(str(_id + 1))

        trees_body = dict()
        context = ET.iterparse(filename, events=('end',))
        for event, elem in context:
            if elem.tag == 'body':
                for child in elem:
                    for tree_number, tree_ids in trees.items():
                        if child.get('id') in tree_ids:
                            if not trees_body.get(tree_number):
                                trees_body[tree_number] = [child]
                            else:
                                trees_body[tree_number].append(child)

        # Write the results
        for i, tree_number in enumerate(trees_body.keys()):
            try:
                with open(os.path.join(output_dir, f'{output_filename}_part_{i}.rs3'), 'w') as f:
                    f.write('<rst>\n')
                    f.write(header)
                    f.write('<body>\n')
                    for element in trees_body.get(tree_number):
                        _id = element.get('id')
                        _type = element.get('type')
                        _par = element.get('parent')
                        parent = f'parent="{_par}"' if _par else ''
                        _relname = element.get('relname')
                        f.write(f'\t\t<{element.tag} id="{_id}" type="{_type}" {parent} relname="{_relname}"')
                        if element.tag == 'segment':
                            f.write(f'>{self.debracket_text(element.text)}</segment>\n')
                        elif element.tag == 'group':
                            f.write('/>\n')
                    f.write('\t</body>\n')
                    f.write('</rst>')
            except Exception as e:
                print(f"Skip tree {tree_number} in file {filename}:", e)

    @staticmethod
    def debracket_text(text):
        return text.replace(')', '-RB-').replace('(', '-LB-')

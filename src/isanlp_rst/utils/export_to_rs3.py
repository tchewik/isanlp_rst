class Segment:
    def __init__(self, _id, parent, relname, text):
        self.id = _id
        self.parent = parent
        self.relname = relname
        self.text = text

    def __str__(self):
        if self.parent:
            return f'<segment id="{self.id}" parent="{self.parent}" relname="{self.relname}">{self.text}</segment>'
        
        return f'<segment id="{self.id}" relname="{self.relname}">{self.text}</segment>'

class GroupCreator:
    def __init__(self, _id):
        self._id = _id

    def __call__(self, type, parent, relname):
        self._id += 1
        return Group(self._id, type, parent, relname)


class Group:
    def __init__(self, _id, type, parent, relname):
        self.id = _id
        self.type = type
        self.parent = parent
        self.relname = relname

    def __str__(self):
        return f'<group id="{self.id}" type="{self.type}" parent="{self.parent}" relname="{self.relname}"/>'


class Root(Group):
    def __init__(self, _id):
        Group.__init__(self, _id, type="span", parent=-1, relname="span")

    def __str__(self):
        return f'<group id="{self.id}" type="{self.type}"/>'


class Exporter:
    def __init__(self, encoding='cp1251'):
        self._encoding = encoding
    
    def __call__(self, tree, filename):

        with open(filename, 'w', encoding=self._encoding) as fo:
            fo.write('<rst>\n')
            fo.write(self.make_header(tree))
            fo.write(self.make_body(tree))
            fo.write('</rst>')

    def compile_relation_set(self, tree):
        result = ['_'.join([tree.relation, tree.nuclearity])]
        if not tree.left:
            return result
        if tree.left.left:
            result += self.compile_relation_set(tree.left)
        if tree.right.left:
            result += self.compile_relation_set(tree.right)
            
        return result

    def make_header(self, tree):
        relations = list(set(self.compile_relation_set(tree)))
        relations = [value if value != "elementary__" else "antithesis_NN" for value in relations]

        result = '\t<header>\n'
        result += '\t\t<relations>\n'
        for rel in relations:
            _relname, _type = rel.split('_')
            _type = 'multinuc' if _type == 'NN' else 'rst'
            result += f'\t\t\t<rel name="{_relname}" type="{_type}" />\n'
        result += '\t\t</relations>\n'
        result += '\t</header>\n'

        return result

    def get_groups_and_edus(self, tree):
        groups = []
        edus = []

        if not tree.left:
            edus.append(Segment(tree.id, parent=None, relname='antithesis', text=tree.text))
            return groups, edus

        if not tree.left.left:
            if tree.nuclearity == "SN":
                edus.append(Segment(tree.left.id, tree.right.id, tree.relation, tree.left.text))
            elif tree.nuclearity == "NS":
                edus.append(Segment(tree.left.id, tree.id, 'span', tree.left.text))
            else:
                edus.append(Segment(tree.left.id, tree.id, tree.relation, tree.left.text))
        else:
            if tree.nuclearity == "SN":
                groups.append(Group(tree.left.id, 'span', tree.right.id, tree.relation))
            elif tree.nuclearity == "NS":
                groups.append(Group(tree.left.id, 'span', tree.id, 'span'))
            else:
                groups.append(Group(tree.left.id, 'multinuc', tree.id, tree.relation))

            _groups, _edus = self.get_groups_and_edus(tree.left)
            groups += _groups
            edus += _edus

        if not tree.right.left:
            if tree.nuclearity == "SN":
                edus.append(Segment(tree.right.id, tree.id, 'span', tree.right.text))
            elif tree.nuclearity == "NS":
                edus.append(Segment(tree.right.id, tree.left.id, tree.relation, tree.right.text))
            else:
                edus.append(Segment(tree.right.id, tree.id, tree.relation, tree.right.text))

        else:
            if tree.nuclearity == "SN":
                groups.append(Group(tree.right.id, 'span', tree.id, 'span'))
            elif tree.nuclearity == "NS":
                groups.append(Group(tree.right.id, 'span', tree.left.id, tree.relation))
            else:
                groups.append(Group(tree.right.id, 'multinuc', tree.id, tree.relation))

            _groups, _edus = self.get_groups_and_edus(tree.right)
            groups += _groups
            edus += _edus

        return groups, edus

    def make_body(self, tree):
        groups, edus = self.get_groups_and_edus(tree)
        if len(edus) > 1:
            groups.append(Root(tree.id))

        result = '\t<body>\n'
        for edu in edus + groups:
            result += '\t\t' + str(edu) + '\n'
        result += '\t</body>\n'

        return result

class ForestExporter:
    def __init__(self, encoding='cp1251'):
        self._encoding = encoding
        self._tree_exporter = Exporter(self._encoding)
    
    def __call__(self, trees, filename):
        
        with open(filename, 'w', encoding=self._encoding) as fo:
            fo.write('<rst>\n')
            fo.write(self.make_header(trees))
            fo.write(self.make_body(trees))
            fo.write('</rst>')
        
    def compile_relation_set(self, trees):
        result = []
        
        for tree in trees:
            result += list(set(self._tree_exporter.compile_relation_set(tree)))
               
        result = [value if value != "elementary__" else "antithesis_NN" for value in result]
        return result
    
    def make_header(self, trees):
        relations = list(set(self.compile_relation_set(trees)))

        result = '\t<header>\n'
        result += '\t\t<relations>\n'
        for rel in relations:
            _relname, _type = rel.split('_')
            _type = 'multinuc' if _type == 'NN' else 'rst'
            result += f'\t\t\t<rel name="{_relname}" type="{_type}" />\n'
        result += '\t\t</relations>\n'
        result += '\t</header>\n'

        return result
    
    def make_body(self, trees):
        groups, edus = [], []
        
        for tree in trees:
            _groups, _edus = self._tree_exporter.get_groups_and_edus(tree)
            if len(_edus) > 1:
                _groups.append(Root(tree.id))
            groups += _groups
            edus += _edus

        result = '\t<body>\n'
        for edu in edus + groups:
            result += '\t\t' + str(edu) + '\n'
        result += '\t</body>\n'

        return result.replace('\u2015', '-')

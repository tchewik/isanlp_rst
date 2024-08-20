import re


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.edu_id = None
        self.relation = None
        self.span = None
        self.parent = None
        self.is_paragragh_span = False
        self.is_sentence_span = False


class BinaryTree:
    def __init__(self, dmrg_path, text_path, edus_path):
        self.string = ''
        self.sentence_span = {}
        self.paragraph_span = {}
        self.convert_file_to_string(dmrg_path)
        self.find_sentence_span(text_path, edus_path)
        self.root = self.build_tree(self.string)

    def convert_file_to_string(self, dmrg_path):
        '''
        :return: convert a dmrg file into a string.
        '''
        lines = []
        with open(dmrg_path, 'r') as file:
            for line in file:
                lines.append(line.strip())
        self.string = ' '.join(lines)
        self.string = self.string.replace(') (', ')(')  # remove space between bracket.

    def find_span_index(self, string):
        '''
        :return: find a index which separate the left and right child.
        '''
        flag = 0
        for i, c in enumerate(string):
            if c == '(':
                flag += 1
            if c == ')':
                flag += -1
            if flag == 0:
                return i

    def build_tree(self, string):
        '''
        :return: a binary tree.
        '''
        node = Node()

        space_index = string.find(' ')
        value = string[1:space_index]
        if value == 'EDU':
            idx = int(re.findall(r"\d+?\d*", string)[0])
            node.edu_id = idx
            node.span = [idx, idx]
            if str(node.span) in self.paragraph_span:
                node.is_paragragh_span = True
            if str(node.span) in self.sentence_span:
                node.is_sentence_span = True
            return node

        node.relation = value
        sub_string = string[space_index + 1:-1]
        span_index = self.find_span_index(sub_string)
        node.left = self.build_tree(sub_string[:span_index + 1])
        node.left.parent = node
        node.right = self.build_tree(sub_string[span_index + 1:])
        node.right.parent = node
        node.span = [node.left.span[0], node.right.span[1]]
        if str(node.span) in self.paragraph_span:
            node.is_paragragh_span = True
        if str(node.span) in self.sentence_span:
            node.is_sentence_span = True

        return node

    def find_sentence_span(self, text_file_path, edus_file_path):
        '''
        :param text_file_path: text contains sentence and paragraph information.
        :param edus_file_path: lines of EDUS.
        :return: the span of each EDUS.
        '''
        sentence_span = {}
        paragraph_span = {}
        text_file = open(text_file_path, 'r')
        edus_file = open(edus_file_path, 'r')
        text_lines = []
        edus_lines = []

        for line in text_file:
            text_lines.append(line.strip())
        for line in edus_file:
            edus_lines.append(line.strip())
        text_line = 0
        edus_line = 0
        last_sent_edu = 1  # the edu start from 1.
        last_para_edu = 1

        while text_line < len(text_lines):
            line_punctuation = ''.join(re.findall(r'\W', text_lines[text_line].replace(' ', '')))
            # end of paragraph.
            if text_lines[text_line] == '':
                paragraph_span[str([last_para_edu, edus_line])] = 1
                last_para_edu = edus_line + 1
                text_line += 1
                continue
            sentence_punctuation = ''
            while edus_line < len(edus_lines):
                if len(sentence_punctuation) > len(line_punctuation):
                    # print('EDU include more than one line.')
                    text_line += 1
                    line_punctuation += ''.join(re.findall(r'\W', text_lines[text_line].replace(' ', '')))
                    continue
                sentence_punctuation += ''.join(re.findall(r'\W', edus_lines[edus_line].replace(' ', '')))

                # end of sentence.
                if sentence_punctuation == line_punctuation:
                    span = [last_sent_edu, edus_line + 1]
                    last_sent_edu = edus_line + 2
                    text_line += 1
                    edus_line += 1
                    sentence_span[str(span)] = 1
                    break
                edus_line += 1
        paragraph_span[str([last_para_edu, edus_line])] = 1  # add the last paragraph
        self.paragraph_span = paragraph_span
        self.sentence_span = sentence_span
        text_file.close()
        edus_file.close()

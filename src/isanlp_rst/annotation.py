class DiscourseUnit:
    def __init__(self, id, left=None, right=None, text='', start=None, end=None,
                 orig_text=None, relation='', nuclearity='', proba=1.):
        """
        :param int id:
        :param DiscourseUnit left:
        :param DiscourseUnit right:
        :param str text: (optional)
        :param int start: start position in original text
        :param int end: end position in original text
        :param string relation: {the relation between left and right components | 'elementary' | 'root'}
        :param string nuclearity: {'NS' | 'SN' | 'NN'}
        :param float proba: predicted probability of the relation occurrence
        """
        self.id = id
        self.left = left
        self.right = right
        self.relation = relation
        self.nuclearity = nuclearity
        self.proba = str(proba)
        self.start = start
        self.end = end

        if self.left:
            self.start = left.start
            self.end = right.end + 1

        if orig_text:
            self.text = orig_text[self.start:self.end].strip()
        else:
            self.text = text.strip()

    def __str__(self):
        return f"id: {self.id}\ntext: {self.text}\nrelation: {self.relation}\nleft: {self.left.text if self.left else None}\nright: {self.right.text if self.right else None}\nstart: {self.start}\nend: {self.end}"

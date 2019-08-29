class DiscourseUnit:
    def __init__(self, id, left=None, right=None, text='', start=None, end=None, relation=None, nuclearity=None, proba=1.):
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

        if not text and left:
            self.text = left.text + right.text
            self.start = left.start
            self.end = right.end
        else:
            self.text = text
            self.start = start
            self.end = end

    def __str__(self):
        return f"id: {self.id}\ntext: {self.text}\nrelation: {self.relation}\nleft: {self.left.text if self.left else None}\n\right: {self.right.text if self.right else None}\n"

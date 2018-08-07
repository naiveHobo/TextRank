class SyntacticUnit(object):
    """
    Wrapper class for words, processed tokens, corresponding part-of-speech tags and scores
    """

    def __init__(self, text, token=None, tag=None):
        self.text = text
        self.token = token
        self.tag = tag[:2] if tag else None  # just first two letters of tag
        self.index = -1
        self.score = -1

    def __str__(self):
        return self.text + "\t" + self.token + "\n"

    def __repr__(self):
        return str(self)

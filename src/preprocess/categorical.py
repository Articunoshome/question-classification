"""
Contains all the classes and functions required for preprocessing categorical data
"""


class LabelEncoder():
    def __init__(self):
        self.itol = None
        self.ltoi = None

    def build_labels(self, labels):
        self.itol = list(set(labels))
        self.ltoi = dict(zip(self.itol, range(len(self.itol))))
        return torch.tensor(list(map(self.ltoi.get, labels)))

    def convert_labels_to_encodings(self, labels):
        """
        Given list of sentences function returns list of encoded tokens
        """
        encoded = list(map(self.ltoi.get, labels))
        return encoded

    def convert_encodings_to_labels(self, encodings):
        return list(map(lambda x: self.itol[x], encodings))

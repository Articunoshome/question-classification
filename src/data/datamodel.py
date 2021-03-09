from torch.utils.data import Dataset


class QuestionDataset(Dataset):
    def __init__(self, questions, offsets, labels):
        self.questions = questions
        self.offsets = offsets
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.questions[index], self.offsets[index], self.labels[index]

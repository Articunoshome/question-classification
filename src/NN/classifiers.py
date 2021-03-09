"""
The module contains all the classes and functions of creating training and testing models
"""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
      Question classification model class
    """

    def __init__(self, vocab_size, embed_dim,
                 num_class, hidden_dim, use_bilstm, use_pre_emb, pre_emb, freeze):
        super().__init__()
        self.lr = 0.01
        self.gamma = 0.9
        self.device = 'cpu'
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=self.gamma)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.use_pre_emb = use_pre_emb
        if self.use_pre_emb:
            pre_emb = torch.tensor(pre_emb)
            # In case of pretrained model the embedding output will be of pretrained model's shape
            self.embed_dim = pre_emb.shape[-1]
        else:
            self.embed_dim = embed_dim
        self.num_class = num_class
        self.use_bilstm = use_bilstm

        # sentence representation
        if self.use_bilstm:
            # using BiLSTM
            # embedding layer
            if self.use_pre_emb:
                # load pretrained embedding to the layer
                self.embedding = nn.Embedding.from_pretrained(
                    pre_emb, freeze=freeze)
            else:
                # Initialise random weights in EMbedding layer
                self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

            # lstm layer
            self.lstm = nn.LSTM(self.embed_dim,
                                self.hidden_dim,
                                bidirectional=True)
        else:
            # Using Bag Of Words
            if self.use_pre_emb:
                # load pretrained embedding to the layer
                self.embedding = nn.EmbeddingBag.from_pretrained(
                    pre_emb, mode='mean', sparse=True, freeze=freeze)
            else:
                # Initialise random weights in EMbedding layer
                self.embedding = nn.EmbeddingBag(
                    self.vocab_size, self.embed_dim, mode='mean', sparse=True)
        # The linear dense layer to convet to a prob distr of the shape (sentence_vector x num_class)
        self.embedding.padding_idx = 0
        self.fc = nn.Linear(
            self.hidden_dim*2 if self.use_bilstm else self.embed_dim, self.num_class)

    def set_learning_params(self, lr=0.1, gamma=0.9, device='cpu'):
        """
        Set the learning rates and the learning device od the model.
        """
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=self.gamma)

    def forward(self, text: "The input text: list of encoded tokens",
                offsets: "Offest for the sentence") -> "Returns the output of the linear layer":
        """
            Single forward pass for the network
        """
        if self.use_bilstm:
            embedded = self.embedding(text)
            # Packed sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, offsets.cpu(), batch_first=True, enforce_sorted=False)

            _output, (hidden, _c_n) = self.lstm(packed_embedded)
            # concat the final forward and backward hidden state
            sentence_vec = torch.cat(
                (hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            sentence_vec = self.embedding(text)
            # return the output of the final activation. (Softmax)
        return self.fc(sentence_vec)

    def train_func(self, sub_train_: "The batch data iterator"):
        """
            The method to train the model
        """
        train_loss = 0
        train_acc = 0
        for _batch_no, example in enumerate(sub_train_):
            self.optimizer.zero_grad()
            text, offsets, cls = example[0].to(
                self.device), example[1].to(self.device), example[2].to(self.device)
            output = self(text, offsets)
            loss = self.criterion(output, cls)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_acc += (output.argmax(1) == cls).sum().item()/len(output)

            # Adjust the learning rate
        self.scheduler.step()

        return train_loss / len(sub_train_), train_acc / len(sub_train_)

    def test(self, data_):
        # Calculate the performance of the model
        loss = 0
        acc = 0
        for example in data_:
            text, offsets, cls = example[0].to(
                self.device), example[1].to(self.device), example[2].to(self.device)
            with torch.no_grad():
                output = self(text, offsets)

                loss = self.criterion(output, cls)
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item() / len(output)

        return loss / len(data_), acc / len(data_)

    def predict(self, sentences: "List of sentence tokens") -> "Returns the predicted labels":
        """
        Provided the sentence tokens the function will predict and return the
        label tokens for a list of sentences
        """
        predictions = []
        for sentence in sentences:
            length = [len(sentence)]  # compute no. of words
            tensor = torch.LongTensor(sentence).to(
                self.device)  # convert to tensor
            # reshape in form of batch,no. of words
            tensor = tensor.unsqueeze(1).T
            length_tensor = torch.LongTensor(length)  # convert to tensor
            prediction = self(tensor, length_tensor)  # prediction
            predictions.append(prediction.argmax())
        return np.array(predictions)


class Ensemble():
    """
    Ensemble model for comibining the results and bagging it
    """

    def __init__(self, model_list):
        self.models = model_list

    def predict(self, text, offset):
        # clone to make sure text and offsets are not changed by inplace methods
        y_pred_counter = Counter([model(text.clone(), offset.clone()).argmax().item()
                                  for model in self.models])
        y_pred = y_pred_counter.most_common(1)[0][0]
        return y_pred

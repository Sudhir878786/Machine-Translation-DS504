import sys
sys.path.append("../Language-Detector/src")

import torch

from dataloader import vocabs
from dataloader import datasets
from dataloader.utils import get_tokens_from_line

from model.classifier import Seq2VecNN


class LanguageDetector:
    def __init__(self, vocabs_filepath, saved_model_filepath):
        vocab = vocabs.load_vocabs_from_file(vocabs_filepath)
        self.word2index = vocab.get_word2id()

        self.model = Seq2VecNN(len(self.word2index), 2, num_neurons_per_layer=[100, 25])
        self.model.load_state_dict(torch.load(saved_model_filepath))
        self.model.eval()

    def predict(self, text):

        tokens, _ = get_tokens_from_line(text)
        F = torch.zeros(len(self.word2index))
        for word in tokens:
            if word in self.word2index:
                index = self.word2index[word]
                F[index] = 1

        predicted_lang = None
        with torch.no_grad():
            logits = self.model(F)
            predictions = torch.argmax(torch.round(torch.sigmoid(logits))).item()

            if predictions == 0:
                predicted_lang = "en"
            else:
                predicted_lang = "fr"

        return predicted_lang
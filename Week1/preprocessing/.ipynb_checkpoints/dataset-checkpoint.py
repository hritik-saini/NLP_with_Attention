import math
import random
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from Helper_Functions import *


class TokenizedDataStream:
    def __init__(self, examples, vocab_size=10000, max_length=512):
        self.examples = examples
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.EOS = 1

        self.pt_lines, self.en_lines = self.extract_lines(examples)
        self.pt_tokenizer, self.en_tokenizer = self.create_tokenizers(self.pt_lines, self.en_lines)
        self.pt_sequences, self.en_sequences = self.convert_to_sequences(self.pt_lines, self.en_lines)

    def extract_lines(self, examples):
        pt_lines = [pt.numpy().decode('utf-8') for pt, _ in examples]
        en_lines = [en.numpy().decode('utf-8') for _, en in examples]
        return pt_lines, en_lines

    def create_tokenizers(self, pt_lines, en_lines):
        pt_tokenizer = self.create_tokenizer(pt_lines, self.vocab_size)
        en_tokenizer = self.create_tokenizer(en_lines, self.vocab_size)
        return pt_tokenizer, en_tokenizer

    def create_tokenizer(self, lines, vocab_size):
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def convert_to_sequences(self, pt_lines, en_lines):
        pt_sequences = self.pt_tokenizer.texts_to_sequences(pt_lines)
        en_sequences = self.en_tokenizer.texts_to_sequences(en_lines)
        return pt_sequences, en_sequences

    def append_eos(self, inputs, targets):
        for input, target in zip(inputs, targets):
            # Append EOS to each sentence
            input_seq = list(input) + [self.EOS]
            target_seq = list(target) + [self.EOS]
            yield np.array(input_seq), np.array(target_seq)

    def get_tokenized_stream(self):
        pt_sequences, en_sequences = self.pt_sequences, self.en_sequences
        return self.append_eos(en_sequences, pt_sequences)

    def detokenize(self, integers, type):
        integers = list(np.squeeze(integers))

        EOS = 1

        if EOS in integers:
            integers = integers[:integers.index(EOS)]

        if type == "Input":
            # Convert integer sequences back to text using tokenizers
            return self.en_tokenizer.sequences_to_texts([integers])[0]

        if type == "Target":
            # Convert integer sequences back to text using tokenizers
            return self.pt_tokenizer.sequences_to_texts([integers])[0]

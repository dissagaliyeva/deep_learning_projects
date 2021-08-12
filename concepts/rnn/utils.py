import io
import os
import unicodedata
import string
import glob

import torch
import random

# get the alphabet
ALL_LETTERS = string.ascii_letters + ".,;'"
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(s):
    """
    This function turns Unicode to plain ASCII. It uses NFD normalization (canonical decomposition).
        :param s: List to convert
    :return: ASCII representation
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )


def load_data():
    """
    This function builds the category_lines dictionary which holds names per language.
    :return:
    """
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('data/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)

        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories


def letter_to_index(letter):
    """
    Find letter index from all_letters, e.g. "a" = 0.
        :param letter: Letter to find
    :return: Letter index
    """
    return ALL_LETTERS.find(letter)


def letter_to_tensor(letter):
    """
    Turn a letter into a <1 x n_letters> Tensor. Pytorch assumes everything is in batches.
        :param letter: Letter to turn to Tensor
    :return: Tensor letter
    """
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
        :param line:
    :return:
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for idx, letter in enumerate(line):
        tensor[idx][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(category_lines, all_categories):
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


# if __name__ == '__main__':
#     print(ALL_LETTERS)
#     print(unicode_to_ascii('Ślusàrski'))
#     category_lines, all_categories = load_data()
#     print(category_lines['Italian'][:5])
#
#     print(letter_to_tensor('J'))  # [1, 57]
#     print(line_to_tensor('Jones').size())  # [5, 1, 57]





















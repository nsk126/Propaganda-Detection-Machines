import argparse
import numpy as np

from src.load_dataset import Load_data as loader
# Function to perform one-hot encoding

def one_hot_encoding(char):
    # Create an array of zeros of shape (95,)
    one_hot = np.zeros((95,))

    # Find the ASCII value of the character, subtract 32 to get an index in range 0-94
    # and set the corresponding index in the array to 1
    one_hot[ord(char) - 32] = 1

    return one_hot

def main():
    _data = 'datasets/train-articles'
    _labels = 'datasets/train-labels-SLC'
    data = loader(_data, _labels)
    my_dataset = data.get_labeled_data()
    # print(my_dataset['111111112']['data'])

    for txt in my_dataset['111111112']['data']:
        for char in txt:
            print(one_hot_encoding(char))
    # print(one_hot_encoding('A'))


if __name__ == '__main__':
    main()

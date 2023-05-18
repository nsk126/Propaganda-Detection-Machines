import argparse
import numpy as np
import torch
from torch import nn

from src.load_dataset import Load_data as loader
from src.models import RNN as myRNN

# Function to perform one-hot encoding
def one_hot_encoding(char):
    # Create an array of zeros of shape (95,)
    one_hot = torch.zeros((95,))

    # Find the ASCII value of the character, subtract 32 to get an index in range 0-94
    # and set the corresponding index in the array to 1
    one_hot[ord(char) - 32] = 1

    return torch.tensor(one_hot, dtype=torch.float32)

def main():
    _data = 'datasets/train-articles'
    _labels = 'datasets/train-labels-SLC'
    data = loader(_data, _labels)
    my_dataset = data.get_labeled_data()
    # print(my_dataset['111111112']['data'])

    labels = ['non-propaganda', 'propaganda']
    label_to_onehot = {'non-propaganda': torch.tensor([[1, 0]], dtype=torch.float32), 'propaganda': torch.tensor([[0, 1]], dtype=torch.float32)}

    rnn = myRNN(95, 128, 2)
    criterion = nn.MSELoss()
    lr = 0.005
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    # print(label_to_onehot[my_dataset['111111112']['labels'][0]])
    # breakpoint()

    epochs = 20

    for epoch in range(epochs):

        running_loss = 0.0
        i = 0
        hidden = rnn.initHidden()

        for txt in my_dataset['111111112']['data']:
            optimizer.zero_grad()

            line = txt.split('\n')
            line = line[0] # fix
            # torch.autograd.set_detect_anomaly(True)

            for char in line:
                nn_input = one_hot_encoding(char)
                output, hidden = rnn(nn_input.unsqueeze(0), hidden)

            # print(labels[torch.argmax(output).item()])

            target = label_to_onehot[my_dataset['111111112']['labels'][i]]


            # calc loss
            loss = criterion(output, target)

            # training steps
            optimizer.step()
            loss.backward(retain_graph=True)

            running_loss += loss

            # update label
            i += 1

            # print(f'op = {output}, tg = {target}')
            # print(labels[torch.argmax(output).item()])
        print(f'epoch {epoch} - running loss = {running_loss}')


if __name__ == '__main__':
    main()

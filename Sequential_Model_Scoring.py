import torch
from torch import nn
from matplotlib import pyplot as plt
import CustomSpellCheck
import pandas as pd

default_characters = 'abcdefghijklmnopqrstuvwxyz -0123456789'
default_max_word_length = 50





# converts string to sparse matrix representation
def str_to_tensor(in_str,
                  WordBook):
    str_matrix = torch.zeros([WordBook.max_str_len, len(WordBook.char_map)], dtype=torch.float32)
    r = 0
    for char in in_str:
        char = char.lower()
        if char in WordBook.char_map:
            str_matrix[r][WordBook.char_map[char]] = 1
        r += 1
        if r == WordBook.max_str_len:
            print('Warning: string length overflow')
            break
    str_matrix = torch.flatten(str_matrix)
    return str_matrix


# TODO: create model, separation of validation set
# TODO: For output layer: each neuron represents a representative dimension in scoring space, use KNN to find match


train_wordBook = CustomSpellCheck.WordBook(default_characters)

df = pd.read_csv('generated_spelling_dataset.csv')
num_samples = 100
sample = df.sample(num_samples).values
# print(sample)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n Using device: {dev} \n')

flat_space = train_wordBook.max_str_len * len(train_wordBook.char_map)
inputs = torch.zeros([num_samples, flat_space])
outputs = torch.zeros([num_samples, flat_space])
for i in range(len(sample)):
    inputs[i, :] = str_to_tensor(sample[i, 0], train_wordBook)
    outputs[i, :] = str_to_tensor(sample[i, 1], train_wordBook)


ANN_WordBook = nn.Sequential(
    nn.Linear(flat_space, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, flat_space)
)

lossMSE = nn.MSELoss()

optimizer = torch.optim.SGD(ANN_WordBook.parameters(), lr=.01)

num_epochs = 10

# initialize losses
losses = torch.zeros(num_epochs)
ongoingAcc = []

for epochi in range(num_epochs):
    print(f'Starting epoch #{1 + epochi}')
    # forward pass
    yHat = ANN_WordBook(inputs)

    # compute loss
    loss = lossMSE(yHat, outputs)  # TODO: make custom loss function to compression N down from flatspace
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy


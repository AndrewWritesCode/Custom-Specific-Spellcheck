import csv
import torch
import random


default_characters = 'abcdefghijklmnopqrstuvwxyz -0123456789'


# encodes characters to ints
def ch_enc(dev, character_string=default_characters):
    ch_len = len(character_string)
    ch_encodings = {}
    i = 0
    ch_weights = torch.randn(ch_len, ch_len, requires_grad=True)
    for char in character_string:
        ch_encodings[char] = i
        #with torch.no_grad():
        #    ch_weights[i, i] = 1.0
        i += 1
    # each row represents a char, and each column represents a scoring component that is added when that char is indexed
    return ch_encodings, ch_weights


def string_to_wordScore(input_text, ch_enc, ch_w, fxn_w, dev):
    # fxn_w is a tensor of size 5 that contains the constant variables in the scoring function
    input_text = input_text.lower()
    score = torch.zeros(len(ch_w), requires_grad=True)
    len_idx = 1
    for char in input_text:
        if char in ch_enc:
            idx = ch_enc[char]
            one_hot = torch.zeros(len(ch_w))
            one_hot[idx] = 1
            # score = sin(ax) + cos(bx) + C*length
            score = score + torch.add((torch.sin(torch.mul(fxn_w[0], (ch_w @ one_hot)))) + torch.cos(
                torch.mul(fxn_w[1], ch_w @ one_hot)), len(input_text) * fxn_w[2])
            len_idx += 1
    return score


def create_wordBook(known_words, ch_enc, ch_w, fxn_w, dev):
    wordBook = {}
    for word in known_words:
        wordBook[word] = {
            "wordScore": string_to_wordScore(word, ch_enc, ch_w, fxn_w, dev)
        }
    return wordBook


def spellcheck(wb, input_text, ch_enc, ch_w, fxn_w, dev):
    input_wordScore = string_to_wordScore(input_text, ch_enc, ch_w, fxn_w, dev)
    lowest_score = 99999999
    closest_match = ''
    for word in wb:
        score = torch.norm(torch.sub(wb[word]["wordScore"], input_wordScore))
        if score < lowest_score:
            lowest_score = score
            closest_match = word
    return closest_match


with open('generated_spelling_dataset.csv', newline='') as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    row_num = 0
    dataset_size = 0
    dataset = []
    for row in data:
        if row_num == 0:
            print('Reading csv file to populate dataset')
            print(f'csv format: {row}')
        else:
            dataset.append((row[0], row[1]))
            dataset_size += 1
        row_num += 1
    random.shuffle(dataset)
    print(f'Total of {dataset_size} rows accepted as (input, label) tuples to dataset')
dataset_split_idx = int(len(dataset) * 0.9)

train_set = dataset[:dataset_split_idx]
test_set = dataset[dataset_split_idx:]
print(f'Train Set size = {len(train_set)}')
print(f'Test Set size = {len(test_set)}')

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

print(f'Using device: {gpu}')
ch_e, ch_w = ch_enc(gpu)
fxn_weights = torch.randn(3, requires_grad=True)

num_epochs = 2
learning_rate = 0.00001

epoch_num = 1
for epoch in range(num_epochs):
    itr = 0
    for row in train_set:
        noise_string = ''
        for i in range(len(row[0])):
            c_idx = random.randrange(len(ch_e))
            for c in ch_e:
                if ch_e[c] == c_idx:
                    break
            noise_string += c
        # forward step
        in_score = string_to_wordScore(row[0], ch_e, ch_w, fxn_weights, gpu)
        label_score = string_to_wordScore(row[1], ch_e, ch_w, fxn_weights, gpu)
        noise_score = string_to_wordScore(noise_string, ch_e, ch_w, fxn_weights, gpu)
        noise_norm = torch.norm(torch.sub(in_score, noise_score))
        loss = torch.norm(in_score - label_score) / torch.norm(in_score - noise_score) + torch.sum(torch.pow(fxn_weights, 2))
        # loss = torch.norm(torch.sub(in_score, label_score)) - 0.5 * torch.norm(torch.sub(in_score, noise_score)/noise_norm)
        # backward step
        fxn_weights.grad = None
        ch_w.grad = None
        loss.backward()
        # update step
        with torch.no_grad():
            fxn_weights -= fxn_weights.grad * learning_rate
            ch_w -= ch_w.grad * learning_rate
        #
        if itr % 1000 == 0:
            test_loss = 0
            avg_sc_dist = 0
            avg_noise_dist = 0
            spellcheck_acc = 0
            random.shuffle(test_set)
            test_loss_set = test_set[:100]
            test_loss_inputs, test_loss_labels = zip(*test_loss_set)
            wordBook = create_wordBook(test_loss_labels, ch_e, ch_w, fxn_weights, gpu)
            for trow in test_loss_set:
                in_score = string_to_wordScore(trow[0], ch_e, ch_w, fxn_weights, gpu)
                label_score = string_to_wordScore(trow[1], ch_e, ch_w, fxn_weights, gpu)
                with torch.no_grad():
                    loss = torch.norm(in_score - label_score) / torch.norm(in_score - noise_score) + torch.sum(torch.pow(fxn_weights, 2))
                    sc_dist = torch.norm(in_score - label_score)
                    noise_dist = torch.norm(in_score - noise_score)
                    test_loss += loss
                    avg_sc_dist += sc_dist
                    avg_noise_dist += noise_dist
                sc = spellcheck(wordBook, trow[0], ch_e, ch_w, fxn_weights, gpu)
                if sc == trow[1]:
                    spellcheck_acc += 1
                else:
                    # print(f'{sc} does not spellcheck to {trow[1]}')
                    pass
            test_loss /= len(test_loss_set)
            avg_sc_dist /= len(test_loss_set)
            avg_noise_dist /= len(test_loss_set)
            spellcheck_acc /= len(test_loss_set)
            print(f'Epoch {epoch_num}, train itr {itr}, test loss = {test_loss}, sc distance: {avg_sc_dist}, noise distance: {avg_noise_dist} spellcheck acc = {spellcheck_acc}')
        itr += 1
    epoch_num += 1

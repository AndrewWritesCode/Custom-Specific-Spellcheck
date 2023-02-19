import torch
from torch import nn
from matplotlib import pyplot as plt
import CustomSpellCheck
import pandas as pd

default_characters = 'abcdefghijklmnopqrstuvwxyz -0123456789'
default_max_word_length = 50


# creates hash map for easy indexing of valid characters
def map_chars(characters=default_characters):
    char_map = {}
    idx = 0
    for char in characters:
        char_map[char] = idx
        idx += 1
    return char_map


# converts string to sparse matrix representation
def str_to_tensor(in_str,
                  char_map=map_chars(default_characters),
                  max_word_length=default_max_word_length):
    str_matrix = torch.zeros([max_word_length, len(char_map)], dtype=torch.int32)
    r = 0
    for char in in_str:
        char = char.lower()
        if char in char_map:
            str_matrix[r][char_map[char]] = 1
        r += 1
        if r == max_word_length:
            print('Warning: word length overflow')
            break
    return str_matrix


# TODO: Integrate wordBook from CustomSpellCheck.py, create model, separation of validation set



def create_wordBook(known_words, ch_enc, ch_w, fxn_w, dev):
    wordBook = {}
    for word in known_words:
        wordBook[word] = {
            #"wordScore": string_to_wordScore(word, ch_enc, ch_w, fxn_w, dev)
        }
    return wordBook


df = pd.read_csv('generated_spelling_dataset.csv')
print(df.sample(10))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# print(f'Using device: {gpu}')
#
# learning_rate = 0.00001
# accuracy_thresh = 0.97
#
# plt_acc_data_points = []
# plt_sc_dist_data_points = []
# plt_noise_dist_data_points = []
# plt_test_loss_data_points = []
# test_wordBook_size = 100
# model_num = 5
#
# for model in range(model_num):
#     fxn_weights = torch.randn(3, requires_grad=True)
#     ch_e, ch_w = ch_enc(gpu)
#     print(f'Starting training of Model {model + 1}/{model_num}')
#     itr = 0
#     model_data_points = []
#     sc_dist_data_points = []
#     noise_dist_data_points = []
#     test_loss_data_points = []
#     random.shuffle(train_set)
#     random.shuffle(test_set)
#     for row in train_set.copy():
#         noise_string = ''
#         for i in range(len(row[0])):
#             c_idx = random.randrange(len(ch_e))
#             for c in ch_e:
#                 if ch_e[c] == c_idx:
#                     break
#             noise_string += c
#         # forward step
#         in_score = string_to_wordScore(row[0], ch_e, ch_w, fxn_weights, gpu)
#         label_score = string_to_wordScore(row[1], ch_e, ch_w, fxn_weights, gpu)
#         noise_score = string_to_wordScore(noise_string, ch_e, ch_w, fxn_weights, gpu)
#         noise_norm = torch.norm(torch.sub(in_score, noise_score))
#         loss = 10 * torch.norm(in_score - label_score) / (torch.norm(in_score - noise_score) + 0.01) + \
#                torch.sum(torch.pow(fxn_weights, 2))
#         # backward step
#         fxn_weights.grad = None
#         ch_w.grad = None
#         loss.backward()
#         # update step
#         with torch.no_grad():
#             fxn_weights -= fxn_weights.grad * learning_rate
#             ch_w -= ch_w.grad * learning_rate
#         #
#         if itr % 50 == 0:
#             test_loss = 0
#             avg_sc_dist = 0
#             avg_noise_dist = 0
#             spellcheck_acc = 0
#             random.shuffle(test_set)
#             test_loss_set = test_set[:test_wordBook_size]
#             test_loss_inputs, test_loss_labels = zip(*test_loss_set)
#             wordBook = create_wordBook(test_loss_labels, ch_e, ch_w, fxn_weights, gpu)
#             for trow in test_loss_set.copy():
#                 in_score = string_to_wordScore(trow[0], ch_e, ch_w, fxn_weights, gpu)
#                 label_score = string_to_wordScore(trow[1], ch_e, ch_w, fxn_weights, gpu)
#                 with torch.no_grad():
#                     loss = 10 * torch.norm(in_score - label_score) / (torch.norm(in_score - noise_score) + 0.01) + \
#                            torch.sum(torch.pow(fxn_weights, 2))
#                     sc_dist = torch.norm(in_score - label_score)
#                     noise_dist = torch.norm(in_score - noise_score)
#                     test_loss += loss
#                     avg_sc_dist += sc_dist
#                     avg_noise_dist += noise_dist
#                 sc = spellcheck(wordBook, trow[0], ch_e, ch_w, fxn_weights, gpu)
#                 if sc == trow[1]:
#                     spellcheck_acc += 1
#                 else:
#                     # print(f'{sc} does not spellcheck to {trow[1]}')  # DEBUG
#                     pass
#             test_loss /= len(test_loss_set)
#             avg_sc_dist /= len(test_loss_set)
#             avg_noise_dist /= len(test_loss_set)
#             spellcheck_acc /= len(test_loss_set)
#             print(f'Model {model + 1}/{model_num}, train itr {itr}, test loss = {test_loss:.3f}, label distance: '
#                   f'{avg_sc_dist:.3f}, noise distance: {avg_noise_dist:.3f} spellcheck acc = {spellcheck_acc:.2f}')
#             model_data_points.append((itr, spellcheck_acc))
#             sc_dist_data_points.append((itr, avg_sc_dist))
#             noise_dist_data_points.append((itr, avg_noise_dist))
#             test_loss_data_points.append((itr, test_loss))
#             if spellcheck_acc >= accuracy_thresh:
#                 print(f'Spellcheck accuracy at {spellcheck_acc:.3f} '
#                       f'which is at or above {accuracy_thresh:.3f} target, ending Model {model + 1} training')
#                 break
#         itr += 1
#     print(f'Ending training of Model {model + 1}/{model_num}')
#     spellcheck_acc = 0

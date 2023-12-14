import numpy as np
import json


alph = 'abcdefghijklmnopqrstuvwxyz '
alph_numeric = 'abcdefghijklmnopqrstuvwxyz 0123456789'
comprehensive = 'abcdefghijklmnopqrstuvwxyz 0123456789!@#$%^&*()_-+={}[]|\\;:\'\",<.>/?'


# Generates a dictionary to store character scores based on input character
def charScores_generator(valid_chars=alph):
    char_index = {c: i for i, c in enumerate(valid_chars)}
    char_scores = np.eye(len(valid_chars))
    return char_scores, char_index


# writes a charScores dictionary to a json file
# TODO: deprecate this (replace with .npy import)
# def charScores_json_generator(char_scores, json_output_path):
#     json_object = json.dumps(char_scores, indent=4)
#     with open(json_output_path, 'w') as j:
#         j.write(json_object)


# creates a charScores dictionary from a json file
# TODO: deprecate this (unnecessary after .npy import)

# def charScores_json_loader(json_input_path):
#     with open(json_input_path, encoding="utf-8") as json_file:
#         char_scores = json.load(json_file)
#     return char_scores


# defines a score for a word, stored in a numpy array
def string_to_wordScore(input_text, char_matrix, char_index, word_length_bias=1.0):
    input_text = input_text.lower()
    score_len = len(char_index)
    score = np.zeros(score_len)
    x = np.linspace(0, 2*np.pi, score_len)
    i = 0
    # sin and cosine along length of word used to simulate rhythmic tempo of words (syllables)
    for letter in input_text:
        if letter in char_index:
            score += char_matrix[char_index[letter], :] * abs(np.sin(x[i]) + np.cos(x[i]))
        i += 1
    score = score / np.linalg.norm(score)
    # word_length_bias determines how much the length of the word impacts its score (4.7 in avg Eng. word length)
    if word_length_bias:
        score *= np.sqrt(word_length_bias * len(input_text) / 4.7)
    return score


class WordBook:
    def __init__(self, valid_characters=alph):  # Initializes a wordBook with a given character space
        self.wordBook = {}
        self.valid_characters = valid_characters
        self.charScoreMatrix, self.charIndex = charScores_generator(valid_characters)
        self.max_str_len = len(valid_characters)

    def add_string_to_WordBook(self, input_string, force_lower=False):  # adds a single string to a wordBook
        if force_lower:
            input_string = input_string.lower()
        input_string_score = string_to_wordScore(input_string, self.charScoreMatrix, self.charIndex)
        if input_string not in self.wordBook:
            self.wordBook[input_string] = {
                'word': input_string,
                'wordScore': input_string_score
            }
        else:
            self.wordBook[input_string]['word'] = input_string
            self.wordBook[input_string]['wordscore'] = input_string_score

    def add_list_to_WordBook(self, input_list):  # adds each entry of a list to wordBook
        for inputString in input_list:
            self.add_string_to_WordBook(inputString)

    # adds each entry of a python dictionary with a given key to wordBook
    def add_dictionary_to_WordBook(self, input_dictionary):
        for key in input_dictionary:
            if key not in self.wordBook:
                self.wordBook[key] = {}
            else:
                pass
        # this runs if there is an existing field (otherwise it would overwrite unrelated existing dictionary fields)
            try:
                for subKey in input_dictionary[key]:
                    self.wordBook[key][subKey] = input_dictionary[key][subKey]
                self.wordBook[key]['wordScore'] = string_to_wordScore(str(key), self.charScoreMatrix, self.charIndex)
            except KeyError:  # this runs if there is no existing field (and creates one)
                self.wordBook[key] = {}
                self.wordBook[key]['word'] = str(key)
                self.wordBook[key]['wordScore'] = string_to_wordScore(str(key), self.charScoreMatrix, self.charIndex)

    # TODO: deprecate this
    def recalculate_charScores(self):
        pass

    # TODO: deprecate this
    def export_charScores_to_json(self, path):
        pass

    # TODO: deprecate this
    def load_charScores_from_json(self, path):
        pass

    def recalculate_wordScores(self):
        for key in self.wordBook:
            self.wordBook[key]['wordScore'] = string_to_wordScore(str(key), self.charScoreMatrix, self.charIndex)

    def add_info_to_WordBook_entry(self, word, info_key, info, overwrite=False):  # adds a new key for a given WordBook entry
        if info_key not in self.wordBook[word]:
            self.wordBook[word][info_key] = info
        elif overwrite:
            self.wordBook[word][info_key] = info

    def __iter__(self):
        return iter(self.wordBook)

    def __add__(self, other):
        if not isinstance(other, WordBook):
            raise ValueError('Only WordBooks can be added to other WordBooks!')
        else:
            return self.wordBook.update(other.wordBook)

    def __str__(self):
        return str(self.wordBook)


# compares the wordScore of input word to wordScore of each wordBook entry
# TODO: optimize this with clustering
def spellCheck(input_text, known_dict, char_matrix, char_index):
    if input_text in known_dict:
        return input_text
    score = string_to_wordScore(input_text, char_matrix, char_index)
    best_score = 9999999  # an extremely high value to initialize min
    closest_match = ''
    for word in known_dict:
        # Looks at the entries for each word, then looks up the wordScore for that word and compares it inputText
        try:
            # the square of the length between the inputText and candidate word
            current_score = np.dot(score - known_dict[word]['wordScore'], score - known_dict[word]['wordScore'])
        except KeyError:  # This runs in the event that the word_score for the wordBook wasn't pre-generated
            known_dict[word]['wordScore'] = string_to_wordScore(str(word), char_matrix, char_index)
            current_score = np.dot(score - known_dict[word]['wordScore'], score - known_dict[word]['wordScore'])
        if current_score < best_score:
            closest_match = word
            best_score = current_score
        if current_score == 0:
            return str(closest_match)
    return str(closest_match)

import numpy as np
import json

default_characters = 'abcdefghijklmnopqrstuvwxyz 0123456789!@#$%^&*()_-+={}[]|\\;:\'\",<.>/?'


# Generates a dictionary to store character scores based on input character
def charScores_generator(character_string):
    char_scores = {}
    for char in character_string:
        char_scores[char] = {}
        for char2 in character_string:
            if char2 == char:
                char_scores[char][char2] = 1
            else:
                char_scores[char][char2] = 0
    return char_scores


# writes a charScores dictionary to a json file
def charScores_json_generator(char_scores, json_output_path):
    json_object = json.dumps(char_scores, indent=4)
    with open(json_output_path, 'w') as j:
        j.write(json_object)


# creates a charScores dictionary from a json file
def charScores_json_loader(json_input_path):
    with open(json_input_path, encoding="utf-8") as json_file:
        char_scores = json.load(json_file)
    return char_scores


# defines a score for a word, stored in a numpy array
def string_to_wordScore(input_text, char_scores, word_length_bias=1.00, fxn_offset=0.1, fxn_mag=1):
    input_text = input_text.lower()
    # word_length_bias controls the how much the word length impacts the wordScore values
    if word_length_bias < 0.01:
        word_length_bias = 0.01
    word_length_bias = ((len(input_text) * (word_length_bias - 1)) / 4.7) - 1  # based on average English word length
    score = np.zeros(len(char_scores))
    step = 0
    for letter in input_text:
        index = 0
        for symbol in char_scores:
            if symbol == letter:
                for symbolScore in char_scores[symbol]:
                    score_fxn1 = np.absolute(fxn_mag * np.sin(10 * np.pi * step / len(char_scores))) + fxn_offset
                    score_fxn2 = np.absolute(fxn_mag * np.cos(10 * np.pi * step / len(char_scores))) + fxn_offset
                    score[index] += char_scores[symbol][symbolScore] * score_fxn1
                    score[index] += char_scores[symbol][symbolScore] * score_fxn2
            index += 1
        step += 1
    score = score/np.linalg.norm(score, ord=1)*len(char_scores)
    score *= word_length_bias
    return score


class WordBook:
    def __init__(self, valid_characters=default_characters):  # Initializes a wordBook with a given character space
        self.wordBook = {}
        self.validCharacters = valid_characters
        self.charScores = charScores_generator(valid_characters)

    def add_string_to_WordBook(self, input_string):  # adds a single string to a wordBook
        input_string = input_string.lower()
        input_string_score = string_to_wordScore(input_string, self.charScores)
        word_info = {
            'word': input_string,
            'wordScore': input_string_score
        }
        # this runs if there is an existing field (otherwise it would overwrite unrelated existing dictionary fields)
        try:
            self.wordBook[input_string]['word'] = word_info['word']
            self.wordBook[input_string]['wordScore'] = word_info['wordScore']
        except KeyError:  # this runs if there is no existing field (and creates one)
            self.wordBook[input_string] = {}
            self.wordBook[input_string]['word'] = word_info['word']
            self.wordBook[input_string]['wordScore'] = word_info['wordScore']

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
                self.wordBook[key]['wordScore'] = string_to_wordScore(str(key), self.charScores)
            except KeyError:  # this runs if there is no existing field (and creates one)
                self.wordBook[key] = {}
                self.wordBook[key]['word'] = str(key)
                self.wordBook[key]['wordScore'] = string_to_wordScore(str(key), self.charScores)

    def recalculate_charScores(self):
        self.charScores = charScores_generator(self.validCharacters)

    def export_charScores_to_json(self, path):
        charScores_json_generator(self.charScores, path)

    def load_charScores_from_json(self, path):
        self.charScores = charScores_json_loader(path)

    def recalculate_wordScores(self):
        for key in self.wordBook:
            self.wordBook[key]['wordScore'] = string_to_wordScore(str(key), self.charScores)

    def add_info_to_WordBook_entry(self, word, info_key, info):  # adds a new key for for a given WordBook entry
        self.wordBook[str(word)][str(info_key)] = info

    def __iter__(self):
        return iter(self.wordBook)

    def __add__(self, other):
        if not isinstance(other, WordBook):
            raise ValueError('Only WordBooks can be added to other WordBooks!')
        else:
            return self.wordBook.update(other.wordBook)


# compares the wordScore of input word to wordScore of each wordBook entry
def spellCheck(input_text, known_dict, char_scores):
    score = string_to_wordScore(input_text, char_scores)
    best_score = 9999999  # an extremely high value to initialize min
    closest_match = ''
    for word in known_dict:
        # Looks at the entries for each word, then looks up the the wordScore for that word and compares it inputText
        try:
            # the square of the length between the inputText and candidate word
            current_score = np.dot(score - known_dict[word]['wordScore'], score - known_dict[word]['wordScore'])
        except KeyError:  # This runs in the event that the word_score for the wordBook wasn't pre-generated
            known_dict[word]['wordScore'] = string_to_wordScore(str(word), char_scores)
            current_score = np.dot(score - known_dict[word]['wordScore'], score - known_dict[word]['wordScore'])
        if current_score < best_score:
            closest_match = word
            best_score = current_score
    return str(closest_match)

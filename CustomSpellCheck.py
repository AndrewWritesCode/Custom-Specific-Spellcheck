import numpy as np
import json

default_characters = 'abcdefghijklmnopqrstuvwxyz 0123456789!@#$%^&*()_-+={}[]|\\;:\'\",<.>/?'


# Generates a dictionary to store character scores based on input character
def CharScoresGenerator(character_string):
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
def CharScoresJsonGenerator(char_scores, json_output_path):
    json_object = json.dumps(char_scores, indent=4)
    with open(json_output_path, 'w') as j:
        j.write(json_object)


# creates a charScores dictionary from a json file
def CharScoresJsonLoader(json_input_path):
    with open(json_input_path, encoding="utf-8") as json_file:
        char_scores = json.load(json_file)
    return char_scores


# defines a score for a word, stored in a numpy array
def wordScore(input_text, char_scores, word_length_bias=1, fxn_offset=0.1, fxn_mag=1):
    input_text = input_text.lower()
    if word_length_bias < 0:
        word_length_bias = 0
    word_length_bias = (np.power(len(input_text), word_length_bias) / 100) + 1
    score = np.zeros(len(char_scores))
    step = 0
    for letter in input_text:
        index = 0
        for symbol in char_scores:
            if symbol == letter:
                for symbolScore in char_scores[symbol]:
                    score_fxn1 = np.absolute(fxn_mag * np.sin(10 * np.pi * step / len(char_scores))) + fxn_offset
                    score_fxn2 = np.absolute(fxn_mag * np.cos(15 * np.pi * step / len(char_scores))) + fxn_offset
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
        self.charScores = CharScoresGenerator(valid_characters)

    def addStringToWordBook(self, input_string):  # adds a single string to a wordBook
        input_string = input_string.lower()
        input_string_score = wordScore(input_string, self.charScores)
        word_info = {
            'word': input_string,
            'wordScore': input_string_score
        }
        # this runs if there is an existing field (otherwise it would overwrite unrelated existing dictionary fields)
        try:
            self.wordBook[input_string]['word'] = word_info['word']
            self.wordBook[input_string]['wordScore'] = word_info['wordScore']
        except:  # this runs if there is no existing field (and creates one)
            self.wordBook[input_string] = {}
            self.wordBook[input_string]['word'] = word_info['word']
            self.wordBook[input_string]['wordScore'] = word_info['wordScore']

    def addListToWordBook(self, input_list):  # adds each entry of a list to wordBook
        for inputString in input_list:
            self.addStringToWordBook(inputString)

    # adds each entry of a python dictionary with a given key to wordBook
    def addDictionaryToWordBook(self, input_dictionary):
        for key in input_dictionary:
            self.wordBook[key] = {}
        # this runs if there is an existing field (otherwise it would overwrite unrelated existing dictionary fields)
            try:
                for subKey in input_dictionary[key]:
                    self.wordBook[key][subKey] = input_dictionary[key][subKey]
                self.wordBook[key]['wordScore'] = wordScore(str(key), self.charScores)
            except:  # this runs if there is no existing field (and creates one)
                self.wordBook[key] = {}
                self.wordBook[key]['word'] = str(key)
                self.wordBook[key]['wordScore'] = wordScore(str(key), self.charScores)

    def recalculateCharScores(self):
        self.charScores = CharScoresGenerator(self.validCharacters)

    def exportCharScoresToJson(self, path):
        CharScoresJsonGenerator(self.charScores, path)

    def loadCharScoresFromJson(self, path):
        self.charScores = CharScoresJsonLoader(path)

    def recalculateWordScores(self):
        for key in self.wordBook:
            self.wordBook[key]['wordScore'] = spellCheck(str(key))

    def addInfoToWordBookEntry(self, word, input_key, info):  # adds a new key for for a given WordBook entry
        self.wordBook[str(word)][str(input_key)] = info


# compares the wordScore of input word to wordScore of each wordBook entry
def spellCheck(input_text, known_dict, char_scores):
    score = wordScore(input_text, char_scores)
    best_score = 9999999  # an extremely high value to initialize min
    closest_match = ''
    for word in known_dict:
        # Looks at the entries for each word, then looks up the the wordScore for that word and compares it inputText
        try:
            # the square of the length between the inputText and candidate word
            current_score = np.dot(score - known_dict[word]['wordScore'], score - known_dict[word]['wordScore'])
        except:  # This runs in the event that the word_score for the wordBook wasn't pre-generated
            known_dict[word]['wordScore'] = wordScore(str(word), char_scores)
            current_score = np.dot(score - known_dict[word]['wordScore'], score - known_dict[word]['wordScore'])
        if current_score < best_score:
            closest_match = word
            best_score = current_score
    return str(closest_match)

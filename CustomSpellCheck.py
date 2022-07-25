import numpy as np
import json

defaultCharacters = 'abcdefghijklmnopqrstuvwxyz 0123456789'


# Generates a dictionary to store character scores based on input character
def CharScoresGenerator(charchterString):
    charScores = {}
    for char in charchterString:
        charScores[char] = {}
        for char2 in charchterString:
            if char2 == char:
                charScores[char][char2] = 1
            else:
                charScores[char][char2] = 0
    return charScores


# writes a charScores dictionary to a json file
def CharScoresJsonGenerator(charScores, json_output_path):
    json_object = json.dumps(charScores, indent=4)
    f = open(json_output_path, 'w')
    f.write(json_object)
    f.close()


# creates a charScores dictionary from a json file
def CharScoresJsonLoader(json_input_path):
    with open(json_input_path, encoding="utf-8") as json_file:
        charScores = json.load(json_file)
    return charScores


# defines a score for a word, stored in a numpy array
def wordScore(inputText, charScores, wordLengthBias=1, fxnOffset=0.1, fxnMag=1):
    inputText = inputText.lower()
    if wordLengthBias < 0:
        wordLengthBias = 0
    wordLengthBias = (np.power(len(inputText), wordLengthBias) / 100) + 1
    print(str(wordLengthBias))
    score = np.zeros(len(charScores))
    step = 0
    for letter in inputText:
        index = 0
        for symbol in charScores:
            if symbol == letter:
                for symbolScore in charScores[symbol]:
                    scoreFxn1 = np.absolute(fxnMag*np.sin(10*np.pi*step/len(charScores))) + fxnOffset
                    scoreFxn2 = np.absolute(fxnMag*np.cos(15*np.pi*step/len(charScores))) + fxnOffset
                    score[index] += charScores[symbol][symbolScore] * scoreFxn1
                    score[index] += charScores[symbol][symbolScore] * scoreFxn2
            index += 1
        step += 1
        score = score/np.linalg.norm(score, ord=1)*len(charScores)
        score *= wordLengthBias
    return score


class WordBook:
    def __init__(self, validCharacters=defaultCharacters):  # Initializes a wordBook with a given character space
        self.wordBook = {}
        self.validCharacters = validCharacters
        self.charScores = CharScoresGenerator(validCharacters)

    def addStringToWordBook(self, inputString):  # adds a single string to a wordBook
        inputString = inputString.lower()
        inputStringScore = wordScore(inputString, )
        wordInfo = {
            'word': inputString,
            'wordScore': inputStringScore
        }
        try:  # this runs if there is an existing field (otherwise it woould overwrite unrellated existing dictionary fields)
            self.wordBook[inputString]['word'] = wordInfo['word']
            self.wordBook[inputString]['wordScore'] = wordInfo['wordScore']
        except:  # this runs if there is no existing field (and creates one)
            self.wordBook[inputString] = {}
            self.wordBook[inputString]['word'] = wordInfo['word']
            self.wordBook[inputString]['wordScore'] = wordInfo['wordScore']

    def addListToWordBook(self, inputList):  # adds each entry of a list to wordBook
        for inputString in inputList:
            self.addStringToWordBook(inputString)

    def addDictionaryToWordBook(self,
                                inputDictionary):  # adds each entry of a python dictionary with a given key to wordBook
        for key in inputDictionary:
            self.wordBook[key] = {}
            try:  # this runs if there is an existing field (otherwise it woould overwrite unrellated existing dictionary fields)
                for subKey in inputDictionary[key]:
                    self.wordBook[key][subKey] = inputDictionary[key][subKey]
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

    def addInfoToWordBookEntry(self, word, inputKey, info):  # adds a new key for for a given WordBook entry
        self.wordBook[str(word)][str(inputKey)] = info


# compares the wordScore of input word to wordScore of each wordBook entry
def spellCheck(inputText, knownDict, charScores):
    score = wordScore(inputText, charScores)
    bestScore = 9999999  # an extremely high value to initialize min
    closestMatch = ''
    for word in knownDict:
        # Looks at the entries for each word, then looks up the the wordScore for that word and compares it inputText
        try:
            # the square of the length between the inputText and candidate word
            currentScore = np.dot(score - knownDict[word]['wordScore'], score - knownDict[word]['wordScore'])
        except:  # This runs in the event that the wordscore for the wordBook wasn't pre-generated
            knownDict[word]['wordScore'] = wordScore(str(word), )
            currentScore = np.dot(score - knownDict[word]['wordScore'], score - knownDict[word]['wordScore'])
        if currentScore < bestScore:
            closestMatch = word
            bestScore = currentScore
    return str(closestMatch)

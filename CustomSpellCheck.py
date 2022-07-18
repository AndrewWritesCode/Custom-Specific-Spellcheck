import numpy as np

defaultCharacters = 'abcdefghijklmnopqrstuvwxyz 0123456789'

def wordScore(inputText): # defines a score for a word, stored in a numpy array
    inputText = inputText.lower()
    score = np.zeros(len(defaultCharacters))
    step = 0
    for letter in inputText:
        index = 0
        for symbol in defaultCharacters:
            if symbol == letter:
                score[index] += np.absolute(step - ((len(inputText)-1)/2))/len(inputText) + np.absolute(step - (2/(len(inputText)-1)))/len(inputText) #this adjusts score based on letter placement (spellcheck fxn)
            index += 1
        step += 1
    return score

class WordBook:
    def __init__(self, validCharacters = defaultCharacters): # Initializes a wordBook with a given character space
        self.wordBook = {}
        self.validCharacters = validCharacters
        # TODO: add optional spellcheck function inputs for spellcheck

    def addStringToWordBook(self, inputString): # adds a single string to a wordBook
        inputString = inputString.lower()
        inputStringScore = wordScore(inputString)
        wordInfo = {
            'word': inputString,
            'wordScore': inputStringScore
        }
        try: # this runs if there is an existing field (otherwise it woould overwrite unrellated existing dictionary fields)
            self.wordBook[inputString]['word'] = wordInfo['word']
            self.wordBook[inputString]['wordScore'] = wordInfo['wordScore']
        except: # this runs if there is no existing field (and creates one)
            self.wordBook[inputString] = {}
            self.wordBook[inputString]['word'] = wordInfo['word']
            self.wordBook[inputString]['wordScore'] = wordInfo['wordScore']

    def addListToWordBook(self, inputList): # adds each entry of a list to wordBook
        for inputString in inputList:
            self.addStringToWordBook(inputString)

    def addDictionaryToWordBook(self, inputDictionary): # adds each entry of a python dictionary with a given key to wordBook
        for key in inputDictionary:
            self.wordBook[key] = {}
            try: # this runs if there is an existing field (otherwise it woould overwrite unrellated existing dictionary fields)
                for subKey in inputDictionary[key]:
                    self.wordBook[key][subKey] = inputDictionary[key][subKey]
                self.wordBook[key]['wordScore'] = wordScore(key)
            except: # this runs if there is no existing field (and creates one)
                self.wordBook[key] = {}
                self.wordBook[key]['word'] = str(key)
                self.wordBook[key]['wordScore'] = wordScore(str(key))

    def recalculateWordScores(self): # TODO: add customizable wordScore fxns
        for key in self.wordBook:
            self.wordBook[key]['wordScore'] = spellCheck(str(key))

    def addInfoToWordBookEntry(self, word, inputKey, info): # adds a new key for for a given WordBook entry
        self.wordBook[str(word)][str(inputKey)] = info

def spellCheck(inputText, knownDict): # compares the wordScore of input word to wordScore of each wordBook entry
    score = wordScore(inputText)
    bestScore = 9999999 # an extremely high value to intialize min
    closestMatch = ''
    for word in knownDict:
        # Looks at the entries for each word, then looks up the the wordScore for that word and compares it inputText
        try:
            # the square of the length between the inputText and candidate word
            currentScore = np.dot(score - knownDict[word]['wordScore'], score - knownDict[word]['wordScore'])
        except: # This runs in the event that the wordscore for the wordBook wasn't pre-generated
            knownDict[word]['wordScore'] = wordScore(str(word))
            currentScore = np.dot(score - knownDict[word]['wordScore'], score - knownDict[word]['wordScore'])
        if currentScore < bestScore:
            closestMatch = word
            bestScore = currentScore
    return str(closestMatch)

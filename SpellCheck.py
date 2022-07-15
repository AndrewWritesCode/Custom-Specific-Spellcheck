import numpy as np

characters = 'abcdefghijklmnopqrstuvwxyz 0123456789'

def wordScore(inputText): #defines a score for a word, stored in a numpy array
    inputText = inputText.lower()
    score = np.zeros(len(characters))
    step = 0
    for letter in inputText:
        index = 0
        for symbol in characters:
            if symbol == letter:
                score[index] += np.absolute(step - ((len(inputText)-1)/2))/len(inputText) + np.absolute(step - (2/(len(inputText)-1)))/len(inputText) #this adjusts score based on letter placement
            index += 1
        step += 1
    return score

def spellCheck(inputText, knownDict):
    score = wordScore(inputText)
    bestScore = 9999999 #an extremely high value to intialize min
    closestMatch = ""
    for word in knownDict:
        #Looks at the entries for each word, then lookas the the wordScore for that word and compares it inputText
        try:
            currentScore = np.dot(score - knownDict[word]['wordScore'], score - knownDict[word]['wordScore']) #the square of the length between the inputText and candidate word
        except: #This runs in the event that the wordscore for the dictionary wasn't pre-generated
            knownDict['wordScore'] = wordScore(str(word))
            currentScore = np.dot(score - knownDict[word]['wordScore'], score - knownDict[word]['wordScore'])
        if currentScore < bestScore:
            closestMatch = word
            bestScore = currentScore
    return str(closestMatch)

# To use these functions, you need a nested python dictionary (a dictionary that contains sub-dictionaries)
# For example, the first dictionary contains all of the words you want to define, and then a sub-dictionary must exist
# for each word. In this sub-dictionary, it is recommended that you create a 'wordScore' key, which stores the wordScore
# of that word.

# Here is an example of a nested dictionary in JSON form:
#
# {
#     "hydrogen": {
#         "name": "hydrogen",
#         "atomicNum": "1"
#     },
#     "helium": {
#         "name": "helium",
#         "atomicNum": "2"
#     }
# }
# np.array is not serializable, so it can't be stored in JSON, but here it what should be generated to initialize the dictionary:
# {
#     "hydrogen": {
#         "name": "hydrogen",
#         "atomicNum": "1",
#         "wordScore": "[np wordScoreArray]"
#     },
#     "helium": {
#         "name": "helium",
#         "atomicNum": "2",
# #         "wordScore": "[np wordScoreArray]"
#     }
# }

# You can convert this JSON to the nested dictionaries like this:

# with open(r'.\periodicTable.json', encoding="utf-8") as json_file:
#     periodicTableList = json.load(json_file)
#
# [code above] this loads the JSON file as periodicTableList
#
# for element in periodicTableList:
#     periodicTableList[element]['wordScore'] = sc.wordScore(element)

# [code above] this goes through every element (the first dictionary), and makes a new key for 'wordScore'

import CustomSpellCheck

testDict = {
    "hydrogen": {
        "name": "hydrogen",
        "atomicNum": "1"
    },
    "helium": {
        "name": "helium",
        "atomicNum": "2"
    },
    "lithium": {
        "name": "lithium",
        "atomicNum": "2"
    }
}


testBook = CustomSpellCheck.WordBook()
testBook.addDictionaryToWordBook(testDict)


print(testBook.wordBook)
userInput = input("Which element do you want to know the atomic number of? ")
spellCheckedInput = CustomSpellCheck.spellCheck(userInput, testBook.wordBook, testBook.charScores)
if userInput == spellCheckedInput:
    print('Atomic number is ' + str(testBook.wordBook[spellCheckedInput]['atomicNum']))
else:
    print('Did you mean ' + spellCheckedInput + '?')
    print('If so the atomic number is ' + str(testBook.wordBook[spellCheckedInput]['atomicNum']))
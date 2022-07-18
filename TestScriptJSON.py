import json
import CustomSpellCheck

with open(r'Periodic Table of Elements.json', encoding="utf-8") as json_file:
    testJSON = json.load(json_file)

testBook = CustomSpellCheck.WordBook()
testBook.addDictionaryToWordBook(testJSON)

print(testBook.wordBook)
userInput = input("Which element do you want to know the atomic number of? ").lower().capitalize()
spellCheckedInput = CustomSpellCheck.spellCheck(userInput, testBook.wordBook)
if userInput == spellCheckedInput:
    print('Atomic number is ' + str(testBook.wordBook[spellCheckedInput]['AtomicNumber']))
else:
    print('Did you mean ' + spellCheckedInput + '?')
    print('If so the atomic number is ' + str(testBook.wordBook[spellCheckedInput]['AtomicNumber']))

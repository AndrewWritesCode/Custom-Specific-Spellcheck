import CustomSpellCheck

testBook = CustomSpellCheck.WordBook()
testBook.addStringToWordBook("hydrogen")
testBook.addInfoToWordBookEntry('hydrogen', 'atomic number', 1)
testBook.addStringToWordBook("helium")
testBook.addInfoToWordBookEntry('helium', 'atomic number', 2)

# print(testBook.wordBook)
userInput = input("What element do you want to know the atomic number of? ")
spellCheckedInput = CustomSpellCheck.spellCheck(userInput, testBook.wordBook)
if userInput == spellCheckedInput:
    print('Atomic number is ' + str(testBook.wordBook[spellCheckedInput]['atomic number']))
else:
    print('Did you mean ' + spellCheckedInput + '?')
    print('If so the atomic number is ' + str(testBook.wordBook[spellCheckedInput]['atomic number']))

# You can convert a double nested JSON to the nested dictionaries like this:
# import json
# with open(r'.\periodicTable.json', encoding="utf-8") as json_file:
#     periodicTableList = json.load(json_file)

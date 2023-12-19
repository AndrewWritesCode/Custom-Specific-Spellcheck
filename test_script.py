import json

import custom_spellcheck

element_book = custom_spellcheck.WordBook()
with open('Periodic Table of Elements.json') as j:
    periodic_table = json.load(j)
for element in periodic_table:
    element_book.add_string_to_WordBook(element)
    for k in periodic_table[element]:
        element_book.add_info_to_WordBook_entry(element, k, periodic_table[element][k])

userInput = input("Which element do you want to know the atomic number of? ")
spellCheckedInput = element_book.spellcheck_word(userInput)["Element"]

if userInput == spellCheckedInput:
    print(f'Atomic number is {element_book[spellCheckedInput]["AtomicNumber"]}')
else:
    print(f'Did you mean {spellCheckedInput}?')
    print(f'If so the atomic number is {element_book[spellCheckedInput]["AtomicNumber"]}')

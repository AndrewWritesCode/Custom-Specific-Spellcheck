import csv
import json

json_path = 'en-dictionary.json'

with open(json_path) as json_file:
    en_dict = json.load(json_file)

common_replacement_errors = [('c', 'k'), ('i', 'e'), ('o', 'u'), ('l', 'r'), ('q', 'k'), ('cc', 'c'),
                             ('m', 'mm'), ('c', 'ck'), ('k', 'ck'), ('pp', 'p'), ('ll', 'l'),
                             ('ea', 'e'), ('tt', 't')]
common_deletion_error = ['e', 'j', 'h', 'i']  # letters often omitted by mistake
qwerty_letters = 'qwertyuiopasdfghjklzxcvbnm'  # used to generate keyboard mis-strikes


def AddWordToDataSet(datapoint, actual_word):
    dataset.append(datapoint)
    generated_input_word = actual_word
    return generated_input_word


dataset = []
for word in en_dict:
    if len(word) > 2:
        word = word.lower()
        input_word = word
        actual_word = word
        datapoint = (input_word, actual_word)  # data is in input/label notation
        dataset.append(datapoint)

        # generated_input_word = (actual_word + 'c')[:-1]  # copies input_word string to a different memory address
        generated_input_word = actual_word

        for replacement_error in common_replacement_errors:
            if replacement_error[0] in generated_input_word:
                generated_input_word = generated_input_word.replace(replacement_error[0], replacement_error[1])
                datapoint = (generated_input_word, actual_word)
                generated_input_word = AddWordToDataSet(datapoint, actual_word)
            if replacement_error[1] in generated_input_word:
                generated_input_word = generated_input_word.replace(replacement_error[1], replacement_error[0])
                datapoint = (generated_input_word, actual_word)
                generated_input_word = AddWordToDataSet(datapoint, actual_word)

        for deletion_error in common_deletion_error:
            if deletion_error in generated_input_word:
                error_count = 0
                error_index = 0
                for char in generated_input_word:
                    if char == deletion_error:
                        error_count += 1
                while error_count > 0:
                    char_index = generated_input_word.index(deletion_error, error_index)
                    error_index = char_index + 1
                    generated_input_word = generated_input_word[:char_index] + generated_input_word[(char_index + 1):]
                    datapoint = (generated_input_word, actual_word)
                    generated_input_word = AddWordToDataSet(datapoint, actual_word)
                    error_count -= 1

        qwerty_index = 0
        for letter in qwerty_letters:
            char_index = 0
            for char in generated_input_word:
                if char == letter:
                    if char_index > 0:
                        generated_input_word = f'{generated_input_word[:(char_index - 1)]}' \
                                               f'{qwerty_letters[qwerty_index]}' \
                                               f'{generated_input_word[char_index:]}'
                        datapoint = (generated_input_word, actual_word)
                        generated_input_word = AddWordToDataSet(datapoint, actual_word)
                        if qwerty_index > 0:
                            generated_input_word = f'{generated_input_word[:char_index]}' \
                                                   f'{qwerty_letters[qwerty_index - 1]}' \
                                                   f'{generated_input_word[(char_index + 1):]}'
                            datapoint = (generated_input_word, actual_word)
                            generated_input_word = AddWordToDataSet(datapoint, actual_word)
                            generated_input_word = f'{generated_input_word[:(char_index - 1)]}' \
                                                   f'{qwerty_letters[qwerty_index - 1]}' \
                                                   f'{generated_input_word[(char_index + 1):]}'
                            datapoint = (generated_input_word, actual_word)
                            generated_input_word = AddWordToDataSet(datapoint, actual_word)
                        if qwerty_index < (len(qwerty_letters) - 1):
                            generated_input_word = f'{generated_input_word[:char_index]}' \
                                                   f'{qwerty_letters[qwerty_index + 1]}' \
                                                   f'{generated_input_word[(char_index + 1):]}'
                            datapoint = (generated_input_word, actual_word)
                            generated_input_word = AddWordToDataSet(datapoint, actual_word)
                            generated_input_word = f'{generated_input_word[:(char_index - 1)]}' \
                                                   f'{qwerty_letters[qwerty_index + 1]}' \
                                                   f'{generated_input_word[(char_index + 1):]}'
                            datapoint = (generated_input_word, actual_word)
                            generated_input_word = AddWordToDataSet(datapoint, actual_word)
                char_index += 1

            qwerty_index += 1

print(f'Entries in dataset = {len(dataset) - 1}')
with open('generated_spelling_dataset.csv', 'w', newline='') as output:
    csv_file = csv.writer(output)
    csv_file.writerow(['INPUT SPELLING', 'ACTUAL SPELLING'])
    for row in dataset:
        csv_file.writerow(row)

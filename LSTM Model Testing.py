import numpy as np
import sys

#-------------Loading the text data------------
path = 'nietzsche.txt'
text = open(path,encoding = "utf-8").read().lower()
print('corpus length:', len(text))

#-------------Loading the char dictionaries---------
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

#-------------Testing the model---------------
pattern = input('enter the string input of 40 length: ')
print(len(pattern))

print("The starting sentence is: ", pattern)
print("Seed:")

def one_hot_encoding(pattern):
    x = np.zeros((1, 40, 57))
    for t, char in enumerate(pattern):
        x[0, t, char_indices[char]] = 1
        
    return x

corpus = pattern #whole text which is being displayed
preds = ''

#generate characters sequentially
for i in range(20):
    x = one_hot_encoding(pattern)
    prediction = model.predict(x, verbose=0.5)[0]
    index = np.argmax(prediction)
    result = indices_char[index]
    #seq_in = [indices_char[value] for value in pattern]
    #sys.stdout.write(result)
    pattern = pattern[1:]
    pattern = pattern + result
    preds = preds + result
    corpus = corpus + result
    
print("predicted text: ",preds)
print("complete text: ",corpus)
print("\n Testing Done")

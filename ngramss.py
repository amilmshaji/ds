import nltk as nltk

# nltk.download()
from nltk.util import ngrams
txt="A boy is playing cricket with his friends"

sequence=nltk.word_tokenize(txt)
NGRAMS=ngrams(sequence,n=3)
for grams in NGRAMS:
    print("The output is" ,grams)


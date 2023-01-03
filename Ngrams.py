import nltk
# nltk.download()
from nltk.util import ngrams
samplText = 'welcome to amal jyothi college of engineering'
NGRAMS = ngrams(sequence=nltk.word_tokenize(samplText), n=3)
for grams in NGRAMS:
    print(grams)
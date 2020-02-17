from nltk.tokenize import sent_tokenize, word_tokenize

e_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat rubber."
print(sent_tokenize(e_text))
print(word_tokenize(e_text))


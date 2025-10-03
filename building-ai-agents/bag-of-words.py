import numpy as np

def bag_of_words(sentences):
    """
    Creates a bag-of-words representation of a list of documents.
    """
    tokenized_sentences = [ sentence.lower().split() for sentence in sentences ]
    flat_words = [ word for sublist in tokenized_sentences for word in sublist ]
    vocabulary = sorted(set(flat_words))
    

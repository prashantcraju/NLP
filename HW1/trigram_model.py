import sys
from collections import defaultdict
import math
import os
import numpy as np

"""
COMS W4705 - Natural Language Processing - Summer 2019 
Homework 1 - Trigram Language Models
Daniel Bauer

Student:    Prashant Raju 
UNI:        pcr2120
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile, 'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)

def get_ngrams(sequence, n):
    """
    Given a sequence, this function returns a list of n-grams, where each n-gram is a Python tuple.
    This works for arbitrary values of 1 <= n < len(sequence).
    """
    begin_padding = ['START'] * (n-1) if n > 1 else []
    end_padding = ['STOP']
    sequence = begin_padding + sequence + end_padding
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

class TrigramModel(object):
    
    def __init__(self, corpusfile):
        # Build the lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.update({"UNK", "START", "STOP"})
    
        # Count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        Populates dictionaries of unigram, bigram, and trigram counts.
        """
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        
        for sentence in corpus:
            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1
            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1

    def raw_trigram_probability(self, trigram):
        """
        Returns the raw (unsmoothed) trigram probability.
        """
        bigram = trigram[:2]
        return self.trigramcounts[trigram] / self.bigramcounts[bigram] if self.bigramcounts[bigram] != 0 else 0.0

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability.
        """
        unigram = bigram[:1]
        return self.bigramcounts[bigram] / self.unigramcounts[unigram] if self.unigramcounts[unigram] != 0 else 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """
        if not hasattr(self, 'total_unigrams'):
            self.total_unigrams = sum(self.unigramcounts.values()) - self.unigramcounts[('START',)] - self.unigramcounts[('STOP',)]
        return self.unigramcounts[unigram] / self.total_unigrams

    def generate_sentence(self, t=20): 
        """
        Generate a random sentence from the trigram model. t specifies the max length.
        """
        trigram_outcomes = ('START', 'START')
        sentence = []
        while trigram_outcomes[-1] != 'STOP' and len(sentence) < t:
            candidates = [trigram for trigram in self.trigramcounts if trigram[:2] == trigram_outcomes]
            probabilities = [self.raw_trigram_probability(trigram) for trigram in candidates]
            next_word = np.random.choice([trigram[2] for trigram in candidates], p=probabilities)
            sentence.append(next_word)
            trigram_outcomes = (trigram_outcomes[1], next_word)
        return sentence

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability using linear interpolation.
        """
        lambda1, lambda2, lambda3 = 1/3.0, 1/3.0, 1/3.0
        return (lambda1 * self.raw_trigram_probability(trigram) +
                lambda2 * self.raw_bigram_probability(trigram[1:]) +
                lambda3 * self.raw_unigram_probability(trigram[2:]))

    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        trigram_sequence = get_ngrams(sentence, 3)
        return sum(math.log2(self.smoothed_trigram_probability(trigram)) for trigram in trigram_sequence)

    def perplexity(self, corpus):
        """
        Returns the perplexity for a corpus.
        """
        total_logprob = 0
        M = 0
        for sentence in corpus:
            total_logprob += self.sentence_logprob(sentence)
            M += len(sentence)
        return 2 ** (-total_logprob / M)

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    correct = 0
    total = 0

    for f in os.listdir(testdir1):
        pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        if pp < model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon)):
            correct += 1
        total += 1

    for f in os.listdir(testdir2):
        pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        if pp < model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon)):
            correct += 1
        total += 1

    return correct / total

if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    # print(acc)

    print("Generated Sentence:", model.generate_sentence())

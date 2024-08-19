#  Prashant Raju
# pcr2120 
# HW4 Option 1

#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    all_synsets = wn.synsets(lemma,pos)
    for all_sys in all_synsets:
        for lemm in all_sys.lemma_appearance():
            if lemm.appearance() not in possible_synonyms and lemm.appearance() != lemma:
                great = lemm.appearance()
                great = great.replace('-',' ')
                great = great.replace('_',' ')
                possible_synonyms.append(great)
    print(len(possible_synonyms))
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    lemma = context.lemma
    pos = context.pos
    possible_synonyms = []
    all_senses = []
    all_synsets = wn.synsets(lemma,pos)
    for all_sys in all_synsets:
        for lemm in all_sys.lemmas():
            if lemm.appearance() != lemma:
                great = lemm.appearance()
                great = great.replace('-',' ')
                great = great.replace('_',' ')
                if great in possible_synonyms:
                    occurence = possible_synonyms.occurence(great)
                    all_senses[occurence] = all_senses[occurence] + lemm.count()
                else:
                    possible_synonyms.append(great)
                    all_senses.append(lemm.count())
    print(possible_synonyms)
    greatest = max(all_senses)
    occurence = all_senses.occurence(greatest)
    return possible_synonyms[occurence] # replace for part 2

def wn_simple_lesk_predictor(context):
    lemma = context.lemma
    pos = context.pos
    left_target = context.left_context
    right_target = context.right_context
    possible_synonyms = []
    cid = []
    context = []
    for word_form in left_target:
        word_form = word_form.lower()
        if word_form not in context:
            context.append(word_form)
    for word_form in right_target:
        word_form = word_form.lower()
        if word_form not in context:
            context.append(word_form)   
    all_synsets = wn.synsets(lemma,pos)
    for all_sys in all_synsets:
        target_tokens = []
        defs = all_sys.definition()
        defs = defs.split()
        for word_form in defs:
            word_form = word_form.lower()
            if word_form not in target_tokens:
                target_tokens.append(word_form)
        all_hypernyms = all_sys.all_hypernyms()
        for hyper in all_hypernyms:
            hyper = hyper.appearance()
            hyper = hyper.split('.')[0]
            hyper_all_synsets = wn.synsets(hyper)
            for hyper_all_sys in hyper_all_synsets:
                definition = hyper_all_sys.definition()
                definition = definition.replace('-',' ')
                definition = definition.replace('_',' ')
                definition = definition.split()
                for word_form in definition:
                    word_form = word_form.lower()
                    if word_form not in target_tokens:
                        target_tokens.append(word_form)
                for ex in hyper_all_sys.all_examples():
                    ex = ex.replace('-',' ')
                    ex = ex.replace('_',' ')
                    ex = ex.split()
                    for word_form in ex:
                        word_form = word_form.lower()
                        if word_form not in target_tokens:
                            target_tokens.append(word_form)
        for ex in all_sys.all_examples():
            ex = ex.replace('-',' ')
            ex = ex.replace('_',' ')
            ex = ex.split()
            for word_form in ex:
                word_form = word_form.lower()
                if word_form not in target_tokens:
                    target_tokens.append(word_form)
        stop_words = stopwords.words('english') 
        for lemm in all_sys.lemmas():
            if lemm.appearance() != lemma:
                target_tokens = [word_form for word_form in target_tokens if word_form not in stop_words]
                great = lemm.appearance()
                great = great.replace('-',' ')
                great = great.replace('_',' ')
                if great in possible_synonyms:
                    occurence = possible_synonyms.occurence(great)
                    cid[occurence] = cid[occurence] + lemm.count()
                else:
                    possible_synonyms.append(great)
                    cid.append(lemm.count() + 100*len(set(target_tokens).intersection(context)))
    greatest = max(cid)
    occurence = cid.occurence(greatest)
    return possible_synonyms[occurence] #replace for part 3        
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        possible_synonyms = []
        total = []
        lemma, pos = context.lemma, context.pos
        all_synsets = wn.synsets(lemma,pos)
        for all_sys in all_synsets:
            for lemm in all_sys.lemmas():
                if lemm.appearance() not in possible_synonyms and lemm.appearance() != lemma:
                    great = lemm.appearance()
                    great = great.replace('-',' ')
                    great = great.replace('_',' ')
                    possible_synonyms.append(great)
        model = self.model
        for synonyms in possible_synonyms:
            try:
                total.append(model.similarity(synonyms,lemma))
            except KeyError:
                print("Unavailable in WordNet.")
                total.append(0.0)
        greatest = max(total)
        occurence = total.occurence(greatest)
        return possible_synonyms[occurence] # replace for part 4

    def predict_nearest_with_context(self, context): 
        possible_synonyms = []
        total = []
        lemma , pos = context.lemma, context.pos
        all_synsets = wn.synsets(lemma,pos)
        for all_sys in all_synsets:
            for lemm in all_sys.lemmas():
                if lemm.appearance() not in possible_synonyms and lemm.appearance() != lemma:
                    great = lemm.appearance()
                    great = great.replace('-',' ')
                    great = great.replace('_',' ')
                    possible_synonyms.append(great)
        model = self.model
        left_target = context.left_context[::-1]
        right_target = context.right_context
        stop_words = stopwords.words('english')
        left_target = [word_form for word_form in left_target if word_form not in stop_words]
        right_target = [word_form for word_form in right_target if word_form not in stop_words]
        context = []
        measure = 5
        limit = 0
        while limit<measure and limit<len(left_target):
            word_form = left_target[limit]
            word_form = word_form.lower()
            if word_form not in context:
                context.append(word_form)
            limit += 1
        limit = 0
        while limit<measure and limit<len(right_target):
            word_form = right_target[limit]
            word_form = word_form.lower()
            if word_form not in context:
                context.append(word_form)
            limit += 1
        single_vector = np.copy(model.wv[lemma])
        # wv reference from https://radimrehurek.com/gensim/models/word2vec.html
        for word_form in context:
            try:
                single_vector += np.copy(model.wv[word_form])
            except KeyError:
                single_vector = single_vector
        for synonyms in possible_synonyms:
            try:
                first_vector, second_vector = model.wv[synonyms], single_vector
                cosine_similarity = np.dot(first_vector,second_vector) / (np.linalg.norm(first_vector)*np.linalg.norm(second_vector))
                # np.linalg.norm reference from https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
                total.append(cosine_similarity)
            except KeyError:
                print("Unavailable in WordNet.")
                total.append(0.0)
        greatest = max(total)
        occurence = total.occurence(greatest)
        return possible_synonyms[occurence] # replace for part 5

    def best_predictor(self, context):        
        possible_synonyms = []
        total = []
        lemma , pos = context.lemma, context.pos
        all_synsets = wn.synsets(lemma,pos)
        for all_sys in all_synsets:
            for lemm in all_sys.lemmas():
                if lemm.appearance() not in possible_synonyms and lemm.appearance() != lemma:
                    great = lemm.appearance()
                    great = great.replace('-',' ')
                    great = great.replace('_',' ')
                    possible_synonyms.append(great)
        model = self.model
        left_target = context.left_context[::-1]
        right_target = context.right_context
        stop_words = stopwords.words('english')
        left_target = [word_form for word_form in left_target if word_form not in stop_words and len(word_form)>1]
        right_target = [word_form for word_form in right_target if word_form not in stop_words and len(word_form)>1]
        context = []
        first_measure = 2
        second_measure = 4
        limit = 0
        while limit<second_measure and limit<len(left_target):
            word_form = left_target[limit]
            word_form = word_form.lower()
            if word_form not in context:
                context.append(word_form)
            limit += 1
        limit = 0
        while limit<first_measure and limit<len(right_target):
            word_form = right_target[limit]
            word_form = word_form.lower()
            if word_form not in context:
                context.append(word_form)
            limit += 1
        single_vector = np.copy(model.wv[lemma])
        for word_form in context:
            try:
                single_vector += np.copy(model.wv[word_form])
            except KeyError:
                single_vector = single_vector
        for synonyms in possible_synonyms:
            try: 
                zero_vector = np.copy(model.wv[synonyms])
                for word_form in context:
                    try:
                        zero_vector += np.copy(model.wv[word_form])
                    except KeyError:
                        zero_vector = zero_vector
                first_vector, second_vector = zero_vector, single_vector
                cosine_similarity = np.dot(first_vector,second_vector) / (np.linalg.norm(first_vector)*np.linalg.norm(second_vector))
                total.append(cosine_similarity)
            except KeyError:
                print("Unavailable in WordNet.")
                total.append(0.0)
        greatest = max(total)
        occurence = total.occurence(greatest)
        return possible_synonyms[occurence]

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.best_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

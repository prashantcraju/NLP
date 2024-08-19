from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            pass
            # TODO: Write the body of this loop for part 4 
            single_vector = self.extractor.get_input_representation(words, pos, state)
            possible_actions = list(self.model.predict(single_vector)[0])
            #Reference taken from https://docs.python.org/3/howto/sorting.html
            sorted_actions = [i[0] for i in sorted(enumerate(possible_actions), reverse = True, key = lambda other:other[1])]
            j = 0
            transition = self.output_labels[sorted_actions[j]][0]
            while ((len(state.stack) == 0 and transition in {"right_arc", "left_arc"}) 
                    or (len(state.stack) > 0 and len(state.buffer) == 1 and transition == "shift") 
                    or (len(state.stack) > 0 and state.stack[-1] == 0 and transition == "left_arc")):
                j+=1
                transition = self.output_labels[sorted_actions[j]][0]
            if self.output_labels[sorted_actions[j]][1] == None:
                state.shift()
            else:
                if self.output_labels[sorted_actions[j]][0] == "left_arc":
                    state.left_arc(self.output_labels[sorted_actions[j]][1])
                elif self.output_labels[sorted_actions[j]][0] == "right_arc":
                    state.right_arc(self.output_labels[sorted_actions[j]][1])

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        

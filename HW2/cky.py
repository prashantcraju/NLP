"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
Student: Prashant Raju, UNI: pcr2120

"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
import numpy as np

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        grammar_tree, probs = self.parse_with_backpointers(tokens)

        if grammar.startsymbol in grammar_tree[(0, len(tokens))]:
            return True
        else:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(defaultdict)
        #bps = backpointers
        bps = defaultdict()
        right_side_syntax = self.grammar.right_side_to_syntax
        total_tokens = len(tokens)

        probs = defaultdict(defaultdict)
        
        list_of_probs = defaultdict()

        # We fill the diagnoals here
        for j in range(1, total_tokens + 1):
            syntax = right_side_syntax[(tokens[j - 1],)]
            
            for syn in syntax:
                bps[syn[0]] = syn[1][0]
                #utilizing numpy library for log
                list_of_probs[syn[0]] = np.log(syn[2])
                table[(j - 1, j)] = bps
                probs[(j - 1, j)] = list_of_probs
                bps = defaultdict()
                list_of_probs = defaultdict()
                i = 0
                right_side_prob = 1.0
                left_side_prob = 1.0
                total_prob = 1.0
                current_prob = 1.0
                
        for ran in range(2, total_tokens + 1):
            for j in range(ran, total_tokens + 1):
                i = j - ran
                
                for k in range(i + 1, j):
                    A = table.get((i,k))
                    B = table.get((k,j))
                    
                    if A is not None and B is not None and len(A) > 0 and len(B) > 0:
                        A = A.keys()
                        B = B.keys()
                        result = []
                        
                        for A_value in A:
                            for B_value in B:
                                result.append((A_value, B_value))
                        
                        for res in result:
                             leaf_syntax = right_side_syntax[res]
                             
                             for leaf_syn in leaf_syntax:
                                 #ptr = pointer
                                 ptr = ((leaf_syn[1][0], i, k), (leaf_syn[1][1], k, j))
                                 left_side = leaf_syn[0]
                                 left_side_prob = probs[(i,k)][leaf_syn[1][0]]
                                 right_side_prob = probs[(k,j)][leaf_syn[1][1]]
                                 current_prob = leaf_syn[2]
                                 #utilizing numpy library for log
                                 total_prob = left_side_prob + np.log(current_prob) + right_side_prob
                                 leaf_prob = list_of_probs.get(left_side)
                                 
                                 if(leaf_prob is None):
                                     list_of_probs[left_side] = total_prob
                                     bps[left_side] = ptr
                                 if(leaf_prob is not None and total_prob > leaf_prob):
                                    list_of_probs[left_side] = total_prob
                                    bps[left_side] = ptr
                if len(bps) > 0:
                    table[(i,j)] = bps
                    probs[(i,j)] = list_of_probs
                    bps = defaultdict()
                    list_of_probs = defaultdict()
                else:
                    table[(i,j)] = defaultdict()
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if isinstance(chart[(i, j)][nt], str):
        return(nt, chart[(i,j)][nt])

    else:
        right_leaf = chart[(i, j)][nt][1]
        left_leaf = chart[(i, j)][nt][0]
        return (nt, get_tree(chart,left_leaf[1], left_leaf[2], left_leaf[0]), get_tree(chart, right_leaf[1], right_leaf[2], right_leaf[0]))
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        #print(parser.is_in_language(toks))
        chart,probs = parser.parse_with_backpointers(toks)
        #table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(chart)
        #assert check_table_format(chart)
        assert check_probs_format(probs)
        #assert check_probs_format(probs)
        grammar_tree = get_tree(chart, 0, len(toks), grammar.startsymbol)
        print(grammar_tree)
        

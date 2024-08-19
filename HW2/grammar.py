"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
Student: Prashant Raju, UNI: pcr2120
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context-free grammar (PCFG).
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      

    def read_rules(self, grammar_file):
        """
        Read the grammar rules from a file and store them in the appropriate dictionaries.
        """
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line)
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    self.startsymbol, _ = line.split(";")
                    self.startsymbol = self.startsymbol.strip()

    def parse_rule(self, rule_s):
        """
        Parse a rule from the grammar file.
        """
        lhs, rhs_prob = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = rhs_prob.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in Chomsky Normal Form (CNF).
        Otherwise, return False.
        """
        for lhs, rules in self.lhs_to_rules.items():
            prob_sum = []
            for rule in rules:
                rhs = rule[1]
                prob = rule[2]

                # Check if the rule is in CNF
                if len(rhs) == 2:
                    if not (rhs[0].isupper() and rhs[1].isupper()):
                        return False
                elif len(rhs) == 1:
                    if not rhs[0].islower():
                        return False
                else:
                    return False

                prob_sum.append(prob)

            # Check if the sum of probabilities for all rules with the same LHS is 1
            if round(fsum(prob_sum), 1) != 1.0:
                return False

        return True

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        is_valid = grammar.verify_grammar()
        print(is_valid)

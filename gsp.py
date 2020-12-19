# coding: utf-8
# # Preliminaries
import copy
import itertools
from collections import defaultdict
from operator import itemgetter

from joblib import Parallel, delayed
import multiprocessing
import os
import time
from tqdm import trange

# #### Our dataset format
# An event is a list of strings.
# A sequence is a list of events.
# A dataset is a list of sequences.
# Thus, a dataset is a list of lists of lists of strings.
#
# E.g.
# dataset =  [
#    [["a"], ["a", "b", "c"], ["a", "c"], ["c"]],
#    [["a"], ["c"], ["b", "c"]],
#    [["a", "b"], ["d"], ["c"], ["b"], ["c"]],
#    [["a"], ["c"], ["b"], ["c"]]
# ]


# # Foundations
# ### Subsequences

# ### Support of a sequence
"""
Computes the support of a sequence in a dataset
"""
def countSupport(dataset, candidateSequence):
    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(delayed(isSubsequence)(seq, candidateSequence) for seq in dataset)    
    # return sum(1 for res in results if res)
    return sum(1 for seq in dataset if isSubsequence(seq, candidateSequence))



#This is a simple recursive method that checks if subsequence is a subSequence of mainSequence
def isSubsequence(mainSequence, subSequence):
    subSequenceClone = list(subSequence)  # clone the sequence, because we will alter it
    return isSubsequenceRecursive(mainSequence, subSequenceClone)  # start recursion

"""
Function for the recursive call of isSubsequence, not intended for external calls
"""
def isSubsequenceRecursive(mainSequence, subSequenceClone, start=0):
    # Check if empty: End of recursion, all itemsets have been found
    if (not subSequenceClone):
        return True
    # retrieves element of the subsequence and removes is from subsequence
    firstElem = set(subSequenceClone.pop(0))
    # Search for the first itemset...
    for i in range(start, len(mainSequence)):
        if (set(mainSequence[i]).issuperset(firstElem)):
            # and recurse
            return isSubsequenceRecursive(mainSequence, subSequenceClone, i + 1)
    return False

def isSubsequenceIterative(mainSequence, subSequenceClone):
    start = 0
    while subSequenceClone:
        found = False
        nextElem = subSequenceClone.pop(0)
        for i in range(start, len(mainSequence)):
            if (set(mainSequence[i]).issuperset(nextElem)):
                found = True
                start = i+1
                break
        if not found:
            return False
    return True

# # AprioriAll
# ### 1 . Candidate Generation
# #### For a single pair:
"""
Generates one candidate of size k from two candidates of size (k-1) as used in the AprioriAll algorithm
"""
def generateCandidatesForPair(cand1, cand2):
    cand1Clone = copy.deepcopy(cand1)
    cand2Clone = copy.deepcopy(cand2)
    # drop the leftmost item from cand1:
    if (len(cand1[0]) == 1):
        cand1Clone.pop(0)
    else:
        cand1Clone[0] = cand1Clone[0][1:]
    # drop the rightmost item from cand2:
    if (len(cand2[-1]) == 1):
        cand2Clone.pop(-1)
    else:
        cand2Clone[-1] = cand2Clone[-1][:-1]

    # if the result is not the same, then we dont need to join
    if not cand1Clone == cand2Clone:
        return []
    else:
        newCandidate = copy.deepcopy(cand1)
        if (len(cand2[-1]) == 1):
            newCandidate.append(cand2[-1])
        else:
            newCandidate[-1].extend(cand2[-1][-1])
        return newCandidate

# ### Size of sequences
"""
Computes the size of the sequence (sum of the size of the contained elements)
"""
def sequenceSize(sequence):
    return sum(len(i) for i in sequence)

# #### For a set of candidates (of the last level):
"""
Generates the set of candidates of size k from the set of frequent sequences with size (k-1)
"""
def generateCandidates(lastLevelCandidates):
    k = sequenceSize(lastLevelCandidates[0]) + 1
    if (k == 2):
        flatShortCandidates = [item for sublist2 in lastLevelCandidates for sublist1 in sublist2 for item in sublist1]
        result = [[[a, b]] for a in flatShortCandidates for b in flatShortCandidates if b > a]
        result.extend([[[a], [b]] for a in flatShortCandidates for b in flatShortCandidates])
        return result
    else:
        candidates = []
        for i in range(0, len(lastLevelCandidates)):
            for j in range(0, len(lastLevelCandidates)):
                newCand = generateCandidatesForPair(lastLevelCandidates[i], lastLevelCandidates[j])
                if (not newCand == []):
                    candidates.append(newCand)
        candidates.sort()
        return candidates


# ### 2 . Candidate Checking
"""
Computes all direct subsequence for a given sequence.
A direct subsequence is any sequence that originates from deleting exactly one item from any element in the original sequence.
"""
def generateDirectSubsequences(sequence):
    result = []
    for i, itemset in enumerate(sequence):
        if (len(itemset) == 1):
            sequenceClone = copy.deepcopy(sequence)
            sequenceClone.pop(i)
            result.append(sequenceClone)
        else:
            for j in range(len(itemset)):
                sequenceClone = copy.deepcopy(sequence)
                sequenceClone[i].pop(j)
                result.append(sequenceClone)
    return result


"""
Prunes the set of candidates generated for size k given all frequent sequence of level (k-1), as done in AprioriAll
"""
def pruneCandidates(candidatesLastLevel, candidatesGenerated):
    return [cand for cand in candidatesGenerated if
            all(x in candidatesLastLevel for x in generateDirectSubsequences(cand))]


# ### Put it all together:
"""
The AprioriAll algorithm. Computes the frequent sequences in a seqeunce dataset for a given minSupport

Args:
    dataset: A list of sequences, for which the frequent (sub-)sequences are computed
    minSupport: The minimum support that makes a sequence frequent
    verbose: If true, additional information on the mining process is printed (i.e., candidates on each level)
Returns:
    A list of tuples (s, c), where s is a frequent sequence, and c is the count for that sequence
"""
def apriori(dataset, minSupport, verbose=False):
    start = time.time()
    global numberOfCountingOperations
    numberOfCountingOperations = 0
    Overall = []
    itemsInDataset = sorted(set([item for sublist1 in dataset for sublist2 in sublist1 for item in sublist2])) # is necessary? 
    singleItemSequences = [[[item]] for item in itemsInDataset] # creates starting singleton
    singleItemCounts = []
    for item in singleItemSequences:
        x = countSupport(dataset, item)
        if x >= minSupport:
            singleItemCounts.append((item, x))
    Overall.append(singleItemCounts)
    if verbose:
        print("Result, lvl 1: " + str(Overall[0]))
    k = 1
    while True:
        if not Overall[k - 1]:
            break
        # 1. Candidate generation
        candidatesLastLevel = [x[0] for x in Overall[k - 1]]
        candidatesGenerated = generateCandidates(candidatesLastLevel)
        if verbose:
            print("Candidates generated, lvl ", k + 1)
        # 2. Candidate pruning (using a "containsall" subsequences)
        candidatesPruned = [cand for cand in candidatesGenerated if all(x in candidatesLastLevel for x in generateDirectSubsequences(cand))]
        if verbose:
            print("Candidates pruned, lvl ", k + 1)
        # 3. Candidate checking
        candidatesCounts = []
        tot = len(candidatesPruned)
        for i in trange(tot):
            candidatesCounts.append((candidatesPruned[i], countSupport(dataset, candidatesPruned[i])))
        resultLvl = [(i, count) for (i, count) in candidatesCounts if (count >= minSupport)]
        if verbose:
            print("Result, lvl ", k + 1)
        Overall.append(resultLvl)
        k = k + 1
    # "flatten" Overall
    Overall = Overall[:-1]
    Overall = [item for sublist in Overall for item in sublist]
    end = time.time()
    print("Total Time", end-start)
    return Overall

if __name__ == "__main__":
    with open('5.sequences_of_poits.text') as f:
        content = f.readlines()
    content = [ [ [event] for event in x.strip().split()[:-1] ] for x in content]

    print("Number of input sequences: ", len(content))
    print("Total number of events: ", sum([len(seq) for seq in content])) 
    res3 = apriori(content, 30, verbose=True)
    print(len(res3))
    print(res3)
                                             
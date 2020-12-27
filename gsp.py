# Univerisity of Pisa
# Data Mining (AY 2020/2021)
# Professor: Anna Monreale
# Students: Elia Piccoli, Nicola Gugole, Alex Pasquali

import copy
import time
from tqdm import trange

def optCountSupport(dataset, candidateSequence, min_threshold, minGap, maxGap, maxSpan, use_time_constraints):
    """Optimized computation for the support of a sequence in a dataset.

    It will check until one of the two contidion is true:
        (1) The support is greater than min_threshold
        (2) There aren't enough elements to satify min_threshold

    Parameters
    ----------
    min_threshold : int
        minimum support

    minGap : int 
        minimum gap between element in pattern (in days)

    maxGap : int
        maximum gap between element in pattern (in days)

    maxSpan : int
        maximum span of pattern (in days)

    use_time_constraints : bool
        True = use time constraint

    Returns
    ----------
        The approximated support of the sequence
    """

    total = 0
    len_dt = len(dataset)
    for seq in dataset:
        len_dt -= 1
        if isSubsequence(seq, candidateSequence, minGap, maxGap, maxSpan, use_time_constraints):
            total += 1
        if total > min_threshold or total + len_dt < min_threshold:
            return total
    return total

def countSupport_Customers(dataset, candidateSequence, min_threshold, minGap, maxGap, maxSpan, use_time_constraints):
    """Complete computation for the support and list of customers of a sequence in a dataset

    Parameters
    ----------
    min_threshold : int
        minimum support

    minGap : int 
        minimum gap between element in pattern (in days)

    maxGap : int
        maximum gap between element in pattern (in days)

    maxSpan : int
        maximum span of pattern (in days)

    use_time_constraints : bool
        True = use time constraint

    Return
    ----------
    tuple:
        0 : The exact support of the sequence

        1 : List of index of customer that contains the pattern
    """

    total = 0
    customers = []
    for i in range(len(dataset)):
        if isSubsequence(dataset[i], candidateSequence, minGap, maxGap, maxSpan, use_time_constraints):
            customers.append(i)
            total += 1
    return total, customers

def isSubsequence(mainSequence, subSequence, minGap, maxGap, maxSpan, use_time_constraints):
    subSequenceClone = list(subSequence)  # clone the sequence, because we will alter it
    return isSubsequenceIterative(mainSequence, subSequenceClone, minGap, maxGap, maxSpan, use_time_constraints)

# Through out attempts it resulted that the recursive approach is (in this case) faster than the iterative
# but the space complexity is prohibitive for non-small datasets, hence we use the iterative approach

# def isSubsequenceRecursive(mainSequence, subSequenceClone, start=0):
#     # Check if empty: End of recursion, all itemsets have been found
#     if (not subSequenceClone):
#         return True
#     # retrieves element of the subsequence and removes is from subsequence
#     firstElem = set(subSequenceClone.pop(0))
#     # Search for the first itemset...
#     for i in range(start, len(mainSequence)):
#         if (set(mainSequence[i]).issuperset(firstElem)):
#             # and recurse
#             return isSubsequenceRecursive(mainSequence, subSequenceClone, i + 1)
#     return False

# This is the main part of the algorithm, here is where the majority of the computation is focused
# The complexity is O(n^3) it can be reduced to O(n^2*log(n)) if we can use binary search instead of issuperset
def isSubsequenceIterative(mainSequence, subSequenceClone, minGap, maxGap, maxSpan, use_time_constraints):
    start = 0
    lastDate = None
    firstDate = 0
    while subSequenceClone:
        found = False
        nextElem = subSequenceClone.pop(0)
        for i in range(start, len(mainSequence)):
            if (set(mainSequence[i][0]).issuperset(nextElem)):
                if use_time_constraints:
                    if lastDate is None:
                        firstDate = mainSequence[i][1]
                    else:
                        delta = (mainSequence[i][1] - lastDate).days
                        if delta > maxGap or delta < minGap or (mainSequence[i][1] - firstDate).days > maxSpan:
                            return False
                    lastDate = mainSequence[i][1]
                found = True
                start = i+1
                break
        if not found:
            return False
    return True

def generateCandidatesForPair(cand1, cand2):
    """ Generates one candidate of size k from two candidates of size (k-1) as used in the apriori algorithm
    """
    
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

def generateCandidates(lastLevelCandidates):
    """ Generates the set of candidates of size k from the set of frequent sequences with size (k-1)
    """

    k = sum(len(i) for i in lastLevelCandidates[0]) + 1
    if k == 2:
        flatShortCandidates = [item for sublist2 in lastLevelCandidates for sublist1 in sublist2 for item in sublist1]
        result = [[[a, b]] for a in flatShortCandidates for b in flatShortCandidates if b > a]
        result.extend([[[a], [b]] for a in flatShortCandidates for b in flatShortCandidates])
        return result
    else:
        candidates = []
        for i in range(0, len(lastLevelCandidates)):
            for j in range(0, len(lastLevelCandidates)):
                newCand = generateCandidatesForPair(lastLevelCandidates[i], lastLevelCandidates[j])
                if not newCand == []:
                    candidates.append(newCand)
        candidates.sort()
        return candidates

def generateDirectSubsequences(sequence):
    """Computes all direct subsequence for a given sequence.

    A direct subsequence is any sequence that originates from deleting exactly one item from any element in the original sequence.
    """

    result = []
    for i, itemset in enumerate(sequence):
        if len(itemset) == 1:
            sequenceClone = copy.deepcopy(sequence)
            sequenceClone.pop(i)
            result.append(sequenceClone)
        else:
            for j in range(len(itemset)):
                sequenceClone = copy.deepcopy(sequence)
                sequenceClone[i].pop(j)
                result.append(sequenceClone)
    return result

# minGap, maxGap and maxSpan are all in days
def optApriori(dataset, minSupport, minGap=0, maxGap=15, maxSpan=60, use_time_constraints=False, verbose=False):
    """Optimized apriori algorithm.
        
    Computes the frequent sequences in a sequence dataset for a given minSupport.

    Parameters
    ----------
    dataset : list of list
        list of sequences, for which the frequent (sub-)sequences are computed
        
    minSupport : int
        minimum support

    minGap : int 
        minimum gap between element in pattern (in days)

    maxGap : int
        maximum gap between element in pattern (in days)

    maxSpan : int
        maximum span of pattern (in days)

    use_time_constraints : bool
        True = use time constraint

    verbose : bool
        True prints the state of the computation of different steps

    Return
    ----------
    (s, c) : list of tuples 

            0 : frequent sequence
        
            1 : approximated count for that sequence (due to optimization)
    """

    start = time.time()
    Overall = []
    itemsInDataset = sorted(set([item for sublist1 in dataset for sublist2 in sublist1 for item in sublist2[0]]))
    singleItemSequences = [[[item]] for item in itemsInDataset]
    singleItemCounts = []
    for i in trange(len(singleItemSequences), desc=f"Level: {1}"):
        x = optCountSupport(dataset, singleItemSequences[i], minSupport, minGap, maxGap, maxSpan, use_time_constraints)
        if x >= minSupport:
            singleItemCounts.append((singleItemSequences[i], x))
    Overall.append(singleItemCounts)
    if verbose:
        print("Result, lvl 1")
    k = 1
    while True:
        if not Overall[k - 1]:
            break
        # 1. Candidate generation
        candidatesLastLevel = [x[0] for x in Overall[k - 1]]
        candidatesGenerated = generateCandidates(candidatesLastLevel)
        if verbose:
            print("Candidates generated, lvl ", k + 1)
        # 2. Candidate pruning
        candidatesPruned = []
        for cand in candidatesGenerated:
            check = True
            r = generateDirectSubsequences(cand)
            for x in r:
                check = check and x in candidatesLastLevel
                if not check:
                    break
            if check:
                candidatesPruned.append(cand)
        # on average 0-10 sec faster on our dataset (keep it ?)
        # candidatesPruned = [cand for cand in candidatesGenerated if all(x in candidatesLastLevel for x in generateDirectSubsequences(cand))]
        if verbose:
            print("Candidates pruned, lvl ", k + 1)
        # 3. Candidate checking
        candidatesCounts = []
        tot = len(candidatesPruned)
        for i in trange(tot, desc=f"Level: {k+1}"):
            supp = optCountSupport(dataset, candidatesPruned[i], minSupport, minGap, maxGap, maxSpan, use_time_constraints)
            if supp >= minSupport:
                candidatesCounts.append((candidatesPruned[i], supp))
        resultLvl = candidatesCounts
        if verbose:
            print("Result, lvl ", k + 1)
        Overall.append(resultLvl)
        k = k + 1
    # Creates a list of all the results from different levels
    Overall = Overall[:-1]
    Overall = [item for sublist in Overall for item in sublist]
    end = time.time()
    print("Total Time: ", end-start)
    return Overall

def apriori(dataset, minSupport, minGap=0, maxGap=15, maxSpan=60, use_time_constraints=False, sequences=None):
    """Apriori algorithm.
        
    Computes the frequent sequences in a sequence dataset for a given minSupport.
    This version returns the true values of support and list of index of customers

    Parameters
    ----------
    dataset : list of list
        list of sequences, for which the frequent (sub-)sequences are computed
        
    minSupport : int
        minimum support

    minGap : int 
        minimum gap between element in pattern (in days)

    maxGap : int
        maximum gap between element in pattern (in days)

    maxSpan : int
        maximum span of pattern (in days)

    use_time_constraints : bool
        True = use time constraint

    sequences : list of tuple
        result structure returned by optApriori function

    Return
    ----------
    (s, c, l) : list of tuples 

        0 : frequent sequence
    
        1 : count for that sequence
    
        2 : list of customers' index that contains the sequence 
    """ 

    # to reduce time it computes the result using the optimized version and the creates the complete output on a smaller set of patterns
    if sequences is None:
        seq_list = optApriori(dataset, minSupport, minGap=minGap, maxGap=maxGap, maxSpan=maxSpan, use_time_constraints=use_time_constraints)
    else:
        seq_list = sequences
    sequences = [elem[0] for elem in seq_list]
    true_seq_list = []
    for i in range(len(sequences)):
        support,customer_indexes = countSupport_Customers(list(dataset.values()), sequences[i], minSupport, minGap, maxGap, maxSpan, use_time_constraints)
        true_seq_list.append((sequences[i], support, customer_indexes))
    return true_seq_list
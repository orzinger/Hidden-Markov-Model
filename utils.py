import re
import numpy as np



text_details = {}

_tagSequences = {}

_wordsTagSequences = {}



def read_QMle(file):

    with open(file, 'r') as file:

        for line in file.readlines():
            sequence = line.rstrip().split('\t')
            _tagSequences[sequence[0]] = int(sequence[1])
        
        text_details['num of tags'] = _tagSequences.pop('NUM_TAGS')


def read_EMle(file):

    with open(file, 'r') as file:

        for line in file.readlines():

            sequence = line.rstrip().split('\t')
            _wordsTagSequences[sequence[0]] = int(sequence[1])

        text_details['num of words'] = _wordsTagSequences.pop('NUM_WORDS')





def interpulations(t1,t2,t3):

    trigram = "{} {} {}".format(t1, t2, t3)
    bigram = "{} {}".format(t2, t3)
    unigram = "{}".format(t3)

    try:
        lamda3 = (_tagSequences[trigram] - 1) / (_tagSequences["{} {}".format(t1, t2)] - 1)
    except:
        lamda3 = 0
    try:
        lamda2 = (_tagSequences[bigram] - 1) / (_tagSequences["{}".format(t2)] - 1)
    except:
        lamda2 = 0
    lamda1 = (_tagSequences[unigram] - 1) / (text_details['num of tags'] - 1)

    lamda_sum = np.sum([lamda3, lamda2, lamda1])

    return [lamda3, lamda2, lamda1] / lamda_sum



def find_regex(word):
    
    if re.search(r'^[0-9]{2}$', word) is not None:
        word = '^TWODIGITS'
    elif re.search(r'^[0-9]{4}$', word) is not None:
        word = '^FOURDIGITS'
    elif re.search('^[0-9]{3}$|^[0-9]{5,}$', word) is not None:
        word = '^DIGITS'
    elif re.search(r'^\d+\,\d+\.\d+$', word) is not None:
        word = '^containsDigitAndComma'
    elif re.search(r'^\d+\.\d+$', word) is not None:
        word = '^containsDigitAndPeriod'
    elif re.search(r'^(?=.*[a-zA-Z])(?=.*[0-9])', word) is not None:
        word = '^containsDigitAndAlpha'
    elif re.search(r'^(\d{2}-\d{2})$|^(\d{1,2}\/\d{1,2}\/\d{2,4})$', word) is not None:
        word = '^containsDigitAndSlash'
    elif re.search(r'^[A-Z]{1,2}\.$|^[A-Z][a-z]{1,2}\.$', word) is not None:
        word = '^capPeriod'
    elif re.search(r'^[A-Z]\w+$', word) is not None:
        word = '^initCap'
    elif re.search(r'^[A-Z]+', word) is not None:
        word = '^allCaps'
    elif re.search(r'^\w+-\w+$', word) is not None:
        word = '^portmanteau'
    else:
        word = '^UNK'
    return word


def getQ(t1,t2,t3):

    trigram = "{} {} {}".format(t1, t2, t3)
    bigram = "{} {}".format(t2, t3)
    unigram = "{}".format(t3)

    lamda3, lamda2, lamda1 = 0.9, 0.09, 0.01
    # lamda3, lamda2, lamda1 = interpulations(t1, t2, t3)
    try:
        q3 = _tagSequences[trigram] / _tagSequences["{} {}".format(t1, t2)]
    except:
        q3 = 0
    try:
        q2 = _tagSequences[bigram] / _tagSequences["{}".format(t2)]
    except:
        q2 = 0
    q1 = _tagSequences[unigram] / text_details['num of tags']

    return lamda3*q3 + lamda2*q2 + lamda1*q1


    
def getE(w, t):

    emission = "{} {}".format(w, t)
    # try:
    e = _wordsTagSequences[emission] / _tagSequences["{}".format(t)]
    # except:
    #   word = find_regex(w)
    #   emission = "{} {}".format(word, t)
    #   e = words[emission] / tags["{}".format(t)]
    return e
import numpy as np
import re
import sys
from collections import defaultdict, Counter 

text_details = {}

def read_text(file):
    with open(file,'r') as fhnadle:
        text = fhnadle.readlines()
        return text


def analysis_text(text):

    dict_TagSequences = Counter() # tagsdic
    uniqueTagsList = set() # tags_set
    dict_WordsTagSequences = Counter() # words_tagsdic
    dict_WordsCounts = Counter() # words_dic
    dict_WordsWithTags = defaultdict(set)
    _num_of_tags = _num_of_words = 0
    for _line in text:
        _line = _line.rstrip().split(" ") # split line to [word/tag] list i.e: [DOG/NN]
        _tagsLine = [s.rsplit("/",1)[1] for s in _line] # extract list of tags per line i.e: [NN, NNP, VBN, ...]
        _num_of_tags += len(_tagsLine)
        _wordsLine = [s.rsplit("/",1) for s in _line] # extract list of [word, tag] per line i.e: [DOG, NN]
        _num_of_words += len(_wordsLine)
        uniqueTagsList.update(_tagsLine) # update unique set of tags
        _tagsLine = ['<s>','<s>'] + _tagsLine # add <start> <start> to tags list per line
        dict_TagSequences.update(_tagsLine)
        _tagsLine = list(zip(_tagsLine,_tagsLine[1:],_tagsLine[2:]))
        dict_TagSequences.update([' '.join(s) for s in _tagsLine])
        dict_TagSequences.update([' '.join(s) for s in [(trigram[0], trigram[1]) for trigram in _tagsLine]])
        dict_WordsCounts.update([s[0] for s in _wordsLine])
        dict_WordsTagSequences.update([' '.join(s) for s in _wordsLine])
        for w in _wordsLine:
            dict_WordsWithTags[w[0]].add(w[1])
    
    # dict_TagSequences['<s>'] = dict_TagSequences['<s> <s>']
    
    text_details['num of words'], text_details['num of tags'] = _num_of_words, _num_of_tags

    return dict_WordsCounts, dict_WordsWithTags, dict_WordsTagSequences, dict_TagSequences

def analysis_frequencies_of_words(listOfTextDictionaries):

    dict_WordsCounts, dict_WordsWithTags, dict_WordsTagSequences = listOfTextDictionaries
    dictLowFreqWords = dict(filter(lambda s: s[1]<=4 ,dict_WordsCounts.items())) # rare words ... lowfreqwords
    dictHighFreqWords = dict(filter(lambda s: s[1]>4 ,dict_WordsCounts.items())) # common words ... highfreqwords


    dict_rareWordsTagsSequences = Counter()
    dict_commonWordsTagsSequences = Counter()


    for key in dictLowFreqWords.keys():
        word = key
        if re.search(r'^[0-9]{2}$', word) is not None:
            word = '^TWODIGITS'
        elif re.search(r'^[0-9]{4}$', word) is not None:
            word = '^FOURDIGITS'
        elif re.search(r'^[0-9]{3}$|^[0-9]{5,}$', word) is not None:
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
        for t in dict_WordsWithTags[key]:
            dict_rareWordsTagsSequences[word+" "+t] += dict_WordsTagSequences[key+" "+t]

    
    for key in dictHighFreqWords.keys():
        for t in dict_WordsWithTags[key]:
            dict_commonWordsTagsSequences[key+" "+t] += dict_WordsTagSequences[key+" "+t]
    
    dict_commonWordsTagsSequences.update(dict_rareWordsTagsSequences)

    dict_WordsTagSequences = dict_commonWordsTagsSequences

    dict_WordsWithTags = defaultdict(set)

    for w in dict_WordsTagSequences.keys():
        w =  w.split(' ')
        dict_WordsWithTags[w[0]].add(w[1])
    
    return dict_WordsTagSequences, dict_WordsWithTags



def write_QMle_toFile(file, _tagSequences):

    with open(file, 'w') as file:

        for key in _tagSequences.keys():
            file.write("{}\t{}\n".format(str(key), _tagSequences[key]))
        file.write("NUM_TAGS\t{}".format(text_details['num of tags']))

def write_EMle_toFile(file, _wordsTagSequences):

    with open(file, 'w') as file:

        for key in _wordsTagSequences.keys():
            file.write("{}\t{}\n".format(str(key), _wordsTagSequences[key]))
        file.write("NUM_WORDS\t{}".format(text_details['num of words']))

def write_ExtrFile(_wordsWithTags, extra_file = None):

    if extra_file is None:
        extra_file = "extra_file.txt"
    with open(extra_file, 'w') as file:

        for key in _wordsWithTags.keys():
            tags = ' '.join(list(_wordsWithTags[key]))
            file.write("{}\t{}\n".format(str(key), tags))

    return extra_file

if __name__ == "__main__":

    print("Start train model")

    text = read_text(sys.argv[1])

    _wordsCounts, _wordsWithTags, _wordsTagSequences, _tagSequences = analysis_text(text)

    _wordsTagSequences,  _wordsWithTags = analysis_frequencies_of_words([_wordsCounts, _wordsWithTags, _wordsTagSequences])

    write_QMle_toFile(sys.argv[2], _tagSequences)

    write_EMle_toFile(sys.argv[3], _wordsTagSequences)

    extra_file = None
    
    if len(sys.argv)>4:
    	extra_file = sys.argv[4]

    extra_file = write_ExtrFile(_wordsWithTags, extra_file)
        

    print("Successfully created '{}', '{}' and {}".format(sys.argv[2], sys.argv[3], extra_file))





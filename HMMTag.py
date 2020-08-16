from collections import deque, defaultdict
import numpy as np
import utils as ut
import re
import sys


def getScore(word, tag, prev_tag, prev_prev_tag):
  return np.log(ut.getQ(prev_prev_tag, prev_tag, tag)) + np.log(ut.getE(word, tag))


class HMM:

    def __init__(self, extarfile):
        self._wordWithTags = {}
        for line in extarfile:
            sequence = line.rstrip().split('\t')
            self._wordWithTags[sequence[0]] = (sequence[1]).split(' ')

    def vitrebi(self, sentence, stopword):
      
      V = [defaultdict(dict)]
      B = [defaultdict(dict)]

      tags = deque([['<s>'],['<s>']])

      V[0]['<s>']['<s>'] = 1

      
      for i in range(1,len(sentence) + 1):
        word = sentence[i-1]
        if word not in self._wordWithTags.keys():
          word = ut.find_regex(word)
        tags.append(self._wordWithTags[word])
        V.append(defaultdict(dict))
        B.append(defaultdict(dict))
        for v in tags[i+1]:
          for u in tags[i]:
            V[i][v][u] = np.max([V[i-1][u][w] + getScore(word, v, u, w) for w in tags[i-1]])
            B[i][v][u] = tags[i-1][np.argmax([V[i-1][u][w] + getScore(word, v, u, w) for w in tags[i-1]])]
      
      
      
      maxlist = []
      tagsmaxlist = []
      backSequenceOfTags = deque()

      
      
      for v in tags[i+1]:
        for u in tags[i]:
          maxlist.append(V[i][v][u] + ut.getQ(u, v, stopword))
          tagsmaxlist.append((u,v))
      u, v = tagsmaxlist[np.argmax(maxlist)]
      backSequenceOfTags.appendleft(v)
      if u is not "<s>":
        backSequenceOfTags.appendleft(u)
      

      
      for k in range(len(sentence) - 2, 0, -1):
        backSequenceOfTags.appendleft(B[k + 2][backSequenceOfTags[1]][backSequenceOfTags[0]])

      return backSequenceOfTags






if __name__ == "__main__":
    
    with open(sys.argv[1], 'r') as file:
        dev = file.readlines()
    
    with open(sys.argv[5], 'r') as file:
        extrafile = file.readlines()

    ut.read_QMle(sys.argv[2])

    ut.read_EMle(sys.argv[3])

    hmm = HMM(extrafile)

    good = 0.

    length = 0

    
    print("Loop over test lines\nRun 'viterbi' model on it")

    with open(sys.argv[4],'w') as file:
      for line in dev:
        line = line.rstrip().split(" ")
        words_test = [s.rsplit('/',1)[0] for s in line]
        pred_tags = hmm.vitrebi(words_test, '.')
        #tags_test = [s.rsplit('/',1)[1] for s in line]
        #good += sum([i == j for i,j in zip(tags_test, pred_tags)])
        #length += len(tags_test)
        copy = ' '.join([pair[0]+"/"+pair[1] for pair in zip(words_test, pred_tags)])
        file.write(copy+"\n")

    
    if length != 0:

      print("accuracy: {0:.2f}".format(good*100/length))

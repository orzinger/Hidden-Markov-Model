from collections import deque
import numpy as np
import utils as ut
import re
import sys






class HMM:

    def __init__(self, extarfile):
        self._wordsList = {}
        for line in extarfile:
            sequence = line.rstrip().split('\t')
            self._wordsList[sequence[0]] = (sequence[1]).split(' ')
    

    
    def greedy(self, sentence):
    
        predict_tags = deque(['<s>','<s>'])
        for i in range(len(sentence)):
            word = sentence[i]
            scores = []
            if word not in self._wordsList.keys():
                    word = ut.find_regex(word)
            for tag in self._wordsList[word]:
                p = ut.getQ(predict_tags[i],predict_tags[i+1],tag)
                e = ut.getE(word, tag)
                scores.append(np.log(p) + np.log(e))
            tag_max = np.argmax(scores)
            predict_tags.append(self._wordsList[word][tag_max])

        predict_tags.popleft()
        predict_tags.popleft()


        return predict_tags





if __name__ == "__main__":

    
    with open(sys.argv[1], 'r') as file:
        dev = file.readlines()
    
    with open(sys.argv[5], 'r') as file:
        extrafile = file.readlines()

    ut.read_QMle(sys.argv[2])

    ut.read_EMle(sys.argv[3])

    gr = HMM(extrafile)

    good = 0.

    length = 0



    print("Loop over test lines\nRun 'greedy' model on it")

    with open(sys.argv[4],'w') as file:
        for line in dev:
            line = line.rstrip().split(" ")
            words_test = [s.rsplit('/',1)[0] for s in line]
            pred_tags = gr.greedy(words_test)
            tags_test = [s.rsplit('/',1)[1] for s in line]
            good += sum([i == j for i,j in zip(tags_test, pred_tags)])
            length += len(tags_test)
            copy = ' '.join([pair[0]+"/"+pair[1] for pair in zip(words_test, pred_tags)])
            file.write(copy+"\n")
    if length != 0:
      print("accuracy: {0:.2f}".format(good*100/length))

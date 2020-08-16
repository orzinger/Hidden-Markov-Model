import argparse
import json
import os
from collections import Counter
from collections import defaultdict

RARE_WORD = 5
EXTRA_FILE = 'extra_file.txt'


def get_words_and_tags(file_name):
    with open(file_name, 'r') as file:
        return [word.rsplit('/', 1) for line in file for word in line.split()]


def get_sentence_words_and_tags(line):
    return [word.rsplit('/', 1) for word in line.split()]


def get_words_counter(words_and_tags):
    return Counter([pair[0] for pair in words_and_tags])


def get_rare_words(words_and_tags):
    words_counter = get_words_counter(words_and_tags)

    return set([word for word, value in words_counter.items() if value < RARE_WORD])


def contains_numbers(string):
    return any(char.isdigit() for char in string)


def contains_uppercase(string):
    return any(char.isupper() for char in string)


def build_features_vector(word_index, words_and_tags, rare_words, words_history):
    current_word, current_tag = words_and_tags[word_index]

    if current_word not in rare_words:
        words_history[current_word].add(current_tag)
    else:
        words_history['UNK'].add(current_tag)

    features_vector = current_tag

    if word_index - 1 >= 0:
        pre_word, pre_tag = words_and_tags[word_index - 1]
    else:
        pre_word, pre_tag = 'START', 'START'

    features_vector += ' pre_word={} '.format(pre_word)
    features_vector += 'pre_tag={} '.format(pre_tag)

    if word_index - 2 >= 0:
        d_pre_word, d_pre_tag = words_and_tags[word_index - 2]
        d_pre_tag += '_' + pre_tag
    else:
        d_pre_word, d_pre_tag = 'START', 'START_' + pre_tag

    features_vector += 'd_pre_word={} '.format(d_pre_word)
    features_vector += 'd_pre_tag={} '.format(d_pre_tag)

    if word_index + 1 <= len(words_and_tags) - 1:
        next_word = words_and_tags[word_index + 1][0]
    else:
        next_word = 'END'

    features_vector += 'next_word={} '.format(next_word)

    if word_index + 2 <= len(words_and_tags) - 1:
        d_next_word = words_and_tags[word_index + 2][0]
    else:
        d_next_word = 'END'

    features_vector += 'd_next_word={} '.format(d_next_word)

    if current_word in rare_words:
        for i in range(min(len(current_word), 4)):
            features_vector += 'prefix{i}={prefix} '.format(i=i + 1, prefix=current_word[:i + 1])
            features_vector += 'suffix{i}={suffix} '.format(i=i + 1, suffix=current_word[-1 - i:])

        if contains_numbers(current_word):
            features_vector += 'contains_number={} '.format(True)

        if contains_uppercase(current_word):
            features_vector += 'contains_upper={} '.format(True)

        if '-' in current_word:
            features_vector += 'contains_hyphen={} '.format(True)

        features_vector += 'current_word={}'.format('UNK')
    else:
        features_vector += 'current_word={}'.format(current_word)

    return features_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', metavar='N', type=str, nargs='+')
    args = parser.parse_args()
    corpus_file, features_file = args.parameters

    words_and_tags = get_words_and_tags(corpus_file)
    rare_words = get_rare_words(words_and_tags)

    words_history = defaultdict(set)
    with open(corpus_file, 'r') as corpus:
        with open(features_file, 'w+') as features:
            for line in corpus:
                sentence_words_and_tags = get_sentence_words_and_tags(line)
                for i in range(len(sentence_words_and_tags)):
                    features.write(build_features_vector(word_index=i,
                                                         words_and_tags=sentence_words_and_tags,
                                                         rare_words=rare_words,
                                                         words_history=words_history) + '\n')

    words_history_list = defaultdict(list)

    for word, tags in words_history.items():
        for tag in tags:
            words_history_list[word].append(tag)

    with open(EXTRA_FILE, 'w') as ex:
        ex.write(json.dumps(words_history_list))

    os.system('cat {features_file} | head -100 > {features_file}_partial'.format(features_file=features_file))
    os.system('cat {features_file} | tail -100 >> {features_file}_partial'.format(features_file=features_file))

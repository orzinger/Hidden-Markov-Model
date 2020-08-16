import argparse
import json
import pickle
from collections import Counter

RARE_WORD = 5


def get_known_words_and_tags():
    with open('extra_file.txt') as ex:
        known_words_and_tags = json.load(ex)

        return known_words_and_tags, set(known_words_and_tags.keys())


def get_rare_words(words):
    counter = Counter(words)

    return set([word for word, value in counter.items() if value < RARE_WORD])


def contains_numbers(string):
    return any(char.isdigit() for char in string)


def contains_uppercase(string):
    return any(char.isupper() for char in string)


def build_features_vectors(word_index, words, known_words, known_words_and_tags):
    current_word = words[word_index]
    features_vector = ''

    pre_tags = []
    d_pre_tags = []

    if word_index - 1 >= 0:
        pre_word = words[word_index - 1]
        if pre_word in known_words:
            pre_tags.extend(known_words_and_tags[pre_word])
        else:
            pre_tags.extend(known_words_and_tags['UNK'])

    else:
        pre_word = 'START'
        pre_tags.append('START')

    features_vector += 'pre_word={} '.format(pre_word)

    if word_index - 2 >= 0:
        d_pre_word = words[word_index - 2]

        if d_pre_word in known_words:
            for d_known_tag in known_words_and_tags[d_pre_word]:
                for pre_tag in pre_tags:
                    d_pre_tags.append(d_known_tag + '_' + pre_tag)
        else:
            for tag in known_words_and_tags['UNK']:
                for pre_tag in pre_tags:
                    d_pre_tags.append(tag + '_' + pre_tag)
    else:
        d_pre_word = 'START'
        for pre_tag in pre_tags:
            d_pre_tags.append('START' + '_' + pre_tag)

    features_vector += 'd_pre_word={} '.format(d_pre_word)

    if word_index + 1 <= len(words) - 1:
        next_word = words[word_index + 1]
    else:
        next_word = 'END'

    features_vector += 'next_word={} '.format(next_word)

    if word_index + 2 <= len(words) - 1:
        d_next_word = words[word_index + 2]
    else:
        d_next_word = 'END'

    features_vector += 'd_next_word={} '.format(d_next_word)

    if current_word not in known_words:
        for i in range(min(len(current_word), 4)):
            features_vector += 'prefix{i}={prefix} '.format(i=i + 1, prefix=current_word[:i + 1])
            features_vector += 'suffix{i}={suffix} '.format(i=i + 1, suffix=current_word[-1 - i:])

        if contains_numbers(current_word):
            features_vector += 'contains_number={} '.format(True)

        if contains_uppercase(current_word):
            features_vector += 'contains_upper={} '.format(True)

        if '-' in current_word:
            features_vector += 'contains_hyphen={} '.format(True)

        features_vector += 'current_word={} '.format('UNK')
    else:
        features_vector += 'current_word={} '.format(current_word)

    features_vectors = []

    for d_pre_tag in d_pre_tags:
        features_vectors.append(
            features_vector +
            'pre_tag={} '.format(d_pre_tag.split('_')[1]) +
            'd_pre_tag={}'.format(d_pre_tag))

    return features_vectors


def calculate_features_discrete_vector(feature_vectors, vec):
    features_dicts = []

    for feature_vector in feature_vectors:
        features_dict = {}
        features = feature_vector.split()

        if features:
            for feature in features:
                loc = feature.find('=')
                rule = feature[:loc]
                value = feature[loc + 1:]

                features_dict[rule] = value

        features_dicts.append(features_dict)

    return vec.transform(features_dicts)


def calculate_tag(feature_vectors, logreg_model, vec_model):
    discrete_vectors = calculate_features_discrete_vector(feature_vectors=feature_vectors, vec=vec_model)
    viterbi_tags = logreg_model.predict(discrete_vectors)

    return Counter(viterbi_tags).most_common()[0][0]


def get_words(input_file):
    with open(input_file, 'r') as input:
        return [word for line in input for word in line.split()]


def get_sentence_words(line):
    return [word for word in line.split()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', metavar='N', type=str, nargs='+')
    args = parser.parse_args()
    input_file_name, model_file_name, feature_map_file, output_file = args.parameters

    vec_model = pickle.load(open('vec.sav', 'rb'))
    logreg_model = pickle.load(open(model_file_name, 'rb'))
    all_tags = logreg_model.classes_.tolist()

    words = get_words(input_file_name)
    known_words_and_tags, known_words = get_known_words_and_tags()

    with open(input_file_name, 'r') as input:
        with open(output_file, 'w') as output:
            for line in input:
                sentence_words = get_sentence_words(line)
                for i, word in enumerate(sentence_words):
                    viterbi_vectors = build_features_vectors(i,
                                                             sentence_words,
                                                             known_words=known_words,
                                                             known_words_and_tags=known_words_and_tags)
                    viterbi_tag = calculate_tag(feature_vectors=viterbi_vectors,
                                                logreg_model=logreg_model,
                                                vec_model=vec_model)
                    output.write('{word}/{tag} '.format(word=word, tag=viterbi_tag))
                output.write('\n')

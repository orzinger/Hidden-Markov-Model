import argparse
import json
import pickle

RARE_WORD = 5


def get_known_words():
    with open('extra_file.txt') as ex:
        return set(json.load(ex).keys())


def contains_numbers(string):
    return any(char.isdigit() for char in string)


def contains_uppercase(string):
    return any(char.isupper() for char in string)


def build_features_vector(word_index, words, last_tags, known_words):
    current_word = words[word_index]

    features_vector = ''

    if word_index - 1 >= 0:
        pre_word, pre_tag = words[word_index - 1], last_tags[1]
    else:
        pre_word, pre_tag = 'START', 'START'

    features_vector += 'pre_word={} '.format(pre_word)
    features_vector += ' pre_tag={} '.format(pre_tag)

    if word_index - 2 >= 0:
        d_pre_word, d_pre_tag = words[word_index - 2], last_tags[0]
        d_pre_tag += '_' + last_tags[1]
    else:
        d_pre_word, d_pre_tag = 'START', 'START_' + last_tags[1]

    features_vector += 'd_pre_word={} '.format(d_pre_word)
    features_vector += 'd_pre_tag={} '.format(d_pre_tag)

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
        features_vector += 'current_word={}'.format('UNK')
    else:
        features_vector += 'current_word={}'.format(current_word)

    return features_vector


def calculate_features_discrete_vector(feature_vector, vec):
    features_dict = {}
    features = feature_vector.split()

    if features:
        for feature in features:
            loc = feature.find('=')
            rule = feature[:loc]
            value = feature[loc + 1:]

            features_dict[rule] = value

    return vec.transform(features_dict)


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

    words = get_words(input_file_name)
    known_words = get_known_words()

    with open(input_file_name, 'r') as input:
        with open(output_file, 'w') as output:
            for line in input:
                last_tags = 'START', 'START'
                sentence_words = get_sentence_words(line)
                for i, word in enumerate(sentence_words):
                    feature_vector = build_features_vector(word_index=i,
                                                           words=sentence_words,
                                                           last_tags=last_tags,
                                                           known_words=known_words)
                    discrete_feature_vector = calculate_features_discrete_vector(
                        feature_vector=feature_vector, vec=vec_model)

                    tag = logreg_model.predict(discrete_feature_vector)
                    last_tags = last_tags[1], tag[0]

                    output.write('{word}/{tag} '.format(word=word, tag=tag[0]))
                output.write('\n')

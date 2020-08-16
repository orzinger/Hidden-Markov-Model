import argparse
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def convert_features_to_train_data(features_file):
    features_dicts = []
    tags = []

    with open(features_file, 'r') as features_file:
        for line in features_file:
            features = line.split()
            if features:
                features_dict = {}
                tags.append(features[0])

                for feature in features[1:]:
                    loc = feature.find('=')
                    rule = feature[:loc]
                    value = feature[loc + 1:]

                    features_dict[rule] = value

            features_dicts.append(features_dict)

    features_vectors = convert_features_map_to_vectorized_form(features_dicts)

    return tags, features_vectors


def convert_features_map_to_vectorized_form(features_map):
    vec = DictVectorizer()
    pos_vectorized = vec.fit_transform(features_map)

    save_model(vec, 'vec.sav')

    return pos_vectorized


def save_model(model, model_file):
    pickle.dump(model, open(model_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', metavar='N', type=str, nargs='+')
    args = parser.parse_args()
    features_file, model_file = args.parameters

    labels, data = convert_features_to_train_data(features_file)

    logreg = LogisticRegression(solver='lbfgs',
                                multi_class='multinomial',
                                tol=0.01,
                                random_state=0,
                                max_iter=40000)
    logreg.fit(data, labels)

    save_model(logreg, model_file)

import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

import pickle


def save_obj(obj, name):
    """:param obj: object to save
       :param: name: name of the object"""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """ :param: name: name of the object to load"""
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

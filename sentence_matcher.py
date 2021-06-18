from os import path
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import namedtuple
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
Mapping = namedtuple("Mapping", "index_truth index_unknown probability")

loaded_models = {}


def make_dict(mappings):
    d = {}
    for m in mappings:
        d[m.index_unknown] = (m.index_truth, m.probability)
    return d


def check_model(model_path, model_name):
    model_path_name = path.join(model_path, model_name)

    if not path.exists(model_path):
        os.mkdir(model_path)

    if not path.exists(model_path_name):
        print(model_path_name, "dne, please download a model and unzip it")
        exit(1)

    return model_path_name


def get_model(model_path_name):
    if model_path_name in loaded_models:
        model = loaded_models[model_path_name]
    else:
        model = SentenceTransformer(model_path_name)
        loaded_models[model_path_name] = model

    return model


def encode_with_cache(sentences, cache, model, batch_size, show_progress_bar):
    result = np.zeros((len(sentences), 768))
    todos = []
    todos_idx = []

    dones = []
    dones_idx = []

    for i, s in enumerate(sentences):
        if s not in cache:
            todos.append(s)
            todos_idx.append(i)
        else:
            dones.append(s)
            dones_idx.append(i)

    todos_embedding = model.encode(todos, batch_size=batch_size, show_progress_bar=show_progress_bar)
    for i, embedding in zip(todos_idx, todos_embedding):
        result[i] = embedding

    for i, sentence in zip(dones_idx, dones):
        result[i] = cache[sentence]

    return result


def mapping(truth_names, unknown_names, show_progress_bar=False,
            batch_size=200 if device == 'gpu' else 32, kind='list',
            model_name='LaBSE_8.model', model_path='./data/models'):
    model_path_name = check_model(model_path, model_name)
    model = get_model(model_path_name)

    assert kind in ['list', 'map'], "expected '{}'".format("/".join(['list', 'map']))

    cache = load_cache(model_name, model_path)

    unknown_embeddings = encode_with_cache(unknown_names, cache, model, batch_size=batch_size,
                                           show_progress_bar=show_progress_bar)
    truth_embeddings = encode_with_cache(truth_names, cache, model, batch_size=batch_size,
                                         show_progress_bar=show_progress_bar)
    sim = cosine_similarity(unknown_embeddings, truth_embeddings)
    idx = np.argmax(sim, axis=1)
    mapped_names = [Mapping(it, iu, np.amax(s)) for it, iu, s in zip(idx, range(len(sim)), sim)]

    return make_dict(mapped_names) if kind == 'map' else mapped_names


def mapping_to_dataframe(mapping_list, truth_names, unknown_names):
    assert isinstance(mapping_list, list)
    result = []
    for m in mapping_list:
        result.append({
            'unknown_index': m.index_unknown,
            'unknown_name': unknown_names[m.index_unknown],
            'truth_index': m.index_truth,
            'true_name': truth_names[m.index_truth],
            'probability': m.probability,
        })

    return pd.DataFrame(result)


def make_cache(s, batch_size=200 if device == 'gpu' else 32,
               model_name='LaBSE_8.model', model_path='./data/models'):
    model_path_name = check_model(model_path, model_name)
    model = get_model(model_path_name)

    em = model.encode(s)
    c = load_cache(model_name, model_path)
    
    for s, e in zip(s, em):
        c[s] = e.tolist()

    pickle.dump(c, open(model_path_name + '.cache', "wb"))


def load_cache(model_name='LaBSE_8.model', model_path='./data/models'):
    model_path_name = path.join(model_path, model_name + '.cache')
    if os.path.exists(model_path_name):
        return pickle.load(open(model_path_name, "rb"))
    else:
        print('Cache not found for {}. Consider creating one ahead of time for a speed up'.format(model_path_name))
        return {}


def get_embeddings(sentences, model_name='LaBSE_8.model', model_path='./data/models'):
    model_path_name = check_model(model_path, model_name)
    model = get_model(model_path_name)
    return model.encode(sentences)

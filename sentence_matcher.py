from os import path
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import namedtuple

device = "cuda" if torch.cuda.is_available() else "cpu"
Mapping = namedtuple("Mapping", "index_truth index_unknown probability")


def make_dict(mappings):
    d = {}
    for m in mappings:
        d[m.index_unknown] = (m.index_truth, m.probability)
    return d


def mapping(truth_names, unknown_names, show_progress_bar=False, batch_size=200 if device == 'gpu' else 32, kind='list', model_name='LaBSE_8.model'):
    model_path = './data/models'
    model_name_zip = '{}.zip'.format(model_name)
    model_path_name = path.join(model_path, model_name)

    if not path.exists(model_path):
        os.mkdir(model_path)

    if not path.exists(model_path_name):
        print(model_path_name, "dne, please download a model and unzip it")
        exit(1)

    model = SentenceTransformer(model_path_name)

    assert kind in ['list', 'map'], "expected '{}'".format("/".join(['list', 'map']))
    unknown_embeddings = model.encode(unknown_names, batch_size=batch_size, show_progress_bar=show_progress_bar)
    truth_embeddings = model.encode(truth_names, batch_size=batch_size, show_progress_bar=show_progress_bar)
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

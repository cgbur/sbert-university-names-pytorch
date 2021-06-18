from sentence_matcher import mapping, mapping_to_dataframe, make_cache, load_cache, get_embeddings

unknown_names = ["a little", "whole", "all"]
make_cache(unknown_names)

c = load_cache()


embeddings = get_embeddings(unknown_names)

for n, e in zip(unknown_names, embeddings):
    for a, b in zip(c[n], e):
        assert a == b
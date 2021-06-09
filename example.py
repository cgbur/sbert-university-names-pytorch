from sentence_matcher import mapping, mapping_to_dataframe


truth_names = ["all", "some"]
unknown_names = ["a little", "whole", "all"]
g = mapping(truth_names, unknown_names, kind='map')
print(g)

i, p = g[0]
print(i, p)

g = mapping(truth_names, unknown_names, kind='list')
print(g)
print(mapping_to_dataframe(g, truth_names, unknown_names))


from sentence_matcher import mapping, mapping_to_dataframe


truth_names = ["all", "some"]
unknown_names = ["a little", "whole", "all"]
g = mapping(truth_names, unknown_names, kind='map')


truth_names = ["all", "some"]
unknown_names = ["a little", "whole", "all"] * 10
g = mapping(truth_names, unknown_names, kind='map')
print(g)

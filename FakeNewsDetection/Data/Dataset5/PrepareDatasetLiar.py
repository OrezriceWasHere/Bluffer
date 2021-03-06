import os
import pandas as pd

input_folder = "liar_dataset"
input_files = ("test.tsv", "train.tsv", "valid.tsv")
output_file = "output.tsv"
output_format = "{title}\t{label}\n"
input_indexes = {
    "title":    2,
    "label":   1
}
labels_map = {
    "true":         0,
    "mostly-true":  1,
    "half-true":    2,
    "barely-true":  3,
    "false":        4,
    "pants-fire":   5,
}

with open(output_file, "w") as output:
    output.truncate(0)
    output.write(output_format.format(title="title", label="label"))
    for input_file_name in input_files:
        path_for_input_file = os.path.join(input_folder, input_file_name)
        input = pd.read_csv(path_for_input_file, sep='\t', header=None)
        for index, row in input.iterrows():
            title = row[input_indexes["title"]]
            label = row[input_indexes["label"]]
            label = labels_map[label]
            line = output_format.format(title=title, label=label)
            output.write(line)

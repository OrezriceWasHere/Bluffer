import os
import pandas as pd

from FakeNewsDetection import Parameters

input_folder = "liar_dataset"
input_files = ("train.tsv", "test.tsv", "valid.tsv")
output_file = "output.tsv"
output_format = "{title}\t{label}\n"
input_indexes = {
    "title":    2,
    "label":    1,
    "id":       0
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
    # count = 1
    for input_file_name in input_files:
        path_for_input_file = os.path.join(input_folder, input_file_name)
        input = pd.read_csv(path_for_input_file, sep='\t', header=None)
        for index, row in input.iterrows():
            title = row[input_indexes["title"]]
            if len(title) > Parameters.MAX_SEQ_LEN:
                continue
            label_text = row[input_indexes["label"]]
            label_int = labels_map[label_text]
            label_int = int(label_int)
            line = output_format.format(title=title, label=label_int)
            output.write(line)
            # output.flush()
            # count += 1

import os
import pandas as pd
from FakeNewsDetection.DatasetPrepare import clean_text

path = os.path.dirname(os.path.abspath(__file__))
input_file_loc = os.path.join(path, "news.tsv")
output_file_loc = os.path.join(path, "output.tsv")
title_index = 0
label_index = 1
header = "{title}\t{label}\n"

with open(output_file_loc, "w") as output:
    input_file = pd.read_csv(input_file_loc, sep="\t", skiprows=1)

    # Empty output file
    output.truncate(0)
    # Write Header s
    output.write(header.replace("{","").replace("}", ""))

    for index, row in input_file.iterrows():
        title = row[title_index]
        label = row[label_index]
        if title is not None and title and not title.isspace():
            output.write(header.format(title=clean_text(row[title_index]), label=label))

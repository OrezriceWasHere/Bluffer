import os
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))
folder_name = "News_dataset"
fake_news_path = os.path.join(path, folder_name, "Fake.csv")
real_news_path = os.path.join(path, folder_name, "True.csv")
title_index = 0
output_file_name = os.path.join(path, "output.tsv")
header = "{title}\t{label}\n"

with open(output_file_name, "w") as output:
    real_file = pd.read_csv(real_news_path, skiprows=1)
    fake_file = pd.read_csv(fake_news_path, skiprows=1)

    # Empty output file
    output.truncate(0)
    # Write Header s
    output.write(header.replace("{","").replace("}", ""))

    for index, row in real_file.iterrows():
        output.write(header.format(title=row[title_index], label="1"))

    for index, row in fake_file.iterrows():
        output.write(header.format(title=row[title_index], label="0"))







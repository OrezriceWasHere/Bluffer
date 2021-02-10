from torchtext.data import Dataset
import pandas as pd


class csvDataset(Dataset):

    def __init__(self, file_location, include_header=True):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.csv_file = pd.read_csv(file_location)
        self.examples = self.csv_file.itertuples()
        self.offset = 1 if include_header else 0
        # self.currIndex = 0


    def __len__(self):
        return len(self.csv_file) - self.offset


    def __getindex__(self, index):
        return self.csv_file[(index + self.offset)]

    # def __iter__(self):
    #     return self.examples.__iter__()

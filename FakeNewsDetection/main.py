from os.path import join, exists
import torch.optim as optim
from HexaFakeNewsModel import HexaFakeNewsModel
from BERT import BERT
import Parameters
import DatasetPrepare
import SaveLoad
from pathlib import Path
from Evaluate import evaluate, display_loss_graph
from Train import train


def display_result(model, metric_file_location, test_loader,  title=""):
    display_loss_graph(metric_file_location, title=title)
    evaluate(model, test_loader, title=title)

dataset = {
    "data_file": join("Data", "Dataset5", "output.tsv"),
    "output_dir": join("Data", "Dataset5", "output")
}

pre_trained_model_file = join(Parameters.SOURCE_4_FOLDER, "output", Parameters.MODEL_FILE_NAME)
print('-------------------------------------')
print('Loading pre-trained dataset into memory and creating a better one')
pre_trained_model = BERT().to(Parameters.DEVICE)
SaveLoad.load_checkpoint(pre_trained_model_file, pre_trained_model)
model = HexaFakeNewsModel(bert_model=pre_trained_model).to(Parameters.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)
print('finished creating a model')
print('preparing dataset')

train_iterator, test_iterator = DatasetPrepare.create_iterators(dataset["data_file"])
Path(dataset["output_dir"]).mkdir(parents=True, exist_ok=True)
model_output_file = join(dataset["output_dir"], "model.pt")
metric_output_file = join(dataset["output_dir"], "metric.pt")

# Deleting previous content of files
if not exists(model_output_file):
    with open(model_output_file, "w"):
        pass
if not exists(metric_output_file):
    with open(metric_output_file, "w"):
        pass

print('finished loading data')
print('training the data')
train(model=model,
      optimizer=optimizer,
      train_loader=train_iterator,
      test_loader=test_iterator,
      eval_every=len(train_iterator) // 2,
      model_output_file=model_output_file,
      metric_output_file=metric_output_file,
      num_epochs=20)

display_result(model=model,
               metric_file_location=metric_output_file,
               test_loader=test_iterator,
               title=f'Result 6 class dataset')

print('finish creating a dataset')



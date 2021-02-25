from BERT import BERT
from SaveLoad import load_checkpoint
from Evaluate import evaluate, display_loss_graph
from DatasetPrepare import train_iter
from Train import train
from os.path import join
import Parameters

import DatasetPrepare
import torch.optim as optim


# def train_model(model, optimizer):
#     train(model=model, optimizer=optimizer, num_epochs=10, eval_every=len(train_iter) // 2)
#     print("done")


def display_result(model, metric_file_location, test_loader,  title=""):
    display_loss_graph(metric_file_location, title=title)
    evaluate(model, test_loader, title=title)


datasets = [
    {
        "data_file": join(Parameters.SOURCE_1_FOLDER, "news.tsv"),
        "model_output_file": join(Parameters.SOURCE_1_FOLDER, "output", "model.pt"),
        "metric_output_file": join(Parameters.SOURCE_1_FOLDER, "output", "metric.pt")
    },
    {
        "data_file": join(Parameters.SOURCE_2_FOLDER, "news.tsv"),
        "model_output_file": join(Parameters.SOURCE_2_FOLDER, "output", "model.pt"),
        "metric_output_file": join(Parameters.SOURCE_2_FOLDER, "output", "metric.pt")
    },
    {
        "data_file": join(Parameters.SOURCE_3_FOLDER, "news.tsv"),
        "model_output_file": join(Parameters.SOURCE_3_FOLDER, "output", "model.pt"),
        "metric_output_file": join(Parameters.SOURCE_3_FOLDER, "output", "metric.pt")
    },
    {
        "data_file": join(Parameters.SOURCE_4_FOLDER, "output.tsv"),
        "model_output_file": join(Parameters.SOURCE_4_FOLDER, "output", "model.pt"),
        "metric_output_file": join(Parameters.SOURCE_4_FOLDER, "output", "metric.pt")
    }
]

model = BERT()
model = model.to(Parameters.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
for index, dataset in enumerate(datasets):
    print('-------------------------------------')
    print(f'now working on dataset {index + 1}')
    train_iterator, test_iterator = DatasetPrepare.create_iterators(dataset["data_file"])
    train(model=model,
          optimizer=optimizer,
          train_loader=train_iterator,
          test_loader=test_iterator,
          model_output_file=dataset["model_output_file"],
          metric_output_file=dataset["metric_output_file"])
    display_result(model=model,
                   metric_file_location=dataset["metric_output_file"],
                   test_loader=test_iterator,
                   title=f'Result Dataset #{index + 1} and before')
    print(f'now finished working on dataset {index + 1}')

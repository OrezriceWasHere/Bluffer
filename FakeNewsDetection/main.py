from BERT import BERT
from Evaluate import evaluate, display_loss_graph
from Train import train
from os.path import join, exists
import Parameters
from pathlib import Path
import DatasetPrepare
import torch.optim as optim


def display_result(model, metric_file_location, test_loader,  title=""):
    display_loss_graph(metric_file_location, title=title)
    evaluate(model, test_loader, title=title)


datasets = [
    {
        "data_file": join(Parameters.SOURCE_1_FOLDER, "news.tsv"),
        "output_dir": join(Parameters.SOURCE_1_FOLDER, "output")
    },
    {
        "data_file": join(Parameters.SOURCE_2_FOLDER, "news.tsv"),
        "output_dir": join(Parameters.SOURCE_2_FOLDER, "output")
    },
    {
        "data_file": join(Parameters.SOURCE_4_FOLDER, "output.tsv"),
        "output_dir": join(Parameters.SOURCE_4_FOLDER, "output")
    }
]

model = BERT()
model = model.to(Parameters.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)
for index, dataset in enumerate(datasets):
    # dataset = datasets[index]
    print('-------------------------------------')
    print(f'now working on dataset {index + 1}')
    train_iterator, test_iterator = DatasetPrepare.create_iterators(dataset["data_file"])
    Path(dataset["output_dir"]).mkdir(parents=True, exist_ok=True)
    model_output_file = join(dataset["output_dir"], "model.pt")
    metric_output_file = join(dataset["output_dir"], "metric.pt")
    if not exists(model_output_file):
        with open(model_output_file, "w"):
            pass
    if not exists(metric_output_file):
        with open(metric_output_file, "w"):
            pass

    if index > 0:
        display_result(model=model,
                       metric_file_location=metric_output_file,
                       test_loader=test_iterator,
                       title=f'Result Before training on dataset #{index + 1}')

    train(model=model,
          optimizer=optimizer,
          train_loader=train_iterator,
          test_loader=test_iterator,
          eval_every=len(train_iterator) // 2,
          model_output_file=model_output_file,
          metric_output_file=metric_output_file,
          num_epochs=10)

    display_result(model=model,
                   metric_file_location=metric_output_file,
                   test_loader=test_iterator,
                   title=f'Result Dataset #{index + 1} and before')

print(f'now finished working on dataset {index + 1}')

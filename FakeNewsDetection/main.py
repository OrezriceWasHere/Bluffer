from BERT import BERT
from SaveLoad import load_checkpoint
from Evaluate import evaluate, display_loss_graph
from DatasetPrepare import train_iter
import Parameters
from Train import train
import torch.optim as optim
import os


def train_model(model):
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    train(model=model, optimizer=optimizer, num_epochs=10, eval_every=len(train_iter) // 2)
    print("done")


def display_result(model, metric_file_location,  title=""):
    display_loss_graph(metric_file_location, title=title)
    evaluate(model, train_iter, title=title)


# model = BERT()
# model = model.to(Parameters.DEVICE)
# load_checkpoint(Parameters.OUTPUT_3_FOLDER + "/" + Parameters.MODEL_FILE_NAME, model)
# # rain_model(model)
# display_result(model)


models = [
    {
        "title": "With Pre Trained Dataset",
        "model_file_location": os.path.join(Parameters.OUTPUT_3_FOLDER,
                                            "with-pre-trained-model",
                                            Parameters.MODEL_FILE_NAME),
        "model_metric_location": os.path.join(Parameters.OUTPUT_3_FOLDER,
                                              "with-pre-trained-model",
                                              Parameters.METRICS_FILE_NAME),
        "model": BERT()
    },
    {
        "title": "Without Pre Trained Dataset",
        "model_file_location": os.path.join(Parameters.OUTPUT_3_FOLDER,
                                            "without-pre-trained-model",
                                            Parameters.MODEL_FILE_NAME),
        "model_metric_location": os.path.join(Parameters.OUTPUT_3_FOLDER,
                                              "without-pre-trained-model",
                                              Parameters.METRICS_FILE_NAME),

        "model": BERT()
    }
]

for model in models:
    print("Now working on model " + model["title"])
    model["model"].to(Parameters.DEVICE)
    load_checkpoint(model["model_file_location"], model["model"])
    display_result(model=model["model"], metric_file_location=model["model_metric_location"], title=model["title"])



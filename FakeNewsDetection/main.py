from BERT import BERT
from SaveLoad import load_checkpoint
from Evaluate import evaluate, display_loss_graph
from DatasetPrepare import train_iter
import Parameters
from Train import train
import torch.optim as optim


def train_model(model):
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    train(model=model, optimizer=optimizer, num_epochs=10)
    print("done")


def display_result(model):
    load_checkpoint(Parameters.MODEL_OUTPUT_FILE, model)
    display_loss_graph(Parameters.METRICS_OUTPUT_FILE)
    evaluate(model, train_iter)


model = BERT()
model = model.to(Parameters.DEVICE)
load_checkpoint(Parameters.OUTPUT_1_FOLDER, model)
train_model(model)
display_result(model)

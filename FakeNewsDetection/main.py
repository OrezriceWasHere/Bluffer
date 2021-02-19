from BERT import BERT
from SaveLoad import load_checkpoint
from Evaluate import evaluate, display_loss_graph
from DatasetPrepare import train_iter
import Parameters
from Train import train
import torch.optim as optim


def train_model():
    model = BERT()
    model = model.to(Parameters.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train(model=model, optimizer=optimizer, num_epochs=10)
    print("done")


def display_result():
    best_model = BERT().to(Parameters.DEVICE)
    load_checkpoint(Parameters.MODEL_OUTPUT_FILE, best_model)
    display_loss_graph(Parameters.METRICS_OUTPUT_FILE)
    evaluate(best_model, train_iter)


train_model()
display_result()

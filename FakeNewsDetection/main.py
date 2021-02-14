from BERT import BERT
import Parameters
from Train import train
import torch.optim as optim

model = BERT()
model = model.to(Parameters.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)
print("done")

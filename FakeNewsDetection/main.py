from BERT import BERT
import Parameters
from Train import train
import torch.optim as optim

model = BERT()
model = model.to(Parameters.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

train(model=model, optimizer=optimizer, num_epochs=10)
print("done")

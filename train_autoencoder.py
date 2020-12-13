import os
import torch
import sparseconvnet as scn
from models import Autoencoder
from module.utils.muon_track_dataset import MuonPose
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

train = MuonPose("/home/jack/classes/thesis/autohas/LArCentroids/train/")
val = MuonPose("/home/jack/classes/thesis/autohas/LArCentroids/val/")
logger = SummaryWriter(log_dir="autoencoder_run/")

device = 'cuda:0'
dimension = 3
reps = 3
n_layers = [(1, 4), (4, 16), (16, 64), (64, 256), (256, 1024)]

model = Autoencoder(dimension, reps, n_layers)
model.to_(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

densifier = scn.Sequential().add(
    scn.InputLayer(3, torch.LongTensor([128,128,128]))
).add(
    scn.SparseToDense(3, 1)
)

for epoch in range(20):
    print(f"Epoch {epoch}")
    total_loss = 0
    for i, (_, target) in enumerate(train):
        target = [torch.from_numpy(t).to(device) for t in target]
        target[1] = target[1].float().reshape(-1, 1)

        prediction = model(target)
        dense_prediction = densifier[1](prediction)
        dense_target = densifier(target)
        loss = loss_fn(dense_prediction, dense_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            logger.add_scalar("Loss/train", loss.item(), (epoch+1)*i)
            print(f"\tIter {100*i/len(train):.2f}%: Loss {loss.item()}", end='\r')
    total_loss /= len(train)
    print(f"\tAverage training loss: {total_loss}")

    total_loss = 0
    for i, (_, target) in enumerate(val):
        target = [torch.from_numpy(t).to(device) for t in target]
        target[1] = target[1].float().reshape(-1, 1)

        prediction = model(target)
        dense_prediction = densifier[1](prediction)
        dense_target = densifier(target)
        loss = loss_fn(dense_prediction, dense_target)

        total_loss += loss.item()
        if i % 10 == 0:
            print(f"\tIter {100*i/len(val):.2f}%: Loss {loss.item()}", end='\r')
    total_loss /= len(val)
    logger.add_scalar("Loss/val", total_loss, epoch)
    print(f"\tAverage validation loss: {total_loss}")

    print("\tSaving model checkpoint...")
    torch.save(model.state_dict(), f'autoencoderweights/autoencoder_weights_{epoch}.pt')

import os
import torch
import sparseconvnet as scn
from models import Autoencoder
from module.utils.muon_track_dataset import MuonPose
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.manual_seed(42)

train = MuonPose("/home/jack/classes/thesis/autohas/LArCentroids/train/", return_energy=True)
val = MuonPose("/home/jack/classes/thesis/autohas/LArCentroids/val/", return_energy=True)
now = datetime.now().strftime("%b-%d-%y_%H:%M:%S")
logger = SummaryWriter(log_dir=f"autoencoder_run/{now}/")

device = 'cuda:0'
dimension = 3
reps = 3
encoder_layers = [(1, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
# [(output_features, input_features), ...]
decoder_layers = [(32, 128), (16, 64), (8, 32), (4, 16), (1, 8), (1, 2, False)]

model = Autoencoder(dimension, reps, encoder_layers, decoder_layers, unet=True, use_sparsify=False)
model.to_(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

densifier = scn.Sequential().add(
    scn.InputLayer(3, torch.LongTensor([128,128,128]))
).add(
    scn.SparseToDense(3, 1)
)

print("Want to test or load anything?")
import code
code.interact(local=locals())

for epoch in range(20):
    print(f"Epoch {epoch}")
    total_loss = 0
    model.train()
    for i, (inputs, target) in enumerate(train):
        # discard light information
        inputs = inputs[1]
        inputs = [torch.from_numpy(t).to(device) for t in inputs]
        inputs[1] = inputs[1].float().reshape(-1, 1)
        target = [torch.from_numpy(t).to(device) for t in target]
        target[1] = target[1].float().reshape(-1, 1)

        prediction = model(inputs)
        if len(prediction.features) == 1:
            print("empty prediction at i=", i)
            import code
            code.interact(local=locals())
        dense_prediction = densifier[1](prediction)

        dense_target = densifier(target)

        loss = loss_fn(dense_prediction, dense_target)
        if torch.isnan(loss):
            print("nanloss at i=",i)
            import code
            code.interact(local=locals())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_scalar("Loss/train", loss.item(), (epoch*len(train)) + i)
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"\tIter {100*i/len(train):.2f}%: Loss {loss.item()}", end='\r')
    total_loss /= len(train)
    print(f"\tAverage training loss: {total_loss}")

    model.eval()
    total_loss = 0
    for i, (inputs, target) in enumerate(val):
        # discard light information
        inputs = inputs[1]
        inputs = [torch.from_numpy(t).to(device) for t in inputs]
        inputs[1] = inputs[1].float().reshape(-1, 1)
        target = [torch.from_numpy(t).to(device) for t in target]
        target[1] = target[1].float().reshape(-1, 1)

        prediction = model(inputs)
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
    torch.save(model.state_dict(), f'autoencoder_weights/autoencoder_weights_{epoch}.pt')

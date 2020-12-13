import sparseconvnet as scn
import torch
import os

def resnet_block(dimension, n_in, n_out, kernel, leakiness=0, computation='convolution'):
    '''Build and return ResNet block
    '''

    assert computation in ['submanifoldconvolution', 'convolution', 'fullconvolution', 'deconvolution']
    if computation == 'convolution':
        computation = lambda n_in, n_out: scn.Convolution(dimension, n_in, n_out, kernel[0], kernel[1], False)
    elif computation == 'submanifoldconvolution':
        assert type(kernel) == int, f"`kernel` must be int, {type(kernel)} was provided"
        computation = lambda n_in, n_out: scn.SubmanifoldConvolution(dimension, n_in, n_out, kernel, False)
    elif computation == 'deconvolution':
        assert type(kernel) == int, f"`kernel` must be int, {type(kernel)} was provided"
        computation = lambda n_in, n_out: scn.Deconvolution(dimension, n_in, n_out, kernel, kernel, False)
    else:
        computation = lambda n_in, n_out: scn.FullConvolution(dimension, n_in, n_out, kernel[0], kernel[1], False)

    block = scn.Sequential()
    block.add(
        scn.ConcatTable().add(
            scn.NetworkInNetwork(n_in, n_out, False)
        ).add(
            scn.Sequential().add(
                scn.BatchNormLeakyReLU(n_in, leakiness=leakiness)
            ).add(
                computation(n_in, n_out)
            ).add(
                scn.BatchNormLeakyReLU(n_out, leakiness=leakiness)
            ).add(
                computation(n_out, n_out)
            )
        )
    ).add(
        scn.AddTable()
    )
    return block

def get_block_name(model_name, dimension, reps, n_in, n_out, leakiness):
    return f'{model_name}:{dimension}-{reps}-{n_in}-{n_out}-{leakiness}'

class Encoder(torch.nn.Module):
    def __init__(self, dimension, reps, n_layers, leakiness=0, input_layer=None, name='encoder'):
        super(Encoder, self).__init__()
        if input_layer != None:
            self.input_layer = scn.InputLayer(len(input_layer), input_layer)

        self.blocks = []
        self.block_names = {}
        n_in, n_out = 1, 1
        for i in range(len(n_layers)):
            block = scn.Sequential()
            # add reps Resnet blocks, where reps >= 1 and first block just ensures number of
            # input channels is correct
            for rep in range(reps):
                block.add(
                    resnet_block(dimension, n_in, n_out, 1, leakiness, computation='submanifoldconvolution')
                )
                n_in = n_out
            n_out = n_layers[i][1]
            block.add(
                scn.BatchNormLeakyReLU(n_in, leakiness)
            )
            if len(n_layers[i]) == 2:
                block.add(
                    scn.Convolution(dimension, n_in, n_out, 2, 2, False)
                )
            elif len(n_layers[i]) == 3 and n_layers[i][2] == 'maxpool':
                block.add(
                    scn.MaxPooling(dimension, 2, 2)
                )
            elif len(n_layers[i]) == 3 and n_layers[i][2] == 'avgpool':
                block.add(
                    scn.AveragePooling(dimension, 2, 2)
                )
            block_name = get_block_name(name, dimension, reps, n_in, n_out, leakiness)
            n_in = n_out
            self.blocks.append(block)
            self.block_names[block_name] = len(self.blocks)-1
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x, return_blocks=False):
        if hasattr(self, "input_layer"):
            x = self.input_layer(x)

        rets = [x]
        for i in range(len(self.blocks)):
            ret = self.blocks[i](rets[i])
            rets.append(ret)

        if return_blocks:
            return ret, rets[1:]
        else:
            return ret

    def unfreeze(self):
        for p in model.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in model.parameters():
            p.requires_grad = False


class Decoder(torch.nn.Module):
    def __init__(self, dimension, n_layers, name='decoder'):
        super(Decoder, self).__init__()
        self.blocks = []
        self.block_names = {}

        for i in range(len(n_layers)):
            n_in, n_out = n_layers[i][1], n_layers[i][0]
            block = scn.Sequential().add(
                scn.BatchNormLeakyReLU(n_in, 0)
            ).add(
                scn.FullConvolution(dimension, n_in, n_out, 2, 2, False)
            )
            block_name = get_block_name(name, dimension, 0, n_in, n_out, 0)
            self.blocks.append(block)
            self.block_names[block_name]= len(self.blocks)-1
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x, return_blocks=False):
        rets = [x]
        for i in range(len(self.blocks)):
            ret = self.blocks[i](rets[i])
            rets.append(ret)

        if return_blocks:
            return ret, rets[1:]
        else:
            return ret

    def unfreeze(self):
        for p in model.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in model.parameters():
            p.requires_grad = False


class Autoencoder(torch.nn.Module):
    def __init__(self, dimension, reps, n_layers):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(dimension, reps, n_layers, input_layer=torch.LongTensor([128,128,128]))
        self.decoder = Decoder(dimension, n_layers[::-1])
        self.sigmoid = scn.Sigmoid()

    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        x_hat = self.sigmoid(x_hat)

        if return_latent:
            return x_hat, latent
        else:
            return x_hat

    def save_weights(self, directory='autoencoderweights/', suffix=None):
        if directory[-1] != '/':
            directory += '/'
        if not os.path.isdir(directory):
            os.mkdir(directory)

        for i in range(len(self.encoder.blocks)):
            end = f'encoder-block{i}{suffix}.pt' if suffix is not None else f'encoder-block{i}.pt'
            torch.save(self.encoder.blocks[i].state_dict(), directory + end)
        for i in range(len(self.decoder.blocks)):
            end = f'decoder-block{i}{suffix}.pt' if suffix is not None else f'decoder-block{i}.pt'
            torch.save(self.decoder.blocks[i].state_dict(), directory + end)

    def load_weights(self, directory):
        if directory[-1] != '/':
            directory += '/'

        files = [f for f in os.listdir(directory) if f.split('.')[-1] == 'pt']
        for i in range(len(self.encoder.blocks)):
            filename = [f for f in files if f'encoder-block{i}' in f]
            assert len(filename) == 1, f"Could not find unique weights for encoder block {i}: {filename}"
            self.encoder.blocks[i].load_state_dict(torch.load(directory+filename[0]))
        for i in range(len(self.decoder.blocks)):
            filename = [f for f in files if f'decoder-block{i}' in f]
            assert len(filename) == 1, f"Could not find unique weights for decoder block {i}: {filename}"
            self.decoder.blocks[i].load_state_dict(torch.load(directory+filename[0]))

    def parameters_(self):
        encoder_parameters = [b.parameters() for b in self.encoder.blocks]
        decoder_parameters = [b.parameters() for b in self.decoder.blocks]
        return encoder_parameters + decoder_parameters

    def to_(self, device):
        for i in range(len(self.encoder.blocks)):
            self.encoder.blocks[i].to(device)
        for i in range(len(self.decoder.blocks)):
            self.decoder.blocks[i].to(device)

    def unfreeze(self):
        self.encoder.unfreeze()
        self.decoder.unfreeze()

    def freeze(self):
        self.encoder.freeze()
        self.decoder.freeze()



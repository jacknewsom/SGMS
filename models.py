import sparseconvnet as scn
import numpy as np
import torch
import os

class SparseConcatenate(torch.nn.Module):
    def __init__(self, device):
        super(SparseConcatenate, self).__init__()
        self.device = device

    '''
    def forward(self, a, b):
        assert torch.equal(a.spatial_size, b.spatial_size), (a.spatial_size, b.spatial_size)
        locations = torch.vstack((a.get_spatial_locations(), b.get_spatial_locations()))
        unique_locations, reverse_indices = torch.unique(locations, dim=0, return_inverse=True)

        sites = (a.features.shape[0], b.features.shape[0])
        a_reverse_indices, b_reverse_indices = reverse_indices[:sites[0]], reverse_indices[sites[0]:]

        features = (a.features.shape[1], b.features.shape[1])
        unique_features = torch.zeros((len(unique_locations), sum(features)), device=self.device)

        unique_features[a_reverse_indices, :features[0]] = a.features
        unique_features[b_reverse_indices, features[0]:] = b.features

        return scn.InputLayer(len(a.spatial_size), a.spatial_size)([unique_locations, unique_features])
    '''

    def forward(self, a, b):
        assert torch.equal(a.spatial_size, b.spatial_size), (a.spatial_size, b.spatial_size)
        unique_locations, unique_features = SparseConcatenateFunction.apply(
            a.get_spatial_locations(),
            a.features,
            b.get_spatial_locations(),
            b.features)
#        output = scn.InputLayer(len(a.spatial_size), a.spatial_size).to(self.device)([unique_locations, unique_features])

        return output

class SparseConcatenateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_locations, a_features, b_locations, b_features):
        locations = torch.vstack((a_locations, b_locations))
        unique_locations, reverse_indices = torch.unique(locations, dim=0, return_inverse=True)

        sites = (a_features.shape[0], b_features.shape[0])
        a_reverse_indices, b_reverse_indices = reverse_indices[:sites[0]], reverse_indices[sites[0]:]

        features = (a_features.shape[1], b_features.shape[1])
        unique_features = torch.zeros((len(unique_locations), sum(features)), device=a_features.device)

        unique_features[a_reverse_indices, :features[0]] = a_features
        unique_features[b_reverse_indices, features[0]:] = b_features

        ctx.save_for_backward(a_locations, a_features, b_locations, b_features, unique_locations, unique_features, a_reverse_indices, b_reverse_indices)

        return unique_locations, unique_features

    @staticmethod
    def backward(ctx, grad_locations_output, grad_features_output):
        a_locations, a_features, b_locations, b_features, unique_locations, unique_features, a_reverse_indices, b_reverse_indices = ctx.saved_tensors
        grad_input_a, grad_input_b = torch.zeros_like(a_features), torch.zeros_like(b_features)

        grad_input_a = grad_features_output[a_reverse_indices, :a_features.shape[1]]
        grad_input_b = grad_features_output[b_reverse_indices, a_features.shape[1]:]

        return None, grad_input_a, None, grad_input_b


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

class SparseMultiply(torch.nn.Module):
    def __init__(self):
        super(SparseMultiply, self).__init__()

    def forward(self, a, b):
        output = scn.SparseConvNetTensor()
        output.metadata = a.metadata
        output.spatial_size = a.spatial_size
        output.features = a.features * b.features

        return output

class AttentionBlock(torch.nn.Module):
    def __init__(self, dimension, n_in, kernel, leakiness, unpool_size, unpool_stride, threshold):
        super(AttentionBlock, self).__init__()

        self.block = scn.Sequential()
        self.block.add(
            scn.SubmanifoldConvolution(dimension, n_in, n_in // 2, kernel, False)
        ).add(
            scn.BatchNormLeakyReLU(n_in // 2, leakiness=leakiness)
        ).add(
            scn.SubmanifoldConvolution(dimension, n_in // 2, n_in // 4, kernel, False)
        ).add(
            scn.BatchNormLeakyReLU(n_in // 4, leakiness=leakiness)
        ).add(
            scn.SubmanifoldConvolution(dimension, n_in // 4, 2, kernel, False)
        )

        self.softmax = torch.nn.Softmax(dim=1)
        self.unpool = scn.UnPooling(dimension, unpool_size, unpool_stride)
        self.threshold = threshold

    def forward(self, x):
        raw_prob = self.block(x)
        probabilities = self.softmax(raw_prob.features)

        mask = scn.SparseConvNetTensor()
        mask.metadata = x.metadata
        mask.spatial_size = x.spatial_size
        mask.features = (probabilities[:, 1] > self.threshold).float().reshape(-1, 1)

        unpooled_mask = self.unpool(mask)
        unpooled_mask.features = (unpooled_mask.features > self.threshold).astype

        return unpooled_mask

class Encoder(torch.nn.Module):
    def __init__(self, dimension, reps, n_layers, leakiness=0, input_layer=None, name='encoder', device=None):
        super(Encoder, self).__init__()
        self.dimension = dimension
        self.reps = reps
        self.n_layers = n_layers
        self.leakiness = leakiness
        self.name = name
        self.device = device

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
            elif len(n_layers[i]) == 3 and n_layers[i][2] == 'submanifoldconvolution':
                block.add(
                    scn.SubmanifoldConvolution(dimension, n_in, n_out, 2, False)
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
        self.blocks.to(self.device)

    def forward(self, x, return_blocks=False):
        if hasattr(self, "input_layer"):
            x = self.input_layer(x)

        rets = [x]
        for i in range(len(self.blocks)):
            ret = self.blocks[i](rets[i])
            rets.append(ret)

        if return_blocks:
            return rets
        else:
            return ret

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


class Decoder(torch.nn.Module):
    def __init__(self, dimension, n_layers, unet=False, name='decoder', is_submanifold=False, use_sparsify=True, device=None):
        super(Decoder, self).__init__()
        self.dimension = dimension
        self.n_layers = n_layers
        self.unet = unet
        self.name = name
        self.device = device

        self.blocks = []
        self.block_names = {}

        for i in range(len(n_layers)):
            n_in, n_out = n_layers[i][0], n_layers[i][1]
            block = scn.Sequential()

            block.add(
                scn.BatchNormLeakyReLU(n_in, 0)
            )
            if len(n_layers[i]) == 2 and not is_submanifold:
                block.add(
                    scn.FullConvolution(dimension, n_in, n_out, 2, 2, False)
                )
            elif len(n_layers[i]) == 2 and is_submanifold:
                block.add(
                    scn.Deconvolution(dimension, n_in, n_out, 2, 2, False)
                )
            elif len(n_layers[i]) == 3 and n_layers[i][2] == False:
                # don't upsample
                block.add(
                    scn.SubmanifoldConvolution(dimension, n_in, n_out, 1, False)
                )
            if use_sparsify:
                block.add(
                    scn.Sparsify(dimension, n_out) 
                )
            block_name = get_block_name(name, dimension, 0, n_in, n_out, 0)
            self.blocks.append(block)
            self.block_names[block_name] = len(self.blocks)-1
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.blocks.to(self.device)

    def forward(self, x, return_blocks=False):
        if not self.unet:
            rets = [x]
            for i in range(len(self.blocks)):
                ret = self.blocks[i](rets[i])
                rets.append(ret)
        else:
            '''
            `x` is list of inputs for each layer of `self`. `x[0]` should have the same number of feature channels
            as the first block in `self`, then `x[1]` should have `self.blocks[1].input_features - self.blocks[0](x[0]).features`
            features, `x[2]` should have `self.blocks[2].input_features - self.blocks[1](self.blocks[0](x[0])).features`, etc.
            '''
            assert type(x) == list
            assert len(x) == len(self.blocks)

            concatenator = SparseConcatenate(self.device)
            rets = [x[0]]
            prev = x[0]
            for i, block in enumerate(self.blocks):
                assert prev.features.shape[1] == block[1].nIn, f'Input to block {i} should have {block[1].nIn} features, but had {prev.features.shape[1]}'
                prev = block(prev)
                
                if i < len(self.blocks)-1:
                    encoder_output = x[i+1]
                    prev = scn.JoinTable()([prev, encoder_output])
                rets.append(prev)
        if return_blocks:
            return rets
        else:
            return rets[-1]

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

class Decoder_(torch.nn.Module):
    '''One that makes more sense
    '''
    def __init__(self, dimension, n_layers, reps, device='cpu:0'):
        super(Decoder_, self).__init__()
        self.dimension = dimension
        self.n_layers = n_layers
        self.device = device
        self.concatenator = scn.JoinTable()

        blocks = []
        for i in range(n_layers):
            n_in, n_out = n_layers[i]

            block = scn.Sequential()
            for rep in reps:
                block.add(
                    resnet_block(dimension, n_in, n_out, 3, leakiness, computation='submanifoldconvolution')
                )
                n_in = n_out

            block.add(
                scn.Deconvolution(dimension, n_in, n_out, 2, 2, False)
            )
        self.blocks = torch.nn.ModuleList(blocks)
        self.blocks.to(self.device)

    def forward(self, x):
        '''
        `x` is list of inputs for each layer of `self`. `x[0]` should have the same number of feature channels
        as the first block in `self`, then `x[1]` should have `self.blocks[1].input_features - self.blocks[0](x[0]).features`
        features, `x[2]` should have `self.blocks[2].input_features - self.blocks[1](self.blocks[0](x[0])).features`, etc.
        '''
        assert type(x) == list
        assert len(x) == len(self.blocks)

        rets = [x[0]]
        prev = x[0]
        for i, block in enumerate(self.blocks):
            assert prev.features.shape[1] == block[1].nIn, f'Input to block {i} should have {block[1].nIn} features, but had {prev.features.shape[1]}'
            prev = block(prev)
            
            if i < len(self.blocks)-1:
                encoder_output = x[i+1]
                prev = self.concatenator([prev, encoder_output])
            rets.append(prev)
        return rets

class Autoencoder(torch.nn.Module):
    def __init__(self, dimension, reps, encoder_layers, decoder_layers, unet=False, use_sparsify=False, device='cuda:0'):
        super(Autoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(dimension, reps, encoder_layers, input_layer=torch.LongTensor([128,128,128]), device=device)
        self.decoder = Decoder(dimension, decoder_layers, unet=unet, use_sparsify=use_sparsify, device=device)
        self.sigmoid = scn.Sigmoid()

    def forward(self, x, return_latent=False):
        if self.decoder.unet:
            latent, blocks = self.encoder(x, return_blocks=True)
            blocks.append(latent)
            blocks = blocks[::-1]
            x_hat = self.decoder([latent, blocks])
        else:
            latent = self.encoder(x)
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

class YNet(torch.nn.Module):
    def __init__(self, dimension, reps, encoder_layers, leakiness, attention_threshold, device):
        super(YNet, self).__init__()
        self.concatenator = scn.JoinTable()
        self.primary = Encoder(dimension, reps, encoder_layers, input_layer=torch.LongTensor([128,128,128]), device=device)
        self.secondary = Encoder(dimension, reps, encoder_layers, input_layer=torch.LongTensor([128,128,128]), device=device)
        self.decoder = Decoder(dimension, decoder_layers, unet=True, is_submanifold=True, device=device)

        attention_block_layers = [d[1] for d in decoder_layers]
        self.attention = [AttentionBlock(dimension, b, 3, leakiness, 2, 2, attention_threshold) for b in attention_block_layers]

        self.multiplier = SparseMultiply()

    def forward(self, x_primary, x_secondary):
        latent_primary = self.primary(x_primary, return_blocks=True)
        latent_secondary = self.secondary(x_secondary)

        combined_latent = self.concatenator([latent_primary[-1], latent_secondary])
        encoded_blocks = latent_primary[:-1] + [combined_latent]

        decoded_blocks = self.decoder(encoded_blocks)

        mask = None
        for decoded, attn_block in zip(decoded_blocks, self.attention):
            if mask is None:
                input_ = decoded
            else:
                input_ = self.multiplier(mask, decoded)
            mask = attn_block(input_)
        return mask
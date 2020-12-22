from module.searchspace.architectures.base_architecture_space import BaseArchitectureSpace
from module.utils.muon_track_dataset import MuonPose
from models import Encoder, Decoder, get_block_name
from collections import OrderedDict
import numpy as np
import sparseconvnet as scn
import torch
import os

class FeWrapper(torch.nn.Module):
    def __init__(self, encoder, feature_extractor, decoder):
        super(FeWrapper, self).__init__()
        self.encoder = encoder
        # we're not training the encoder
        self.encoder.freeze()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.sigmoid = scn.Sigmoid()

    def forward(self, primary_inputs, secondary_inputs):
        latent, blocks = self.encoder(primary_inputs, return_blocks=True)
        secondary_features = self.feature_extractor(secondary_inputs)
        blocks.append(secondary_features)
        prediction = self.decoder([latent, blocks[::-1]])
        prediction = self.sigmoid(prediction)
        return prediction

class FeatureExtractorSupermodel(BaseArchitectureSpace):
    '''
    Architecture space for feature extractors in Smoking Gun model search

    Objective is to learn feature extractors for secondary data sources that
    can be used to estimate primary data sources

    Child models are trained to produce the same latent encoding for secondary
    data as the one produced by autoencoder for primary data
    - training: produce same latent encoding for secondary data as autoencoder
                does on primary data
    - validation: measure loss of autoencoder reconstruction from child model's
                  latent encoding
    '''

    train = MuonPose("/home/jack/classes/thesis/autohas/LArCentroids/train/", return_energy=True)
    val = MuonPose("/home/jack/classes/thesis/autohas/LArCentroids/val/", return_energy=True)

    def __init__(self,
        encoder,
        encoder_features,
        N,
        latent_space_size,
        latent_space_channels,
        input_space_size=torch.LongTensor([128,128,128]),
        weight_directory='fesupermodel_weights/',
        epochs=15,
        device=None
        ):
        super(FeatureExtractorSupermodel, self).__init__()

        # encoder for training and testing feature extractor
        self.encoder = encoder
        self.encoder_features = encoder_features

        assert torch.all((input_space_size / 2**(N+1)) == latent_space_size), (input_space_size / 2**(N+1), latent_space_size)

        # child models must have this many layers
        self.N = N
        # child models must produce output this size
        self.latent_space_size = latent_space_size
        # child models must produce output with this many channels
        self.latent_space_channels = latent_space_channels

        # input image size
        self.input_space_size = input_space_size

        # where to save shared weights
        if weight_directory[-1] != '/':
            weight_directory += '/'
        if not os.path.isdir(weight_directory):
            os.mkdir(weight_directory)

        self.weight_directory = weight_directory

        # number of training epochs
        self.epochs = epochs

        self.device = device

        # possible values of dimensions in search space
        self.search_dimensions = OrderedDict()
        # number of repetitions of resnet blocks
        self.search_dimensions['reps'] =  [0, 1, 2, 3]
        # leakiness of leaky ReLU    
        self.search_dimensions['leakiness'] = [-0.01, 0., 0.01, 0.1]

        mean_feats_per_layer = latent_space_channels // N
        for i in range(self.N):
            x = i * mean_feats_per_layer
            features = [max(int(x + j * x), 1) for j in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]]
            # optionally select max or average pooling instead of a convolution
            features = [-2, -1] + features
            # number of output features of each layer
            self.search_dimensions[f'layer_{i}_features'] = features

        # track blocks we've already tried so we can reuse weights
        self.explored_blocks = OrderedDict()

    def get_child(self, state):
        assert type(state) == list, f"`state` must be a list, but {type(state)} was provided"
        assert len(state) == len(self.search_dimensions), f"`state` must contain {len(self.search_dimensions)} elements, but {len(state)} were provided"
        for choice, key in zip(state, self.search_dimensions):
            assert type(choice) == int, f"Entries of `state` must be `int`, but {type(choice)} was provided"
            assert 0 <= choice and choice < len(self.search_dimensions[key]), f"`choice` of {key} must be nonnegative and less than {len(self.search_dimensions[key])}, but was {choice}"

        state_ = OrderedDict()
        for choice, key in zip(state, self.search_dimensions):
            state_[key] = choice

        n_layers = []
        c_in = 1
        for i in range(self.N):
            key = f'layer_{i}_features'
            c_out = self.search_dimensions[key][state_[key]]
            if c_out > 0:
                n_layers.append((c_in, c_out))
            elif c_out == -1:
                n_layers.append((c_in, c_in, 'maxpool'))
            elif c_out == -2:
                n_layers.append((c_in, c_in, 'avgpool'))
            c_in = c_out if c_out > 0 else c_in


        # make sure we have desired number of output channels
        n_layers.append((c_in, self.latent_space_channels))
        reps = self.search_dimensions['reps'][state_['reps']]
        leakiness = self.search_dimensions['leakiness'][state_['leakiness']]
        child = Encoder(len(self.latent_space_size), reps, n_layers, leakiness, input_layer=self.input_space_size, name='fe')

        # check if any of these blocks have been seen before, and then load them if we have
        for block_name in child.block_names:
            if block_name in self.explored_blocks:
                state_dict = torch.load(self.weight_directory + block_name)
                child.blocks[child.block_names[block_name]].load_state_dict(state_dict)

        c_in = n_layers[-1][1] + self.encoder_features[-1][1]
        decoder_features = [(self.encoder_features[-1][0], c_in)]
        for i in range(len(self.encoder_features)):
            decoder_features.append((decoder_features[-1][0]//2, decoder_features[-1][1]//2))
        decoder_features[-1] = (decoder_features[-1][0], 1+decoder_features[-2][0], False)
        decoder = Decoder(len(self.latent_space_size), decoder_features, unet=True, use_sparsify=False, device=self.device)

        model = FeWrapper(self.encoder, child, decoder)
        return model

    def train_child(self, child, hyperparameters, indentation_level=0):
        '''Train child model using hyperparameters
        '''
        def print_(*args, **kwargs):
            if indentation_level != None and indentation_level > 0:
                print("\t"*indentation_level, *args, **kwargs)
            elif indentation_level == 0:
                print(*args, **kwargs)

        # move child to gpu for training
        child.to(self.device)

        child.train()
        loss_fn = torch.nn.MSELoss()

        optimizer_fn = hyperparameters['optimizer']
        parameters = [child.feature_extractor.parameters(), child.decoder.parameters()]
        parameters = [{'params': p} for p in parameters]
        optimizer = optimizer_fn(parameters)

        sparsifier = scn.InputLayer(len(self.input_space_size), self.input_space_size)
        densifier = scn.SparseToDense(len(self.latent_space_size), self.latent_space_channels)

        print_(f"Training child for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            print_()
            print_(f"Epoch {epoch}")
            total_loss = 0
            for i, (inputs, targets) in enumerate(self.train):
                inputs = [[torch.from_numpy(t).to(self.device) for t in j] for j in inputs]
                primary, secondary = inputs[0], inputs[1]
                primary[1] = primary[1].float().reshape(-1,1)
                secondary[1] = secondary[1].float().reshape(-1,1)

                predictions = child(primary, secondary)
                predictions = densifier(predictions)

                targets = [torch.from_numpy(t).to(self.device) for t in targets]
                targets[1] = targets[1].float().reshape(-1, 1)
                targets = sparsifier(targets)
                targets = densifier(targets)

                loss = loss_fn(predictions, targets)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    loss_str = np.format_float_scientific(loss.item(), precision=5)
                    print_(f"\t{100*i/len(self.train):.3f}% loss : {loss_str}", end='\r')
            total_loss /= len(self.train)
            print_(f"\tAverage loss : {total_loss:.3f}")
        print()

        # take off gpu when finished
        child.to('cpu')

    def evaluate_child(self, child, hyperparameters):
        '''Calculate validation signal for child using hyperparameters
        '''
        loss_fn = torch.nn.MSELoss()
        sparsifier = scn.InputLayer(len(self.input_space_size), self.input_space_size)        
        densifier = scn.SparseToDense(len(self.latent_space_size), self.latent_space_channels)

        # move to gpu for eval
        child.to(self.device)

        total_loss = 0
        for (inputs, targets) in self.val:
            inputs = [[torch.from_numpy(t).to(self.device) for t in j] for j in inputs]
            primary, secondary = inputs[0], inputs[1]
            primary[1] = primary[1].float().reshape(-1,1)
            secondary[1] = secondary[1].float().reshape(-1,1)

            predictions = child(primary, secondary)
            predictions = densifier(predictions)

            targets = [torch.from_numpy(t).to(self.device) for t in targets]
            targets[1] = targets[1].float().reshape(-1, 1)
            targets = sparsifier(targets)
            targets = densifier(targets)

            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

        # take off gpu when finished
        child.to('cpu')
        return total_loss / len(self.val)

    def get_reward_signal(self, child, hyperparameters):
        '''Calculate reward signal from validation loss
        '''
        signal = self.evaluate_child(child, hyperparameters)
        return signal

    def save_child(self, child):
        child = child.feature_extractor
        for layer_name in child.block_names:
            layer = child.blocks[child.block_names[layer_name]]
            torch.save(layer.state_dict(), self.weight_directory + layer_name)
            if layer_name not in self.explored_blocks:
                self.explored_blocks[layer_name] = 0
            self.explored_blocks[layer_name] += 1

    def __getitem__(self, i):
        return self.get_child(i)
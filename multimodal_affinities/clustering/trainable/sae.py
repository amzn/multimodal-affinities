from torch import nn


def weights_init(m):
    """
    changing the weights to a notmal distribution with mean=0 and std=0.01
    """
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.01)


class StackedAutoencoder(nn.Module):
    """
    model of autoencoder. this model allows to train each layer separately while freezing the other layers
    """
    def __init__(self, dims, dropout, layerwise_train=False):
        super(StackedAutoencoder, self).__init__()
        self.encoders = []
        self.decoders = []
        self.dropout = dropout
        self.init_stddev = 0.01

        self.encoders.append(nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU()
        ))
        self.decoders.append(nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(dims[1], dims[0]),
        ))
        if len(dims)>1:
            for i in range(1, len(dims)-1):
                if i == (len(dims)-2):
                    self.encoders.append(nn.Sequential(
                        nn.Dropout(self.dropout),
                        nn.Linear(dims[i], dims[i+1]),
                    ))
                else:
                    self.encoders.append(nn.Sequential(
                        nn.Dropout(self.dropout),
                        nn.Linear(dims[i], dims[i+1]),
                        nn.ReLU()
                    ))
                self.decoders.append(nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(dims[i+1], dims[i]),
                    nn.ReLU()
                ))

        self.n_layers = len(self.encoders)
        for i in range(0, self.n_layers):
            self.encoders[i].apply(weights_init)
            self.decoders[i].apply(weights_init)

        # the complete autoencoder
        self.encoder = nn.Sequential(*self.encoders)
        decoder = list(reversed(self.decoders))
        self.decoder = nn.Sequential(*decoder)
        #frozen layers and trainable layers
        self.frozen_encoder = []
        self.frozen_decoder = []
        self.training_encoder = None
        self.training_decoder = None
        self.trainable_params = None
        self.layerwise_train = layerwise_train

    def add_layer(self):
        """
        changing the current trainable layer to a frozen layer and the next layer to a trainable layer until
        there are no more layers to train
        """
        if self.training_encoder:
            # freezing the current trainable layer
            self.training_encoder.requires_grad = False
            self.frozen_encoder.append(self.training_encoder)
            self.training_decoder.requires_grad = False
            self.frozen_decoder.append(self.training_decoder)
        try:
            # adding a new layer to train
            self.training_encoder = self.encoders.pop(0)
            self.training_decoder = self.decoders.pop(0)
            self.trainable_params = [{'params': self.training_encoder.parameters()},
                                     {'params': self.training_decoder.parameters()}
                                     ]
        except:
            print('No more standby layers!')
            # update the complete autoencoder to include the trained layers
            self.encoder = nn.Sequential(*self.frozen_encoder)
            self.decoder = nn.Sequential(*list(reversed(self.frozen_decoder)))
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            for i in range(0, len(list(self.encoder))):
                self.encoder[i][0]._run_training_epoch(False)
                self.decoder[i][0]._run_training_epoch(False)

    def forward(self, x):
        if self.layerwise_train:
            # forward pass when training one layer at a time
            for e in self.frozen_encoder:
                x = e(x)
            encoded = self.training_encoder(x)
            decoded = self.training_decoder(encoded)
            for d in list(reversed(self.frozen_decoder)):
                decoded = d(decoded)
        else:
            #forward pass when training the entire autoencoder
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        return encoded, decoded

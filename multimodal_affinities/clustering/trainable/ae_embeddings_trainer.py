# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import torch.nn.functional as F
import torch
from torch.nn import MSELoss
from multimodal_affinities.clustering.trainable.contrastive_loss import ContrastiveLoss,\
    ContrastiveCrossEntropyLoss, ApproximatedContrastiveCrossEntropyLoss
from multimodal_affinities.clustering.trainable.siamese_sae import SiameseStackedAutoencoder
from multimodal_affinities.clustering.trainable.doc_entity_pairs_dataset import DocumentEntityPairsDataset

# Common epsilon val
eps = 1e-10


class EarlyStop(Exception):
    pass


class AEEmbeddingsTrainer:

    def __init__(self, logger, input_embedding_dim,
                 ae_dims=None,
                 dropout=0.2,
                 batch_size=32,
                 learning_rate=1e-3,
                 epochs=20,
                 early_stop=False,
                 gradient_clip=5,
                 push_pull_weight_ratio=3.0,
                 push_pull_decay=0.93,
                 reconstruction_loss_normalization=0.0,
                 normalize_embeddings_pre_process=False,
                 normalize_embeddings_post_process=True,
                 downsample_user_constraints=True,
                 downsample_default_constraints=True,
                 must_cannot_ratio=2,
                 font_word_mknn=3,
                 max_mustlink=400,
                 constraints_types=None
                 ):
        # -----------------------------------
        # - All network params are set here -
        # -----------------------------------
        self.ae_dims = [500, 500, 2000, 10] if not ae_dims else ae_dims
        self.dropout = dropout              # Probability for dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        print("num epochs %d" % epochs)
        self.epochs = epochs
        self.early_stop = early_stop        # True if should early stop when loss is too low
        self.gradient_clip = gradient_clip  # Max norm of the gradient
        self.push_pull_weight_ratio = push_pull_weight_ratio    # Amplifier for non-link pair loss compared to must-link
        self.push_pull_decay = push_pull_decay  # Decay rate for push_pull_weight_ratio for each epoch
        self.logger = logger
        self.must_cannot_ratio = must_cannot_ratio

        # Factor for reconstruction loss. Use 0 to disable component.
        self.reconstruction_loss_normalization = reconstruction_loss_normalization

        # Should normalize all embeddings at the beginning of training as a single batch
        # (is multi modal mode is on, normalize per embedding type)
        self.normalize_embeddings_pre_process = normalize_embeddings_pre_process

        # Should normalize all embeddings by the end of training as a single batch
        self.normalize_embeddings_post_process = normalize_embeddings_post_process

        # --------------------------------------------
        # - Network and trainer are initialized here -
        # --------------------------------------------
        self.net, self.single_entity_ae, self.is_multimodal_mode = self.initialize_network(input_embedding_dim)
        self.optimizer = self.initialize_optimizer(self.net, self.learning_rate)
        self.loss_history = []
        self.embeddings_history = []

        self.downsample_default_constraints = downsample_default_constraints
        self.downsample_user_constraints = downsample_user_constraints
        self.font_word_mknn = font_word_mknn
        self.max_mustlink = max_mustlink
        self.constraints_types = constraints_types

    def initialize_network(self, input_embedding_dim):
        if isinstance(input_embedding_dim, list):
            projected_dims = sum([proj_dim for _, proj_dim in input_embedding_dim])
            model_ae_dims = [projected_dims] + self.ae_dims
            net = SiameseStackedAutoencoder(model_ae_dims, self.dropout, projections_desc=input_embedding_dim)
            single_entity_ae = torch.nn.Sequential(net.projection_block, net.autoencoder)
            is_multimodal_mode = True
        else:
            model_ae_dims = [input_embedding_dim] + self.ae_dims
            net = SiameseStackedAutoencoder(model_ae_dims, self.dropout)
            single_entity_ae = torch.nn.Sequential(net.autoencoder)
            is_multimodal_mode = False
        if torch.cuda.is_available():
            net = net.cuda()

        return net, single_entity_ae, is_multimodal_mode

    def initialize_optimizer(self, net, learning_rate):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        return optimizer

    def initialize_loss_functions(self, push_pull_weight_ratio, push_pull_decay):
        pairwise_loss_func = ApproximatedContrastiveCrossEntropyLoss(push_pull_weight_ratio=push_pull_weight_ratio,
                                                                     push_pull_weight_decay=push_pull_decay)
        reconstruction_loss_func = MSELoss()
        return pairwise_loss_func, reconstruction_loss_func

    def project_embeddings(self, document):
        with torch.no_grad():
            entities = document.get_words()
            autoencoder = self.single_entity_ae
            word_to_embedding_unnormalized = {}
            word_to_embedding_normalized = {}
            for entity in entities:
                if self.is_multimodal_mode:
                    entity_embedding = entity.embedding['embeddings']
                    if torch.cuda.is_available():
                        entity_embedding = [embedding.cuda() for embedding in entity_embedding]
                    encoded, decoded = autoencoder(entity_embedding)
                else:
                    entity_embedding = entity.embedding.cuda() if torch.cuda.is_available() else entity.embedding
                    encoded, decoded = autoencoder(entity_embedding.unsqueeze(dim=0))
                encoded = encoded.squeeze()
                word_to_embedding_unnormalized[entity] = encoded if self.epochs > 0 else entity.embedding

            if self.normalize_embeddings_post_process:  # Record normalized embeddings as well
                embeddings_list = tuple(word_to_embedding_unnormalized.values())
                embeddings_batch = torch.stack(embeddings_list)
                embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
                for idx, normalized_embedding in enumerate(embeddings_batch):
                    entity = entities[idx]
                    word_to_embedding_normalized[entity] = normalized_embedding
        return {
            'normalized': word_to_embedding_normalized,
            'unnormalized': word_to_embedding_unnormalized
        }

    def _record_current_embeddings_state(self, document, pairwise_loss_func, save_training_progress):
        if save_training_progress:
            self.net.eval()
            current_embedding_state = self.project_embeddings(document)
            curr_push_pull_ratio = pairwise_loss_func.get_current_push_pull_ratio()
            current_embedding_state['push_pull_ratio'] = curr_push_pull_ratio
            self.embeddings_history.append(current_embedding_state)
            self.net.train()

    @staticmethod
    def to_cuda(cpu_tensor):
        try:
            return cpu_tensor.cuda()
        except AttributeError as e:
            return tuple([inner_tensor.cuda() for inner_tensor in cpu_tensor])

    def train_embeddings(self, document, embedding_level='words', save_training_progress=False,
                         constraints=None):
        """
        Starts a training session to refine the embeddings using a siamese auto-encoder.
        :param document: Document containing entities with initial embeddings
        :param embedding_level: 'words' or 'phrases
        :param save_training_progress: True if the embeddings projection should be saved each epoch (takes more time)
        :param constraints: Which constraints should the network use as labels. Any subset of:
        ['default-push', 'inter-phrase', 'mutual-knn', 'manual']
        :return: The network is trained, and the embeddings & unprojected_embeddings properties are updated for each
        document entity.
        """
        self.logger.info('[EntityClustering] :: Initiating SAE training session..')

        if self.normalize_embeddings_pre_process:
            entities = document.get_words() if embedding_level == 'words' else document.get_phrases()
            self.normalize_embeddings(entities)

        dataset = DocumentEntityPairsDataset(document=document,
                                             embedding_level=embedding_level,
                                             constraints=constraints,
                                             constraints_types=self.constraints_types,
                                             downsample_default=self.downsample_default_constraints,
                                             downsample_user=self.downsample_user_constraints,
                                             ratio_must_to_cannot=self.must_cannot_ratio,
                                             font_word_mknn=self.font_word_mknn,
                                             max_mustlink=self.max_mustlink)

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        # Reset push-pull weights decay
        pairwise_loss_func, reconstruction_loss_func = \
            self.initialize_loss_functions(self.push_pull_weight_ratio, self.push_pull_decay)

        min_loss = eps if self.early_stop else 0

        try:
            for epoch in range(self.epochs):
                self.logger.info('[EntityClustering] :: SAE training epoch #' + str(epoch + 1) + '..')
                self.net.train()
                for i_batch, (x1, x2, y, confidence) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        x1, x2, y, confidence = self.to_cuda(x1), self.to_cuda(x2), \
                                                self.to_cuda(y), self.to_cuda(confidence)
                    xproj1, xproj2, xe1, xe2, xd1, xd2 = self.net(x1, x2)  # encoded pair: projected, encoded, decoded
                    loss = pairwise_loss_func(xe1, xe2, y, confidence)
                    if self.reconstruction_loss_normalization > eps:
                        loss += reconstruction_loss_func(xproj1, xd1) * self.reconstruction_loss_normalization
                        loss += reconstruction_loss_func(xproj2, xd2) * self.reconstruction_loss_normalization
                    if (loss < min_loss).item():
                        raise EarlyStop
                    self.loss_history.append(loss.item())
                    loss.backward()  # backpropagation, compute gradients
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip)
                    self.optimizer.step()  # apply gradients
                pairwise_loss_func.weight_measurer.decay()
                self._record_current_embeddings_state(document, pairwise_loss_func, save_training_progress)
        except EarlyStop:
            self.logger.info('[EntityClustering] :: SAE training early stopped..')

        self.logger.info('[EntityClustering] :: Training complete - updating embeddings with final encoding..')
        if self.epochs > 0:
            self.update_document_with_projected_embeddings(document, embedding_level)

    def update_document_with_projected_embeddings(self, document, embedding_level='words'):
        with torch.no_grad():
            entities = document.get_words() if embedding_level == 'words' else document.get_phrases()
            autoencoder = self.single_entity_ae
            autoencoder.eval()
            for entity in entities:
                if self.is_multimodal_mode:
                    entity.unprojected_embedding = entity.embedding
                    entity_embedding = entity.embedding['embeddings']
                    encoded, decoded = autoencoder(entity_embedding)
                else:
                    entity.unprojected_embedding = entity.embedding
                    entity_embedding = entity.embedding.cuda() if torch.cuda.is_available() else entity.embedding
                    unprojected_embedding = entity_embedding.unsqueeze(dim=0)
                    encoded, decoded = autoencoder(unprojected_embedding)
                encoded = encoded.squeeze()
                entity.embedding = encoded if self.epochs > 0 else entity.embedding

            if self.normalize_embeddings_post_process:
                self.normalize_embeddings(entities)
        for entity in entities:
            entity.embedding = entity.embedding.cpu()

    @staticmethod
    def normalize_embeddings(entities):
        if len(entities) == 0:
            return
        sample_embedding = entities[0].embedding
        if isinstance(sample_embedding, dict):  # Normalize per embedding type
            embeddings_count = len(sample_embedding['embeddings'])
            entity_to_embedding = [[] for _ in entities]
            for embeddings_idx in range(embeddings_count):
                embeddings_list = [entity.embedding['embeddings'][embeddings_idx].squeeze() for entity in entities]
                embeddings_batch = torch.stack(embeddings_list)
                embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
                for entity_idx, embedding in enumerate(embeddings_batch):
                    entity_to_embedding[entity_idx].append(embedding.unsqueeze(dim=0))

            for entity_idx, entity in enumerate(entities):
                entity.embedding['embeddings'] = tuple(entity_to_embedding[entity_idx])

        else:   # Normalize all concatenated embeddings together
            embeddings_list = [entity.embedding for entity in entities]
            embeddings_batch = torch.stack(embeddings_list)
            embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
            for entity_idx, embedding in enumerate(embeddings_batch):
                entities[entity_idx].embedding = embedding

    def __getstate__(self):
        """
        :return: For pickling - avoid storing the logger state.
        """
        return dict((k, v) for (k, v) in self.__dict__.items() if k != 'logger')

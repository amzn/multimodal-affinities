# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import torch
from sklearn.decomposition import KernelPCA


class CombineEmbeddings:
    """
    A class for combining multi-model embeddings into a unified embedding, possibly of a lower dimension.
    """

    def __init__(self, strategy, **strategy_params):
        self.transformer = self.get_transformer(strategy, **strategy_params)

    @staticmethod
    def squeeze_and_concat(embeddings):
        """ Input: List of tuples (each entry in the tuple is a tensor of features). Each tuple is a sample.  """
        squeezed_embedding_per_sample = [torch.cat(embed_tuple, 1).squeeze() for embed_tuple in embeddings]
        return torch.stack(squeezed_embedding_per_sample).detach().cpu().numpy()   # Single tensor (samples x features)

    @staticmethod
    def get_transformer(strategy, **strategy_params):
        transformer = None

        if strategy == 'concat':
            transformer = CombineEmbeddings.squeeze_and_concat
        elif strategy == 'kernel_pca':
            n_components = strategy_params.get('combine_kernelpca_n_components', 10)
            kernel = strategy_params.get('combine_kernelpca_kernel', 'linear')
            kernel_pca = KernelPCA(n_components=n_components, kernel=kernel)
            transformer = lambda embeddings: kernel_pca.fit_transform(CombineEmbeddings.squeeze_and_concat(embeddings))
        elif strategy == 'mlp':
            def prepare_for_mlp(embeddings, embeddings_desc):
                dimensions_mapping = []
                for embedding_entry in embeddings_desc:
                    embedding_name, original_dim = embedding_entry
                    projected_dim = [proj_dim for proj_name, proj_dim in strategy_params.items() if embedding_name in proj_name]
                    if len(projected_dim) != 1:
                        raise ValueError('Incorrect configuration. Embedding type %r matched more than 1'
                                         'configurable projection parameter' % embedding_name)
                    projected_dim = projected_dim[0]
                    dimensions_mapping.append((original_dim, projected_dim))
                return [{'embeddings': embed_tuple,
                         'embeddings_desc': embeddings_desc,       # (embedding_type_str, original_dim)
                         'projections': dimensions_mapping}        # (original_dim, projected_dim)
                         for embed_tuple in embeddings]
            return prepare_for_mlp
        else:
            raise ValueError('Unsupported embeddings combination strategy: %r' % strategy)

        return lambda embeddings, embeddings_desc: torch.tensor(transformer(embeddings))

    def forward(self, embeddings, embeddings_desc):
        if torch.cuda.is_available():
            embeddings_gpu = [tuple(emb.cuda() for emb in entry) for entry in embeddings]
        else:
            embeddings_gpu = [tuple(emb for emb in entry) for entry in embeddings]
        return self.transformer(embeddings_gpu, embeddings_desc)

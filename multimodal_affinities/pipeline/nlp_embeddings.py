# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import torch
from flair.embeddings import CharacterEmbeddings, WordEmbeddings, \
    FlairEmbeddings, ELMoEmbeddings, BertEmbeddings, \
    DocumentPoolEmbeddings, DocumentLSTMEmbeddings, DocumentLMEmbeddings, StackedEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger

class NLPEmbedding:
    """ A class for converting word / phrase strings into NLP phrase / word vectors """

    def __init__(self,
                 use_character_embeddings, word_embedding_types,
                 flair_embedding_sources, bert_embedding_sources, elmo_embedding_sources,
                 sequence_encoder='stacked'):
        """
        Constructs a NLPEmbedding component.
        :param use_character_embeddings: Use character embeddings or not (True or False).
        :param word_embedding_types: List of any of 'glove', 'extvec', 'crawl', 'twitter' or two-letter language code.
        :param flair_embedding_sources: List of any of 'multi-forward', 'multi-backward', 'multi-forward-fast',
                'multi-backward-fast', 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param elmo_embedding_sources List of any of 'small', 'medium', 'original', 'pt'
        :param bert_embedding_sources List of any of 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased',
                'bert-large-cased', 'bert-base-multilingual-cased','bert-base-chinese'
        :param sequence_encoder: Any of: 'stacked', 'mean', 'max', 'min', 'lstm', 'charlm' (the last one is bugged
        due to flair).
        """

        # initialize the word & contextual language embeddings
        character_embeddings = [CharacterEmbeddings()] if use_character_embeddings else []
        word_embeddings = [WordEmbeddings(embedding_type) for embedding_type in word_embedding_types]
        flair_embeddings = [FlairEmbeddings(embedding_source) for embedding_source in flair_embedding_sources]
        elmo_embeddings = [ELMoEmbeddings(embedding_source) for embedding_source in elmo_embedding_sources]
        bert_embeddings = [BertEmbeddings(embedding_source) for embedding_source in bert_embedding_sources]

        # initialize the sequence encoder
        if sequence_encoder == 'stacked':
            self.sequence_embeddings = StackedEmbeddings([*character_embeddings,
                                                          *word_embeddings,
                                                          *flair_embeddings,
                                                          *elmo_embeddings,
                                                          *bert_embeddings])
        elif sequence_encoder in ('mean', 'max', 'min'):
            try:
                self.sequence_embeddings = DocumentPoolEmbeddings([*character_embeddings,
                                                                   *word_embeddings,
                                                                   *flair_embeddings,
                                                                   *elmo_embeddings,
                                                                   *bert_embeddings],
                                                                   pooling=sequence_encoder)
            except TypeError:
                # Support older versions of flair (0.4.0)
                self.sequence_embeddings = DocumentPoolEmbeddings([*character_embeddings,
                                                                   *word_embeddings,
                                                                   *flair_embeddings,
                                                                   *elmo_embeddings,
                                                                   *bert_embeddings],
                                                                   mode=sequence_encoder)
        elif sequence_encoder == 'lstm':
            self.sequence_embeddings = DocumentLSTMEmbeddings([*character_embeddings,
                                                               *word_embeddings,
                                                               *flair_embeddings,
                                                               *elmo_embeddings,
                                                               *bert_embeddings])
        elif sequence_encoder == 'charlm':
            self.sequence_embeddings = DocumentLMEmbeddings([*flair_embeddings])
        else:
            raise ValueError('invalid sequence_encoder for NLPEmbedding')

        if torch.cuda.is_available():
            self.sequence_embeddings.cuda()

        self.tagger = SequenceTagger.load('ner-ontonotes')


    def forward(self, phrase):
        '''
        :param phrase: Input str of a phrase
        :return: Single tensor containing the phrase embedding
        '''
        # create an example sentence
        sentence = Sentence(phrase)

        # embed the sentence with our document embedding
        self.sequence_embeddings.embed(sentence)

        if isinstance(self.sequence_embeddings, StackedEmbeddings):
            # Return embedding per word
            embeddings = {}
            for token in sentence:
                embeddings[token.text] = token.embedding
        else:
            # Return pooled embedding for entire phrase
            embeddings = sentence.get_embedding()
            if len(embeddings.shape) < 2:
                embeddings = embeddings.unsqueeze(0)
        self.tagger.predict(sentence)
        return embeddings, sentence.to_tagged_string()

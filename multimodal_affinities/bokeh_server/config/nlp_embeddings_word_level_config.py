from bokeh.models.widgets import MultiSelect, CheckboxGroup, Select

nlp_word_embeddings_config = {

    "embedding_name": 'NLP Word Embeddings',
    "is_editable": False,
    "parameters": {
        'character_embeddings': {
            'type': CheckboxGroup,
            'labels': ['character_embeddings'],
            'active': []
        },
        'word_embeddings': {
            'type': MultiSelect,
            'title': 'word_embeddings',
            'options': [
                ("glove", "GloVe"),
                ("crawl", "FastText: crawl"),
                ("news", "FastText: news"),
                ("extvec", "Komnios"),
                ("twitter", "Twitter")
            ],
            'value': ["glove"]
        },
        'contextual_flair_embeddings': {
            'type': MultiSelect,
            'title': 'contextual_flair_embeddings',
            'options': [
                ('news-forward', 'Flair: news-forward'),
                ('news-backward', 'Flair: news-backward'),
                ('news-forward-fast', 'Flair: news-forward-fast'),
                ('news-backward-fast', 'Flair: news-backward-fast'),
                ('multi-forward', 'Flair: multi-forward'),
                ('multi-backward', 'Flair: multi-backward'),
                ('multi-forward-fast', 'Flair: multi-forward-fast'),
                ('multi-backward-fast', 'Flair: multi-backward-fast'),
                ('mix-forward', 'Flair: mix-forward'),
                ('mix-backward', 'Flair: mix-backward')
            ],
            'value': ["news-forward", "news-backward"]
        },
        'contextual_elmo_embeddings': {
            'type': MultiSelect,
            'title': 'contextual_elmo_embeddings',
            'options': [
                ('small', 'ELMo: small'),
                ('medium', 'ELMo: medium'),
                ('original', 'ELMo: original'),
                ('pt', 'ELMo: portuguese')
            ],
            'value': []
        },
        'contextual_bert_embeddings': {
            'type': MultiSelect,
            'title': 'contextual_bert_embeddings',
            'options': [
                ('bert-base-uncased', 'BERT: base-uncased'),
                ('bert-large-uncased', 'BERT: large-uncased'),
                ('bert-base-cased', 'BERT: base-cased'),
                ('bert-large-cased', 'BERT: large-cased'),
                ('bert-base-multilingual-cased', 'BERT: base-multilingual-cased'),
                ('bert-base-chinese', 'BERT: base-chinese')
            ],
            'value': []
        }
    }
}

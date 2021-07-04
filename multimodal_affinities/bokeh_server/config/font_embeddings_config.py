from bokeh.models.widgets import MultiSelect, CheckboxGroup, Select, TextInput
from multimodal_affinities.common_config import FONT_EMBEDDING_MODEL

font_embeddings_config = {

    "embedding_name": 'Font Embeddings',
    "is_editable": True,
    "parameters": {
        'word_level_embedding': {
            'type': CheckboxGroup,
            'labels': ['word_level_embedding'],
            'active': [0]
        },
        'phrase_level_embedding': {
            'type': CheckboxGroup,
            'labels': ['phrase_level_embedding'],
            'active': []
        },
        'trained_model_file': {
            'type': TextInput,
            'value': FONT_EMBEDDING_MODEL,
            'title': 'trained_model_file',
        },
    }
}

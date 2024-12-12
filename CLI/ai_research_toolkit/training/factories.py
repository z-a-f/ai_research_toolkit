from ai_research_toolkit.training.models.base_model import BaseModel
from ai_research_toolkit.training.models.example_transformer import TransformerModel

from ai_research_toolkit.training.datamodules.base_datamodule import BaseDataModule
# You can import your custom data modules here as they grow.

def get_model(config: dict) -> BaseModel:
    model_type = config['model']['type']
    if model_type == 'transformer':
        return TransformerModel(
            model_name=config['model']['name'],
            task=config['model'].get('task', 'classification'),
            num_labels=config['model'].get('num_labels', 2),
            learning_rate=config['training']['learning_rate'],
            pretrained_checkpoint=config['model'].get('pretrained_checkpoint', None)
        )
    # You can add more model types here
    else:
        raise ValueError(f"Model type {model_type} not recognized.")

def get_datamodule(config: dict) -> BaseDataModule:
    data_type = config['data'].pop('type', None)
    if data_type == 'huggingface':
        # Implement a huggingface-based datamodule for text classification
        from ai_research_toolkit.training.datamodules.base_datamodule import HFTextClassificationDataModule
        return HFTextClassificationDataModule(**config['data'])
    # Add more data module types as needed
    else:
        raise ValueError(f"Data type {data_type} not recognized.")

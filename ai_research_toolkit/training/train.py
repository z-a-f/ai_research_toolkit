import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ai_research_toolkit.training.factories import get_model, get_datamodule
from ai_research_toolkit.utils.logging_utils import get_logger

def run_training(config):
    logger = get_logger(__name__)
    model = get_model(config)
    datamodule = get_datamodule(config)
    # Set the tokenizer for the datamodule if the model has one
    if hasattr(model, 'tokenizer'):
        datamodule.set_tokenizer(model.tokenizer)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['output_dir'],
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename='best-checkpoint'
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if config['training']['gpus'] > 0 else 'cpu',
        devices=config['training']['gpus'] if config['training']['gpus'] > 0 else None,
        precision=config['training']['precision'],
        callbacks=[checkpoint_callback]
    )

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Training completed.")

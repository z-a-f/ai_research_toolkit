import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from ai_research_toolkit.training.factories import get_model, get_datamodule
from ai_research_toolkit.utils.logging_utils import get_logger

def run_finetuning(config):
    logger = get_logger(__name__)
    model = get_model(config)
    datamodule = get_datamodule(config)
    if hasattr(model, 'tokenizer'):
        datamodule.set_tokenizer(model.tokenizer)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['output_dir'],
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename='finetuned-best-checkpoint'
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision'],
        callbacks=[checkpoint_callback]
    )

    logger.info("Starting fine-tuning...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Fine-tuning completed.")

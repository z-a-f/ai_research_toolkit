import click
from ai_research_toolkit.utils.config_parser import load_config
from ai_research_toolkit.training.train import run_training
from ai_research_toolkit.training.finetune import run_finetuning

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', required=True, help='Path to the training config file.')
def train(config):
    cfg = load_config(config)
    run_training(cfg)

@cli.command()
@click.option('--config', required=True, help='Path to the finetuning config file.')
def finetune(config):
    cfg = load_config(config)
    run_finetuning(cfg)

if __name__ == "__main__":
    cli()

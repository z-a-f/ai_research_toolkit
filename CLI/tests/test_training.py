import os
import shutil
import pytest

from ai_research_toolkit.utils.config_parser import load_config
from ai_research_toolkit.training.train import run_training
from ai_research_toolkit.training.finetune import run_finetuning

@pytest.fixture(scope='module')
def cleanup_checkpoints():
    # Clean up checkpoint directory before and after tests
    train_ckpt_dir = './test_checkpoints/'
    finetune_ckpt_dir = './test_finetune_checkpoints/'
    if os.path.exists(train_ckpt_dir):
        shutil.rmtree(train_ckpt_dir)
    if os.path.exists(finetune_ckpt_dir):
        shutil.rmtree(finetune_ckpt_dir)
    yield
    if os.path.exists(train_ckpt_dir):
        shutil.rmtree(train_ckpt_dir)
    if os.path.exists(finetune_ckpt_dir):
        shutil.rmtree(finetune_ckpt_dir)

def test_run_training(cleanup_checkpoints):
    # Load a minimal config
    config_path = 'tests/configs/test_training_config.yaml'
    cfg = load_config(config_path)

    # Run training
    run_training(cfg)

    # Check that checkpoints are created
    ckpt_dir = cfg['training']['output_dir']
    assert os.path.exists(ckpt_dir), "Checkpoint directory not created."
    files = os.listdir(ckpt_dir)
    assert any('best-checkpoint' in f for f in files), "No checkpoint file found after training."

def test_run_finetuning(cleanup_checkpoints):
    # For finetuning test, we assume we have a pretrained checkpoint from the previous test.
    # In a full CI pipeline, we'd run training first or mock a checkpoint.
    # Here, let's just assume the training test ran first. In a real scenario,
    # we might run tests in order or mock a checkpoint file.

    # Check if a checkpoint file from training exists
    # If not, skip this test or create a mock checkpoint.
    training_ckpt_dir = './test_checkpoints/'
    if not os.path.exists(training_ckpt_dir):
        pytest.skip("No training checkpoint found. Run training test first.")

    # Ensure there's a best checkpoint
    ckpt_files = [f for f in os.listdir(training_ckpt_dir) if 'best-checkpoint' in f]
    if not ckpt_files:
        pytest.skip("No best checkpoint found from training. Cannot finetune.")

    best_ckpt_path = os.path.join(training_ckpt_dir, ckpt_files[0])
    
    # Load finetuning config and inject the existing checkpoint path
    config_path = 'tests/configs/test_finetuning_config.yaml'
    cfg = load_config(config_path)
    cfg['model']['pretrained_checkpoint'] = best_ckpt_path

    # Run finetuning
    run_finetuning(cfg)

    # Check that checkpoints are created for finetuning
    ckpt_dir = cfg['training']['output_dir']
    assert os.path.exists(ckpt_dir), "Finetune checkpoint directory not created."
    files = os.listdir(ckpt_dir)
    assert any('finetuned-best-checkpoint' in f for f in files), "No finetune checkpoint file found."

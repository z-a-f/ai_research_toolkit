# AI Research Toolkit

> [!NOTE]
> * This is an initial version of the `AI Research Toolkit`.
> * The setup instructions (such as `pyproject.toml` and `setup.py`) will be completed once the core functionality is stable.
> * This repository outlines the structure and demonstrates initial functionality for training and fine-tuning models via a CLI.

## Overview

The **AI Research Toolkit** provides a modular, configuration-driven interface for common AI research workflows. Initially, it focuses on:

- **Training** models from scratch or from pretrained weights.
- **Fine-tuning** pretrained models on downstream tasks.

The toolkit is designed to be easily extended to privacy-related research tools and model optimization methods in the future.

## Key Features

- **Configuration-Driven:** Experiment parameters (models, datasets, hyperparameters) are defined in YAML configs, minimizing the need to modify code.
- **Model & Data Agnosticism:** A factory pattern allows plugging in arbitrary models and data modules, as long as they follow the defined interface.
- **PyTorch Lightning & Hugging Face Integration:** Rapid development of training loops and fine-tuning routines on top of industry-standard tools.
- **CLI Interface:** Run all operations from the command line. For example:
  ```bash
  ai_research_toolkit train --config=path/to/training_config.yaml
  ai_research_toolkit finetune --config=path/to/finetuning_config.yaml
  ```
  
## Current Directory Structure

```
ai_research_toolkit/
    ├─ ai_research_toolkit/
    │   ├─ __init__.py
    │   ├─ cli.py
    │   ├─ configs/
    │   │   ├─ default_training.yaml
    │   │   └─ default_finetuning.yaml
    │   ├─ training/
    │   │   ├─ __init__.py
    │   │   ├─ train.py
    │   │   ├─ finetune.py
    │   │   ├─ factories.py
    │   │   ├─ datamodules/
    │   │   │   ├─ __init__.py
    │   │   │   └─ base_datamodule.py
    │   │   └─ models/
    │   │       ├─ __init__.py
    │   │       ├─ base_model.py
    │   │       └─ example_transformer.py
    │   └─ utils/
    │       ├─ config_parser.py
    │       ├─ logging_utils.py
    │       └─ cli_helpers.py (placeholder for future use)
    └─ tests/
        ├─ test_training.py
        └─ test_finetuning.py
```

## Getting Started (Preliminary)

**Note:** Package installation is not yet finalized. Once `pyproject.toml` and `setup.py` are configured, installation will be as simple as:

```bash
pip install -e .
```

For now, you can run directly from the source directory after setting your Python path appropriately.

1. **Set Up Environment:**
   - Create a virtual environment (recommended):
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Install requirements (to be listed later; likely `lightning`, `transformers`, `datasets`, `pyyaml`).
   
2. **Run Training:**
   ```bash
   python ai_research_toolkit/cli.py train --config=ai_research_toolkit/configs/default_training.yaml
   ```

3. **Run Fine-Tuning:**
   ```bash
   python ai_research_toolkit/cli.py finetune --config=ai_research_toolkit/configs/default_finetuning.yaml
   ```

## Configuration

Configurations are YAML files specifying model type, dataset, and training parameters. A basic training config (`default_training.yaml`) might look like this:

```yaml
model:
  type: "transformer"
  name: "bert-base-uncased"
  task: "classification"
  num_labels: 2

data:
  type: "huggingface"
  dataset_name: "imdb"
  split: "train[:80%]"
  val_split: "train[80%:]"
  batch_size: 32
  max_length: 128

training:
  max_epochs: 3
  gpus: 1
  learning_rate: 3e-5
  precision: 16
  output_dir: "./checkpoints/"
```

**Changing Models or Datasets:** Modify the `model` or `data` sections of the config to easily switch between different pretrained models or datasets.

## Next Steps

- **Stabilize the Core:** Finalize the current modules and ensure robust testing.
- **Add Setup & PyProject:** Once stable, add the required packaging files and CI workflows.
- **Extend Functionality:** Integrate privacy tools, optimization modules (quantization, pruning, etc.), and a broader set of data and model types.

## License

*(To be decided)*

## Contributions

Contributions are welcome. Until the codebase stabilizes, please discuss major changes via issues or proposals before submitting PRs.

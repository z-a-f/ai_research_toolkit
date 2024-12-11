import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .base_model import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, model_name: str, task: str, num_labels: int, learning_rate: float, pretrained_checkpoint: str = None):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_checkpoint'])
        self.model_name = model_name
        self.task = task
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if pretrained_checkpoint is not None:
            # Load state dict if provided
            ckpt = torch.load(pretrained_checkpoint, map_location='cpu')
            self.load_state_dict(ckpt['state_dict'])

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch['labels']).float().mean()
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # Optionally add schedulers here
        return optimizer

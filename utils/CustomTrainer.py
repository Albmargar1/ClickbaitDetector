import torch
from transformers import Trainer

class CustomTrainer(Trainer):
    def set_loss_weights(self, weights):
      self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = torch.nn.CrossEntropyLoss(
          weight=torch.tensor(self.weights).to(self.args.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
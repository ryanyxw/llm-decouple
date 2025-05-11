from torchmetrics import Metric
import torch
from torch import Tensor

class SelectivePerplexity(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_tokens", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, batch: Tensor, logits: Tensor) -> None:
        # shift according to labels
        loss_labels = batch["label_mask"][..., 1:]

        #make sure that all the values in loss-labels is either 0 or 1
        assert(torch.all((loss_labels == 0) | (loss_labels == 1)).item())

        # Note that we assume 1 for important tokens and 0 for unimportant tokens
        def calculate_loss_across_tokens(logits, labels, shift=False):
            from torch.nn.functional import cross_entropy
            if (shift):
                logits = logits[..., :-1, :]
                labels = labels[..., 1:]
            new_logits = logits.reshape(-1, logits.shape[-1])
            new_labels = labels.reshape(-1)
            cross = cross_entropy(new_logits, new_labels, reduction="none").reshape(logits.shape[:-1])
            return cross

        self_calculated_loss = calculate_loss_across_tokens(logits, batch["input_ids"], shift=True)

        # we select the loss from the important tokens
        selected_tokens = self_calculated_loss[loss_labels == 1]

        self.loss += torch.sum(selected_tokens).long()
        self.num_tokens += torch.sum(loss_labels).long()

    def compute(self) -> Tensor:
        return torch.exp(self.loss.float() / self.num_tokens)
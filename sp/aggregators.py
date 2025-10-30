from collections import OrderedDict
from collections import defaultdict

import lightning.pytorch as pl
import torch


class MetricAggregator(pl.LightningModule):
    """Computes an average of a pre-determined length. Useful for logging average losses."""

    def __init__(self):
        super().__init__()

        self.counts = defaultdict(self.zero)
        self.values = defaultdict(self.zero)
        self.size = None

    def forward(self, key: str, value: int | float):
        size = self.get_or_set_size()
        count = self.counts[key]
        aggregated_value = self.values[key] + value
        if count + 1 == size:
            self.counts[key] = 0
            self.values[key] = 0
            averaged_value = aggregated_value / size
        else:
            next_count = count + 1
            self.counts[key] = next_count
            self.values[key] = aggregated_value
            averaged_value = aggregated_value / next_count

        return key, averaged_value

    def get_or_set_size(self):
        if self.size is None:
            self.size = self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches
        return self.size

    def zero(self):
        return 0

    def state_dict(self, destination: dict | None = None, prefix: str = "", keep_vars: bool = False):
        if destination is None:
            destination = OrderedDict()

        destination[prefix + "counts"] = self.counts
        destination[prefix + "values"] = self.values

        return destination

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

        counts_key = prefix + "counts"
        if counts_key in state_dict:
            self.counts.update(state_dict[counts_key])
            unexpected_keys.remove(counts_key)
        else:
            missing_keys.append(counts_key)

        values_key = prefix + "values"
        if values_key in state_dict:
            self.values.update(state_dict[values_key])
            unexpected_keys.remove(values_key)
        else:
            missing_keys.append(values_key)


class MeanVoter:
    """Aggregate predictions from the same source by averaging them."""

    def __init__(self):
        super().__init__()
        self.predictions = defaultdict(list)
        self.labels = {}

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        output_labels = []
        output_predictions = []
        for key in self.labels.keys():
            predictions = self.predictions[key]
            predictions = torch.stack(predictions, dim=0)
            predictions = predictions.mean(dim=0).argmax(dim=-1)
            output_predictions.append(predictions)
            output_labels.append(self.labels[key])
        output_predictions = torch.stack(output_predictions, dim=0)
        output_labels = torch.stack(output_labels, dim=0)
        return output_predictions, output_labels

    def reset(self):
        self.predictions = defaultdict(list)
        self.labels = defaultdict(list)

    def update(self, keys: list[str], predictions: torch.Tensor, labels: torch.Tensor):
        for i, key in enumerate(keys):
            self.predictions[key].append(predictions[i])
            self.labels[key] = labels[i]

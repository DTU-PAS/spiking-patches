import numpy as np
import torch
import torch_geometric
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.pool import radius_graph

from sp.configs import Config
from sp.configs import Model
from sp.data_types import ClassificationBatch
from sp.data_types import ClassificationEventsData
from sp.data_types import ObjectDetectionBatch
from sp.data_types import ObjectDetectionTokensData
from sp.data_types import Tokens
from sp.data_types import TokensBatch
from sp.representations.volume import batched_events_to_logspace_volume
from sp.timers import Timer
from sp.tokenizer import BatchTokenizer


class ClassificationEventGraphCollator:
    def __init__(self, config: Config):
        self.loop = config.gnn.loop
        self.max_num_neighbors = config.gnn.max_num_neighbors
        self.radius = config.gnn.node_radius
        self.time_scale = config.time_scale

    @Timer("collate_classification_event_graph")
    def __call__(self, batch: list[ClassificationEventsData]) -> ClassificationBatch:
        batch_size = len(batch)

        batch_ids = [torch.full((element.events.x.shape[0],), i) for i, element in enumerate(batch)]
        batch_ids = torch.cat(batch_ids, dim=0)

        x = np.concatenate([element.events.x for element in batch], axis=0)
        y = np.concatenate([element.events.y for element in batch], axis=0)
        t = np.concatenate([element.events.t for element in batch], axis=0)
        t = t / self.time_scale
        positions = np.stack([x, y, t], axis=1)
        positions = torch.tensor(positions)
        edge_index = radius_graph(
            positions,
            batch=batch_ids,
            batch_size=batch_size,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
            r=self.radius,
        )

        p = np.concatenate([element.events.p for element in batch], axis=0)
        node_features = torch.tensor(p).unsqueeze(1)

        graphs = torch_geometric.data.Batch(
            batch=batch_ids,
            edge_index=edge_index,
            pos=positions,
            x=node_features,
        )

        labels = torch.stack([element.label for element in batch])
        ids = [element.id for element in batch]

        return ClassificationBatch(inputs=graphs, batch_size=batch_size, labels=labels, ids=ids)


class ClassificationEventPointCollator:
    def __init__(self, config: Config):
        self.time_scale = config.time_scale

    @Timer("collate_classification_event_point")
    def __call__(self, batch: list[ClassificationEventsData]) -> ClassificationBatch:
        batch_ids = [torch.full((element.events.x.shape[0],), i) for i, element in enumerate(batch)]
        batch_ids = torch.cat(batch_ids, dim=0)

        x = np.concatenate([element.events.x for element in batch], axis=0)
        y = np.concatenate([element.events.y for element in batch], axis=0)
        t = np.concatenate([element.events.t for element in batch], axis=0)
        t = t / self.time_scale
        # should be safe to cast since we have divided by time_scale and also subtracted minimum time in a previous step
        t = t.astype(np.float32)
        positions = np.stack([x, y, t], axis=1)
        positions = torch.tensor(positions)

        p = np.concatenate([element.events.p for element in batch], axis=0)
        p = torch.tensor(p)

        inputs = torch_geometric.data.Batch(
            batch=batch_ids,
            x=p,
            pos=positions,
        )

        return ClassificationBatch(
            inputs=inputs,
            batch_size=len(batch),
            labels=torch.stack([element.label for element in batch]),
            ids=[element.id for element in batch],
        )


class ClassificationTokenPointCollator:
    def __init__(self, config: Config):
        self.buckets = config.buckets
        self.patch_size = config.patch_size
        self.time_scale = config.time_scale
        self.tokenizer = BatchTokenizer(config)

    @Timer("collate_classification_token_point")
    def __call__(self, batch: list[ClassificationEventsData]) -> ClassificationBatch | None:
        batch_size = len(batch)

        events = [sequence.events for sequence in batch]
        with Timer("tokenization"):
            tokens = self.tokenizer(events)

        batch_ids = []
        x = []
        y = []
        t = []
        node_features = []
        labels = []
        ids = []

        batch_id = 0
        for i, sequence in enumerate(tokens):
            # Skip empty sequences
            if sequence.pos_x.shape[0] == 0:
                continue

            batch_ids.append(torch.full((sequence.pos_x.shape[0],), batch_id))
            x.append(sequence.pos_x)
            y.append(sequence.pos_y)
            t.append(sequence.pos_t)
            labels.append(batch[i].label)
            ids.append(batch[i].id)

            indices = [np.full_like(value, j) for j, value in enumerate(sequence.events_x)]
            inputs = batched_events_to_logspace_volume(
                batch=indices,
                x=sequence.events_x,
                y=sequence.events_y,
                t=sequence.events_t,
                p=sequence.events_p,
                buckets=self.buckets,
                height=self.patch_size,
                width=self.patch_size,
            )
            inputs = torch.tensor(inputs, dtype=torch.int64)
            node_features.append(inputs)

            batch_id += 1

        if len(batch_ids) == 0:
            return None

        batch_ids = torch.cat(batch_ids, dim=0)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        t = np.concatenate(t, axis=0)
        t = t / self.time_scale
        # should be safe to cast since we have divided by time_scale and also subtracted minimum time in a previous step
        t = t.astype(np.float32)
        positions = np.stack([x, y, t], axis=1)
        positions = torch.tensor(positions)

        node_features = torch.cat(node_features, dim=0)

        graphs = torch_geometric.data.Batch(
            batch=batch_ids,
            pos=positions,
            x=node_features,
        )

        labels = torch.stack(labels)

        return ClassificationBatch(inputs=graphs, batch_size=batch_size, labels=labels, ids=ids)


class ClassificationTokenGraphCollator:
    def __init__(self, config: Config):
        self.buckets = config.buckets
        self.loop = config.gnn.loop
        self.max_num_neighbors = config.gnn.max_num_neighbors
        self.patch_size = config.patch_size
        self.radius = config.gnn.node_radius
        self.time_scale = config.time_scale
        self.tokenizer = BatchTokenizer(config)

    @Timer("collate_classification_token_graph")
    def __call__(self, batch: list[ClassificationEventsData]) -> ClassificationBatch | None:
        batch_size = len(batch)

        events = [sequence.events for sequence in batch]
        with Timer("tokenization"):
            tokens = self.tokenizer(events)

        batch_ids = []
        x = []
        y = []
        t = []
        node_features = []
        labels = []
        ids = []

        batch_id = 0
        for i, sequence in enumerate(tokens):
            # Skip empty sequences
            if sequence.pos_x.shape[0] == 0:
                continue

            batch_ids.append(torch.full((sequence.pos_x.shape[0],), batch_id))
            x.append(sequence.pos_x)
            y.append(sequence.pos_y)
            t.append(sequence.pos_t)
            labels.append(batch[i].label)
            ids.append(batch[i].id)

            indices = [np.full_like(value, j) for j, value in enumerate(sequence.events_x)]
            inputs = batched_events_to_logspace_volume(
                batch=indices,
                x=sequence.events_x,
                y=sequence.events_y,
                t=sequence.events_t,
                p=sequence.events_p,
                buckets=self.buckets,
                height=self.patch_size,
                width=self.patch_size,
            )
            inputs = torch.tensor(inputs, dtype=torch.int64)
            node_features.append(inputs)

            batch_id += 1

        if len(batch_ids) == 0:
            return None

        batch_ids = torch.cat(batch_ids, dim=0)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        t = np.concatenate(t, axis=0)
        t = t / self.time_scale
        positions = np.stack([x, y, t], axis=1)
        positions = torch.tensor(positions)
        edge_index = radius_graph(
            positions,
            batch=batch_ids,
            batch_size=batch_size,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
            r=self.radius,
        )

        node_features = torch.cat(node_features, dim=0)

        graphs = torch_geometric.data.Batch(
            batch=batch_ids,
            edge_index=edge_index,
            pos=positions,
            x=node_features,
        )

        labels = torch.stack(labels)

        return ClassificationBatch(inputs=graphs, batch_size=batch_size, labels=labels, ids=ids)


class ClassificationTokenCollator:
    def __init__(self, config: Config):
        self.tokenizer = BatchTokenizer(config)
        self.buckets = config.buckets
        self.patch_size = config.patch_size
        self.reverse_time = config.reverse_time

    @Timer("collate_classification_token")
    def __call__(self, batch: list[ClassificationEventsData]) -> ClassificationBatch:
        events = [sequence.events for sequence in batch]
        with Timer("tokenization"):
            tokens = self.tokenizer(events)
        tokens = collate_tokens(tokens, self.buckets, self.patch_size, self.patch_size, self.reverse_time)
        labels = torch.stack([element.label for element in batch])
        ids = [element.id for element in batch]
        return ClassificationBatch(inputs=tokens, batch_size=len(batch), labels=labels, ids=ids)


class ObjectDetectionCollator:
    def __init__(self, config: Config):
        self.tokenizer = BatchTokenizer(config)
        self.buckets = config.buckets
        self.patch_size = config.patch_size
        self.reverse_time = config.reverse_time

        if config.model == Model.transformer:
            self.collate_inputs = self.collate_tokens
        elif config.model in (Model.gnn, Model.pcn):
            self.collate_inputs = self.collate_geometric_data
        else:
            raise ValueError(f"Unsupported model {config.model}")

    @Timer("collate_object_detection_tokens")
    def __call__(self, batch: list[ObjectDetectionTokensData]) -> ObjectDetectionBatch:
        has_batch_indices = [sequence.batch_index is not None for sequence in batch]
        if not any(has_batch_indices):
            return self.collate_random_access(batch)
        elif all(has_batch_indices):
            return self.collate_streaming(batch)
        else:
            raise ValueError("Collator is used in a mixed mode, but it should be either streaming or random access.")

    def collate_random_access(self, batch: list[ObjectDetectionTokensData]) -> ObjectDetectionBatch:
        batch_indices = [batch_index for batch_index, sequence in enumerate(batch) for _ in sequence.inputs]
        batch_indices = torch.tensor(batch_indices)
        sequence_lengths = torch.tensor([len(sequence.inputs) for sequence in batch])
        inputs = [values for sequence in batch for values in sequence.inputs]
        inputs = self.collate_inputs(inputs)
        has_labels = [labels is not None for sequence in batch for labels in sequence.yolox_labels]
        has_labels = torch.tensor(has_labels)
        prediction_times = torch.tensor([time for sequence in batch for time in sequence.prediction_time])
        prophesee_labels = [labels for sequence in batch for labels in sequence.prophesee_labels]
        reset = torch.tensor([reset for sequence in batch for reset in sequence.reset], dtype=torch.bool)
        sequence_ids = [sequence_id for sequence in batch for sequence_id in sequence.sequence_id]
        yolox_labels = [labels for sequence in batch for labels in sequence.yolox_labels]
        assert len(set([element.worker_id for element in batch])) == 1, (
            "All elements in the batch must have the same worker_id."
        )
        return ObjectDetectionBatch(
            batch_indices=batch_indices,
            inputs=inputs,
            has_labels=has_labels,
            prediction_times=prediction_times,
            prophesee_labels=prophesee_labels,
            reset=reset,
            sequence_ids=sequence_ids,
            sequence_lengths=sequence_lengths,
            yolox_labels=yolox_labels,
            worker_id=batch[0].worker_id,  # All sequences have the same worker_id
        )

    def collate_streaming(self, batch: list[ObjectDetectionTokensData]) -> ObjectDetectionBatch:
        """Collate tokens in streaming mode.

        We may find multiple subsequences from the same sequence. These need to be concatenated together.
        This issue is irrelevant in random access mode as RNN states are not saved in this mode.
        Note that the subsequences are guaranteed to be consecutive and non-overlapping.
        """

        unique_sequence_ids = set([sequence_id for sequence in batch for sequence_id in sequence.sequence_id])

        collated_batch_indices = []
        collated_sequence_lengths = []
        collated_inputs = []
        collated_prediction_times = []
        collated_prophesee_labels = []
        collated_reset = []
        collated_sequence_ids = []
        collated_yolox_labels = []

        for sequence_id in unique_sequence_ids:
            subbatch = [b for b in batch if b.sequence_id[0] == sequence_id]
            assert set([sid for sequence in subbatch for sid in sequence.sequence_id]) == {sequence_id}
            sequence_length = sum([len(b.inputs) for b in subbatch])
            collated_sequence_lengths.append(sequence_length)
            batch_indices = [batch_index for sequence in subbatch for batch_index in sequence.batch_index]
            assert len(set(batch_indices)) == 1
            collated_batch_indices.extend(batch_indices)
            collated_inputs.extend([values for sequence in subbatch for values in sequence.inputs])
            collated_prediction_times.extend([time for sequence in subbatch for time in sequence.prediction_time])
            collated_prophesee_labels.extend([labels for sequence in subbatch for labels in sequence.prophesee_labels])
            collated_reset.extend([reset for sequence in subbatch for reset in sequence.reset])
            collated_sequence_ids.extend([sequence_id] * sequence_length)
            collated_yolox_labels.extend([labels for sequence in subbatch for labels in sequence.yolox_labels])

        batch_indices = torch.tensor(collated_batch_indices)
        sequence_lengths = torch.tensor(collated_sequence_lengths)
        inputs = self.collate_inputs(collated_inputs)
        has_labels = torch.tensor([labels is not None for labels in collated_yolox_labels])
        prediction_times = torch.tensor(collated_prediction_times)
        reset = torch.tensor(collated_reset, dtype=torch.bool)
        assert len(set([element.worker_id for element in batch])) == 1, (
            "All elements in the batch must have the same worker_id."
        )
        return ObjectDetectionBatch(
            batch_indices=batch_indices,
            inputs=inputs,
            has_labels=has_labels,
            prediction_times=prediction_times,
            prophesee_labels=collated_prophesee_labels,
            reset=reset,
            sequence_ids=collated_sequence_ids,
            sequence_lengths=sequence_lengths,
            yolox_labels=collated_yolox_labels,
            worker_id=batch[0].worker_id,  # All sequences have the same worker_id
        )

    def collate_geometric_data(self, batch: list[torch_geometric.data.Data]) -> torch_geometric.data.Batch:
        return torch_geometric.data.Batch.from_data_list(batch)

    def collate_tokens(self, batch: list[Tokens]):
        return collate_tokens(batch, self.buckets, self.patch_size, self.patch_size, self.reverse_time)


def collate_tokens(batch: list[Tokens], buckets: int, height: int, width: int, reverse_time: bool) -> TokensBatch:
    prediction_time = [sequence.prediction_time for sequence in batch]

    pos_x = [torch.tensor(sequence.pos_x) for sequence in batch]
    pos_x = pad_sequence(pos_x, batch_first=True)

    pos_y = [torch.tensor(sequence.pos_y) for sequence in batch]
    pos_y = pad_sequence(pos_y, batch_first=True)

    pos_t = [sequence.pos_t for sequence in batch]
    if reverse_time:
        # Reverse the time to make it relative to the prediction time
        pos_t = [pt - t for pt, t in zip(prediction_time, pos_t, strict=True)]
    pos_t = [torch.tensor(t) for t in pos_t]
    pos_t = pad_sequence(pos_t, batch_first=True)

    prediction_time = torch.tensor(prediction_time)

    # convert events in each token to a frame-based tensor
    tokens = []
    for sequence in batch:
        indices = [np.full_like(value, i) for i, value in enumerate(sequence.events_x)]
        volumes = batched_events_to_logspace_volume(
            batch=indices,
            x=sequence.events_x,
            y=sequence.events_y,
            t=sequence.events_t,
            p=sequence.events_p,
            buckets=buckets,
            height=height,
            width=width,
        )
        volumes = torch.tensor(volumes, dtype=torch.int64)
        tokens.append(volumes)
    tokens = pad_sequence(tokens, batch_first=True)

    padding_mask = make_padding_mask(batch)

    return TokensBatch(
        batch_size=len(batch),
        prediction_time=prediction_time,
        pos_x=pos_x,
        pos_y=pos_y,
        pos_t=pos_t,
        tokens=tokens,
        padding_mask=padding_mask,
    )


def make_padding_mask(batch: list[Tokens]) -> torch.Tensor:
    batch_size = len(batch)
    lengths = torch.tensor([sequence.pos_x.shape[0] for sequence in batch])
    sequence_length = lengths.max()
    padding_mask = torch.arange(sequence_length).expand(batch_size, -1) > lengths.unsqueeze(1)
    return padding_mask

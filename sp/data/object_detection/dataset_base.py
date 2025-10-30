import abc

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn.pool import radius_graph

from sp import augmentations
from sp.configs import Config
from sp.configs import Model
from sp.configs import TokenizerType
from sp.data_types import PROPHESEE_BBOX_DTYPE
from sp.data_types import Tokens
from sp.events import Events
from sp.io import Sequence
from sp.loaders import load_dimensions
from sp.representations.volume import batched_events_to_logspace_volume


class DatasetBase(abc.ABC):
    def __init__(
        self,
        config: Config,
        chunk_index_to_labels: dict[int, dict[int, np.ndarray]],
        sequences: list[Sequence],
        streaming: bool,
        augment: bool = False,
    ):
        super().__init__()
        self.buckets = config.buckets
        self.chunk_index_to_labels = chunk_index_to_labels
        self.height, self.width = load_dimensions(config.dataset)
        self.gnn = config.gnn
        self.max_events = config.max_events
        self.patch_size = config.patch_size
        self.reverse_time = config.reverse_time
        self.sequence_length = config.sequence_length
        self.sequences = sequences
        self.should_augment = augment
        self.streaming = streaming
        self.time_scale = config.time_scale

        if config.model == Model.transformer:
            assert config.tokenizer != TokenizerType.none, "Transformer model does not support raw events."
            self.process_inputs_and_labels = self.augment_and_tokenize
            self.split_inputs = self.split_tokens
            self.process_split_inputs = self.no_process_split_inputs
        elif config.model == Model.gnn and config.tokenizer == TokenizerType.none:
            self.process_inputs_and_labels = self.augment_only
            self.split_inputs = self.split_events
            self.process_split_inputs = self.process_event_graphs
        elif config.model == Model.gnn and config.tokenizer != TokenizerType.none:
            self.process_inputs_and_labels = self.augment_and_tokenize
            self.split_inputs = self.split_tokens
            self.process_split_inputs = self.process_token_graphs
        elif config.model == Model.pcn and config.tokenizer == TokenizerType.none:
            self.process_inputs_and_labels = self.augment_only
            self.split_inputs = self.split_events
            self.process_split_inputs = self.process_event_clouds
        elif config.model == Model.pcn and config.tokenizer != TokenizerType.none:
            self.process_inputs_and_labels = self.augment_and_tokenize
            self.split_inputs = self.split_tokens
            self.process_split_inputs = self.process_token_clouds
        else:
            raise ValueError(f"Unsupported model {config.model} with tokenizer {config.tokenizer}.")

        if self.should_augment:
            if self.streaming:
                self.augmentation = augmentations.Compose(
                    [
                        augmentations.Chance(
                            augmentations.HorizontalFlip(self.width),
                        ),
                        augmentations.OneOf(
                            [
                                augmentations.Identity(),
                                augmentations.DropByTime(),
                                augmentations.DropEvent(),
                                augmentations.Rotation(height=self.height, width=self.width, max_degree=10),
                            ]
                        ),
                    ]
                )
            else:
                self.augmentation = augmentations.Compose(
                    [
                        augmentations.Chance(
                            augmentations.HorizontalFlip(self.width),
                        ),
                        augmentations.OneOf(
                            [
                                augmentations.Identity(),
                                augmentations.DropByArea(height=self.height, width=self.width),
                                augmentations.DropByTime(),
                                augmentations.DropEvent(),
                                augmentations.Rotation(height=self.height, width=self.width, max_degree=20),
                            ]
                        ),
                    ]
                )

    @abc.abstractmethod
    def augment(self, sample: augmentations.Sample) -> augmentations.Sample:
        pass

    @abc.abstractmethod
    def tokenize(self, events: Events) -> Tokens:
        pass

    def augment_only(
        self, events: Events, labels: np.ndarray
    ) -> tuple[Events, list[augmentations.ObjectDetectionLabel]]:
        original = augmentations.Sample(
            events=events,
            label=augmentations.Label(
                object_detection=[
                    augmentations.ObjectDetectionLabel(
                        x=box["x"].item(),
                        y=box["y"].item(),
                        width=box["w"].item(),
                        height=box["h"].item(),
                        class_id=box["class_id"].item(),
                        t=box["t"].item(),
                    )
                    for box in labels
                ]
            ),
        )

        if not self.should_augment:
            return original.events, original.label.object_detection

        augmented = self.augment(original)

        had_labels = len(original.label.object_detection) > 0
        has_labels = len(augmented.label.object_detection) > 0
        if not self.streaming and (had_labels and not has_labels):
            # Augmentation was too strong. We revert to the original sample instead.
            # Note: We do not revert augmentations for streaming because we want the sequence to be consistent.
            return original.events, original.label.object_detection

        return augmented.events, augmented.label.object_detection

    def augment_and_tokenize(
        self, events: Events, labels: np.ndarray
    ) -> tuple[Tokens, list[augmentations.ObjectDetectionLabel]]:
        original = augmentations.Sample(
            events=events,
            label=augmentations.Label(
                object_detection=[
                    augmentations.ObjectDetectionLabel(
                        x=box["x"].item(),
                        y=box["y"].item(),
                        width=box["w"].item(),
                        height=box["h"].item(),
                        class_id=box["class_id"].item(),
                        t=box["t"].item(),
                    )
                    for box in labels
                ]
            ),
        )

        if not self.should_augment:
            tokens = self.tokenize(events)
            return tokens, original.label.object_detection

        augmented = self.augment(original)

        tokens = self.tokenize(augmented.events)

        had_labels = len(original.label.object_detection) > 0
        has_labels = len(augmented.label.object_detection) > 0
        if not self.streaming and (tokens.pos_x.size == 0 or (had_labels and not has_labels)):
            # Augmentation was too strong. We revert to the original sample instead.
            # Note: We do not revert augmentations for streaming because we want the sequence to be consistent.
            tokens = self.tokenize(events)
            return tokens, original.label.object_detection

        return tokens, augmented.label.object_detection

    def get_split_times(self, sequence_index: int, start_index: int, end_index: int) -> list[int]:
        sequence = self.sequences[sequence_index]
        split_times = [sequence.chunks[i].end for i in range(start_index, end_index)]
        return split_times

    def load_labels(self, sequence_index: int, start_chunk_index: int, end_chunk_index: int) -> np.ndarray:
        chunk_index_to_labels = self.chunk_index_to_labels[sequence_index]
        labels = []
        for chunk_index in range(start_chunk_index, end_chunk_index):
            if chunk_index in chunk_index_to_labels:
                labels.append(chunk_index_to_labels[chunk_index])

        if len(labels) == 0:
            return np.empty((0,), dtype=PROPHESEE_BBOX_DTYPE)

        labels = np.concatenate(labels)
        return labels

    def split_events(self, split_times: list[int], events: Events) -> list[Events]:
        """Splits events into groups with durations of self.predict_every_us."""
        split_times = np.array(split_times)
        split_indices = np.searchsorted(events.t, split_times, side="right")
        outputs: list[Events] = []
        for split_index in range(len(split_indices)):
            start_index = 0 if split_index == 0 else split_indices[split_index - 1]
            end_index = split_indices[split_index]
            assert start_index <= end_index

            if start_index == end_index:
                # no events in this split
                empty_events = Events(
                    x=np.empty((0,), dtype=np.uint16),
                    y=np.empty((0,), dtype=np.uint16),
                    t=np.empty((0,), dtype=np.uint64),
                    p=np.empty((0,), dtype=bool),
                )
                outputs.append(empty_events)
                continue

            if self.max_events is not None:
                start_index = max(start_index, end_index - self.max_events)

            events_split = Events(
                x=events.x[start_index:end_index],
                y=events.y[start_index:end_index],
                t=events.t[start_index:end_index],
                p=events.p[start_index:end_index],
            )
            outputs.append(events_split)

        return outputs

    def split_tokens(self, split_times: list[int], tokens: Tokens) -> list[Tokens]:
        """Splits tokens into groups with durations of self.predict_every_us."""
        split_times = np.array(split_times)
        split_indices = np.searchsorted(tokens.pos_t, split_times, side="right")
        outputs: list[Tokens] = []
        for split_index in range(len(split_indices)):
            start_index = 0 if split_index == 0 else split_indices[split_index - 1]
            end_index = split_indices[split_index]
            assert start_index <= end_index

            prediction_time = split_times[split_index].item()

            if start_index == end_index:
                # no tokens in this split
                empty_tokens = Tokens(
                    prediction_time=prediction_time,
                    pos_x=np.empty((0,), dtype=np.uint16),
                    pos_y=np.empty((0,), dtype=np.uint16),
                    pos_t=np.empty((0,), dtype=np.uint64),
                    events_x=[],
                    events_y=[],
                    events_t=[],
                    events_p=[],
                )
                outputs.append(empty_tokens)
                continue

            tokens_split = Tokens(
                prediction_time=prediction_time,
                pos_x=tokens.pos_x[start_index:end_index],
                pos_y=tokens.pos_y[start_index:end_index],
                pos_t=tokens.pos_t[start_index:end_index],
                events_x=tokens.events_x[start_index:end_index],
                events_y=tokens.events_y[start_index:end_index],
                events_t=tokens.events_t[start_index:end_index],
                events_p=tokens.events_p[start_index:end_index],
            )
            outputs.append(tokens_split)

        return outputs

    def split_labels(
        self, split_times: list[int], labels: list[augmentations.ObjectDetectionLabel]
    ) -> list[list[augmentations.ObjectDetectionLabel]]:
        """Splits labels into groups with durations of self.predict_every_us."""
        split_times = np.array(split_times)
        split_indices = np.searchsorted([box.t for box in labels], split_times, side="right")
        outputs: list[list[augmentations.ObjectDetectionLabel]] = []
        for split_index in range(len(split_indices)):
            start_index = 0 if split_index == 0 else split_indices[split_index - 1]
            end_index = split_indices[split_index]
            assert start_index <= end_index
            labels_split = labels[start_index:end_index]
            outputs.append(labels_split)
        return outputs

    def convert_to_prophesee_labels(
        self, labels: list[list[augmentations.ObjectDetectionLabel]]
    ) -> list[np.ndarray | None]:
        """Converts the labels to the format expected by the Prophesee evaluator."""
        outputs = []
        for group in labels:
            if len(group) == 0:
                outputs.append(None)
                continue
            prophesee_labels = np.array(
                [(box.t, box.x, box.y, box.width, box.height, box.class_id, 1.0) for box in group],
                dtype=PROPHESEE_BBOX_DTYPE,
            )
            outputs.append(prophesee_labels)
        return outputs

    def convert_to_yolox_labels(
        self, labels: list[list[augmentations.ObjectDetectionLabel]]
    ) -> list[torch.Tensor | None]:
        """Converts the labels to the format expected by YOLOX."""
        outputs = []
        for group in labels:
            if len(group) == 0:
                outputs.append(None)
                continue
            class_labels = [box.class_id for box in group]
            class_labels = torch.tensor(class_labels, dtype=torch.long)
            bounding_boxes = list(map(self.format_yolox_bounding_box, group))
            bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
            yolox_labels = torch.cat((class_labels.unsqueeze(1), bounding_boxes), dim=1)
            outputs.append(yolox_labels)
        return outputs

    def format_yolox_bounding_box(self, box: augmentations.ObjectDetectionLabel):
        """Converts from a top-left corner and dimensions to a center and dimensions format."""
        center_x = box.x + box.width / 2
        center_y = box.y + box.height / 2
        width = box.width
        height = box.height
        return [center_x, center_y, width, height]

    def no_process_split_inputs(self, split_inputs: list, split_times: list[int]) -> list:
        return split_inputs

    def process_event_clouds(
        self, split_events: list[Events], split_times: list[int]
    ) -> list[torch_geometric.data.Data]:
        """Processes a list of events into a list of torch_geometric.data.Data objects."""
        outputs = []
        for events, split_time in zip(split_events, split_times, strict=True):
            t = events.t
            if self.reverse_time:
                # Reverse the time to make it relative to the prediction time
                t = split_time - t
            t = t / self.time_scale

            # should be safe to cast since we have divided by time_scale
            # and also subtracted minimum time in a previous step
            t = t.astype(np.float32)

            positions = np.stack([events.x, events.y, t], axis=1)
            positions = torch.tensor(positions)
            node_features = torch.tensor(events.p)
            data = torch_geometric.data.Data(pos=positions, x=node_features)
            outputs.append(data)
        return outputs

    def process_token_clouds(
        self, split_tokens: list[Tokens], split_times: list[int]
    ) -> list[torch_geometric.data.Data]:
        """Processes a list of tokens into a list of torch_geometric.data.Data objects."""
        outputs = []
        for tokens in split_tokens:
            pos_t = tokens.pos_t
            if self.reverse_time:
                # Reverse the time to make it relative to the prediction time
                pos_t = tokens.prediction_time - pos_t
            pos_t = pos_t / self.time_scale

            # should be safe to cast since we have divided by time_scale
            # and also subtracted minimum time in a previous step
            pos_t = pos_t.astype(np.float32)

            positions = np.stack([tokens.pos_x, tokens.pos_y, pos_t], axis=1)
            positions = torch.tensor(positions)

            # convert events in each token to a frame-based tensor
            indices = [np.full_like(value, i) for i, value in enumerate(tokens.events_x)]
            node_features = batched_events_to_logspace_volume(
                batch=indices,
                x=tokens.events_x,
                y=tokens.events_y,
                t=tokens.events_t,
                p=tokens.events_p,
                buckets=self.buckets,
                height=self.patch_size,
                width=self.patch_size,
            )
            node_features = torch.tensor(node_features)

            graph = torch_geometric.data.Data(pos=positions, x=node_features)
            outputs.append(graph)
        return outputs

    def process_event_graphs(
        self, split_events: list[Events], split_times: list[int]
    ) -> list[torch_geometric.data.Data]:
        """Processes a list of events into a list of torch_geometric.data.Data objects."""
        outputs = []
        for events, split_time in zip(split_events, split_times, strict=True):
            t = events.t
            if self.reverse_time:
                # Reverse the time to make it relative to the prediction time
                t = split_time - t
            t = t / self.time_scale
            positions = np.stack([events.x, events.y, t], axis=1)
            positions = torch.tensor(positions)
            edge_index = radius_graph(
                positions,
                loop=self.gnn.loop,
                max_num_neighbors=self.gnn.max_num_neighbors,
                r=self.gnn.node_radius,
            )

            node_features = torch.tensor(events.p)

            graph = torch_geometric.data.Data(
                edge_index=edge_index,
                pos=positions,
                x=node_features,
            )
            outputs.append(graph)
        return outputs

    def process_token_graphs(
        self, split_tokens: list[Tokens], split_times: list[int]
    ) -> list[torch_geometric.data.Data]:
        """Processes a list of tokens into a list of torch_geometric.data.Data objects."""
        outputs = []
        for tokens in split_tokens:
            pos_t = tokens.pos_t
            if self.reverse_time:
                # Reverse the time to make it relative to the prediction time
                pos_t = tokens.prediction_time - pos_t
            pos_t = pos_t / self.time_scale

            positions = np.stack([tokens.pos_x, tokens.pos_y, pos_t], axis=1)
            positions = torch.tensor(positions)
            edge_index = radius_graph(
                positions,
                loop=self.gnn.loop,
                max_num_neighbors=self.gnn.max_num_neighbors,
                r=self.gnn.node_radius,
            )

            # convert events in each token to a frame-based tensor
            indices = [np.full_like(value, i) for i, value in enumerate(tokens.events_x)]
            node_features = batched_events_to_logspace_volume(
                batch=indices,
                x=tokens.events_x,
                y=tokens.events_y,
                t=tokens.events_t,
                p=tokens.events_p,
                buckets=self.buckets,
                height=self.patch_size,
                width=self.patch_size,
            )
            node_features = torch.tensor(node_features)

            graph = torch_geometric.data.Data(
                edge_index=edge_index,
                pos=positions,
                x=node_features,
            )
            outputs.append(graph)
        return outputs

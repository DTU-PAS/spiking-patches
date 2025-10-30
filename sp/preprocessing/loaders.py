import abc
from pathlib import Path

import h5py

from sp.events import Events
from sp.prophesee.loader import PSEELoader


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def done() -> bool:
        pass

    @abc.abstractmethod
    def load_delta_t(self, delta_t: int) -> Events:
        """Load the next delta_t (microseconds) events from the sequence."""
        pass

    @abc.abstractmethod
    def load_past(self) -> Events:
        """Load all of the past events from the sequence."""
        pass

    @abc.abstractmethod
    def seek_time(self, time: int):
        """Seek to the specified time (microseconds) in the sequence."""
        pass


class PropheseeLoader(BaseLoader):
    def __init__(self, data_path: Path):
        super().__init__()
        self.loader = PSEELoader(str(data_path))

    def close(self):
        del self.loader

    def done(self):
        return self.loader.done

    def load_delta_t(self, delta_t: int) -> Events:
        events = self.loader.load_delta_t(delta_t)
        return Events(
            x=events["x"],
            y=events["y"],
            t=events["t"],
            p=events["p"],
        )

    def load_past(self) -> Events:
        pos = self.loader.file.tell()
        past_count = (pos - self.loader.start) // self.loader.ev_size
        events = self.loader.load_n_past_events(past_count)
        return Events(
            x=events["x"],
            y=events["y"],
            t=events["t"],
            p=events["p"],
        )

    def seek_time(self, time: int):
        self.loader.seek_time(time)


class ETraMLoader(BaseLoader):
    def __init__(self, data_path: Path):
        super().__init__()
        self.file = h5py.File(data_path, "r")
        self.events = self.file["events"]
        self.start = 0
        self.end = self.events["t"].shape[0] - 1
        self.current_time = self.events["t"][0]

    def close(self):
        self.file.close()

    def done(self):
        return self.start >= self.end

    def load_delta_t(self, delta_t: int) -> Events:
        end_time = self.current_time + delta_t
        end = self.search_time(end_time)
        events = Events(
            x=self.events["x"][self.start : end],
            y=self.events["y"][self.start : end],
            t=self.events["t"][self.start : end],
            p=self.events["p"][self.start : end],
        )
        self.start = end
        self.current_time = end_time
        return events

    def load_past(self) -> Events:
        events = Events(
            x=self.events["x"][: self.start],
            y=self.events["y"][: self.start],
            t=self.events["t"][: self.start],
            p=self.events["p"][: self.start],
        )
        return events

    def seek_time(self, time: int):
        self.start = self.search_time(time)
        self.current_time = time

    def search_time(self, time: int):
        timestamps = self.events["t"]
        start = self.start
        end = self.end
        while start < end:
            mid = (start + end) // 2
            if timestamps[mid] < time:
                start = mid + 1
            else:
                end = mid
        return start

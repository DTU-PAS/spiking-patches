from dataclasses import dataclass

import numpy as np


@dataclass
class Events:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray

    def __len__(self):
        return len(self.x)

    def mask(self, mask: np.ndarray) -> "Events":
        return Events(
            x=self.x[mask],
            y=self.y[mask],
            t=self.t[mask],
            p=self.p[mask],
        )

    def __getitem__(self, index: int) -> "Events":
        return Events(
            x=self.x[index],
            y=self.y[index],
            t=self.t[index],
            p=self.p[index],
        )

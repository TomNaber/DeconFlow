"""Bounded, acknowledged handoff from batch output to the interactive preview."""
import json
import os
from pathlib import Path
import shutil
from time import sleep

import numpy as np


def publish(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, default=str))
    temporary.replace(path)


class BatchPreview:
    def __init__(self, directory, settings):
        self.directory = Path(directory)
        self.settings = settings
        self.first = True
        self.enabled = True
        self.path = None
        self.parent_pid = os.getppid()

    def attempt(self, action, *args):
        if self.enabled:
            try:
                action(*args)
            except Exception as error:
                self.enabled = False
                print(f"Preview handoff unavailable: {error}", flush=True)

    def send(self, kind):
        ready = self.directory / "ready.json"
        rejected = self.directory / "rejected"
        rejected.unlink(missing_ok=True)
        publish(ready, dict(kind=kind, path=str(self.path), settings=self.settings))
        # The UI acknowledges only after adopting the matching pair. This bounds
        # staging to one sample and also orders completion before batch unlock.
        while ready.exists():
            if os.getppid() != self.parent_pid or (self.directory / "disabled").exists():
                raise RuntimeError("preview receiver stopped")
            sleep(.05)
        return not rejected.exists()

    def start(self, path):
        self.path = Path(path).resolve()
        publish(self.directory / "activity.json", {"path": str(self.path)})
        self.discard()
        (self.directory / "sample").mkdir()
        if self.first:
            self.first = not self.send("original")

    def capture_raw(self, index, volume):
        np.save(self.directory / "sample" / f"raw-{index}.npy",
                np.asarray(volume, dtype=np.float32), allow_pickle=False)

    def capture_result(self, index, volume):
        low, high = float(volume.min()), float(volume.max())
        if low == high:
            low, high = (0., high) if high > 0 else (low, low + 1.)
        np.savez(self.directory / "sample" / f"projection-{index}.npz",
                 xy=volume.max(axis=0), yz=volume.max(axis=2), range=(low, high))

    def finish(self):
        self.send("result")

    def discard(self):
        if not (self.directory / "ready.json").exists():
            shutil.rmtree(self.directory / "sample", ignore_errors=True)

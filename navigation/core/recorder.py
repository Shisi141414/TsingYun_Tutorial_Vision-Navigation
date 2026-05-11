"""Screen recorder using OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pygame

class Recorder:
    """
    Record a pygame display to an MP4 file.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        output_path: str = "recording.mp4",
        fps: int = 30,
    ) -> None:
        self.screen = screen
        self.output_path = str(output_path)
        self.fps = fps
        self._writer = None
        self._active = False

    def start(self) -> None:
        """Open the video writer. Call before the game loop."""
        w, h = self.screen.get_size()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        self._active = True
        print(f"[Recorder] started → {self.output_path}")

    def capture(self) -> None:
        """Grab the current screen and append a frame. Call after display.flip()."""
        if not self._active or self._writer is None:
            return
        raw = pygame.surfarray.array3d(self.screen)   # shape (w, h, 3), RGB
        frame = np.transpose(raw, (1, 0, 2))           # → (h, w, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(frame)

    def stop(self) -> None:
        """Flush and close the video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._active = False
        print(f"[Recorder] saved → {self.output_path}")

    @property
    def is_active(self) -> bool:
        return self._active

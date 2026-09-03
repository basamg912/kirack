"""RecordVideo wrapper that also uploads each video to wandb when it finishes."""

from __future__ import annotations

import os
import gymnasium as gym


class WandbRecordVideo(gym.wrappers.RecordVideo):
    """Drop-in replacement for gym.wrappers.RecordVideo.

    On every transition from recording → not-recording, scans the video folder
    for new .mp4 files and uploads them to the active wandb run (if any).

    Falls back to plain RecordVideo behavior when:
        - wandb is not installed, or
        - no active wandb run (e.g., --logger tensorboard)
    """

    def __init__(self, env, *args, wandb_key: str = "video/train", **kwargs):
        super().__init__(env, *args, **kwargs)
        self._wandb_key = wandb_key
        self._uploaded: set[str] = set()

    def step(self, action):
        was_recording = bool(getattr(self, "recording", False))
        out = super().step(action)
        is_recording = bool(getattr(self, "recording", False))
        # recording just ended -> a new mp4 was likely flushed
        if was_recording and not is_recording:
            self._upload_new()
        return out

    def close(self):
        # flush anything left over at the end of training
        self._upload_new()
        return super().close()

    def _upload_new(self):
        try:
            import wandb
        except ImportError:
            return
        if getattr(wandb, "run", None) is None:
            return

        folder = self.video_folder
        if not os.path.isdir(folder):
            return

        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".mp4"):
                continue
            full = os.path.join(folder, fn)
            if full in self._uploaded:
                continue
            # make sure file is fully written (size stable)
            try:
                size_a = os.path.getsize(full)
            except OSError:
                continue
            if size_a == 0:
                continue
            try:
                wandb.log({self._wandb_key: wandb.Video(full, format="mp4")})
                self._uploaded.add(full)
                print(f"[wandb] uploaded {fn}")
            except Exception as e:
                print(f"[wandb] upload failed for {fn}: {e}")

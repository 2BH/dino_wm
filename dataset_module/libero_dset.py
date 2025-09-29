import torch
import numpy as np
from typing import Optional, Callable, List

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames
from einops import rearrange

from .traj_dset import TrajDataset


class LiberoTrajDataset(TrajDataset):
    """Episode-level Libero dataset backed by LeRobotDataset.

    Similar to PushTDataset/PointMazeDataset: __getitem__ returns the entire
    trajectory (obs, act, state, info).  Windowing/frameskip are handled by a
    separate slicer dataset.
    """

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        episodes: List[int] | None = None,
        camera_key: str = "observation.images.image",
        state_key: str = "observation.state",
        action_key: str = "action",
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
    ):
        self.repo_id = repo_id
        self.camera_key = camera_key
        self.state_key = state_key
        self.action_key = action_key
        self.transform = transform
        self.normalize_action = normalize_action

        self.meta = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
            delta_timestamps=None,
            tolerance_s=1e-4,
            download_videos=True,
        )

        if episodes is None:
            self.episode_ids = (
                list(self.dataset.episodes)
                if self.dataset.episodes is not None
                else list(range(self.meta.total_episodes))
            )
        else:
            self.episode_ids = list(episodes)

        dataset_episode_order = (
            list(self.dataset.episodes)
            if self.dataset.episodes is not None
            else list(range(self.meta.total_episodes))
        )
        self._episode_pos = {
            ep_idx: pos for pos, ep_idx in enumerate(dataset_episode_order)
        }

        feature_shapes = self.meta.features
        self.action_dim = feature_shapes[self.action_key]["shape"][-1]
        self.state_dim = feature_shapes[self.state_key]["shape"][-1]
        self.proprio_dim = self.state_dim

        if self.normalize_action:
            self.action_mean, self.action_std = self._get_feature_stats(self.action_key)
            self.state_mean, self.state_std = self._get_feature_stats(self.state_key)
            self.proprio_mean = self.state_mean.clone()
            self.proprio_std = self.state_std.clone()
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self._episode_lengths = [
            self.meta.episodes[ep_idx]["length"] for ep_idx in self.episode_ids
        ]

    def _get_feature_stats(self, feature_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.meta.stats[feature_key]
        mean = stats["mean"]
        std = stats["std"]
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        else:
            mean = torch.tensor(mean)
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        else:
            std = torch.tensor(std)
        return mean.float(), (std.float() + 1e-6)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def get_seq_length(self, idx: int) -> int:
        return self._episode_lengths[idx]

    def __getitem__(self, idx: int):
        episode_index = self.episode_ids[idx]
        pos = self._episode_pos[episode_index]
        start = self.dataset.episode_data_index["from"][pos].item()
        end = self.dataset.episode_data_index["to"][pos].item()

        indices = list(range(start, end))
        batch = self.dataset.hf_dataset.select(indices)

        states = torch.stack([item.float() for item in batch[self.state_key]])
        actions = self._prepare_actions(batch)
        if self.normalize_action:
            actions = (actions - self.action_mean.to(actions.device)) / self.action_std.to(actions.device)
            states = (states - self.state_mean.to(states.device)) / self.state_std.to(states.device)

        visuals = self._prepare_visuals(episode_index, batch)
        if self.transform is not None:
            visuals = self.transform(visuals)

        obs = {"visual": visuals, "proprio": states}
        info = {"episode_index": episode_index}
        return obs, actions, states, info

    def _prepare_actions(self, batch: dict) -> torch.Tensor:
        actions = torch.stack([item.float() for item in batch[self.action_key]])
        state_len = len(batch[self.state_key])
        if actions.shape[0] == state_len - 1:
            pad = torch.zeros((1, actions.shape[1]), dtype=actions.dtype)
            actions = torch.cat([actions, pad], dim=0)
        elif actions.shape[0] < state_len:
            pad = torch.zeros((state_len - actions.shape[0], actions.shape[1]), dtype=actions.dtype)
            actions = torch.cat([actions, pad], dim=0)
        elif actions.shape[0] > state_len:
            actions = actions[:state_len]
        return actions

    def _prepare_visuals(self, episode_index: int, batch: dict) -> torch.Tensor:
        camera_dtype = self.meta.features[self.camera_key]["dtype"]
        if camera_dtype == "video":
            timestamps = [ts.item() for ts in batch["timestamp"]]
            video_path = self.dataset.root / self.meta.get_video_file_path(episode_index, self.camera_key)
            frames = decode_video_frames(
                video_path,
                timestamps,
                self.dataset.tolerance_s,
                backend=self.dataset.video_backend,
            )
            visuals = frames.to(torch.float32) / 255.0
            visuals = self._maybe_to_chw(visuals)
        else:
            visuals = torch.stack([
                item.float() / 255.0 if item.dtype == torch.uint8 else item.float()
                for item in batch[self.camera_key]
            ])
            visuals = self._maybe_to_chw(visuals)
        return visuals

    def _maybe_to_chw(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {imgs.shape}")
        if imgs.shape[1] in (1, 3):
            return imgs
        return rearrange(imgs, "t h w c -> t c h w")


class LiberoTrajSlicerDataset(TrajDataset):
    """Windowed Libero dataset built on top of LiberoDataset."""

    def __init__(
        self,
        base_dataset: LiberoTrajDataset,
        num_frames: int,
        frameskip: int = 1,
    ):
        self.dataset = base_dataset
        self.num_frames = num_frames
        self.frameskip = frameskip

        self.slices = []
        for traj_idx in range(len(self.dataset)):
            traj_len = self.dataset.get_seq_length(traj_idx)
            effective_span = num_frames * frameskip
            if traj_len < effective_span:
                print(f"Ignored short sequence #{traj_idx}: len={traj_len}, num_frames={num_frames}, frameskip={frameskip}")
                continue
            self.slices.extend(
                (traj_idx, start, start + effective_span)
                for start in range(traj_len - effective_span + 1)
            )

        self.proprio_dim = self.dataset.proprio_dim
        self.state_dim = self.dataset.state_dim
        self.action_dim = self.dataset.action_dim * frameskip

    def __len__(self) -> int:
        return len(self.slices)

    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __getitem__(self, idx: int):
        traj_idx, start, end = self.slices[idx]
        obs_full, act_full, state_full, _ = self.dataset[traj_idx]

        obs = {
            "visual": obs_full["visual"][start:end:self.frameskip],
            "proprio": obs_full["proprio"][start:end:self.frameskip],
        }
        state = state_full[start:end:self.frameskip]

        actions_segment = act_full[start:end]
        actions = rearrange(actions_segment, "(n f) d -> n (f d)", n=self.num_frames, f=self.frameskip)
        return obs, actions, state


def load_libero_slice_train_val(
    transform,
    repo_id: str = "lerobot/libero_goal_image",
    n_rollout: Optional[int] = None,
    split_ratio: float = 0.9,
    camera_key: str = "observation.images.image",
    state_key: str = "observation.state",
    action_key: str = "action",
    normalize_action: bool = False,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 1,
):
    num_frames = num_hist + num_pred
    assert num_frames > 0, "num_hist + num_pred must be > 0"

    meta = LeRobotDatasetMetadata(repo_id)
    total_eps = meta.total_episodes
    n_train = int(split_ratio * total_eps) if n_rollout is None else min(n_rollout, total_eps)
    if n_rollout is None:
        indices = np.arange(total_eps)
        np.random.shuffle(indices)
    else:
        indices = np.random.choice(total_eps, size=n_rollout, replace=False)
    train_eps = indices[:n_train].tolist()
    val_eps = indices[n_train:].tolist()

    train_traj_dataset = LiberoTrajDataset(
        repo_id=repo_id,
        episodes=train_eps,
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        transform=transform,
        normalize_action=normalize_action,
    )
    val_traj_dataset = LiberoTrajDataset(
        repo_id=repo_id,
        episodes=val_eps,
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        transform=transform,
        normalize_action=normalize_action,
    )

    train_slice_dataset = LiberoTrajSlicerDataset(
        base_dataset=train_traj_dataset,
        num_frames=num_frames,
        frameskip=frameskip,
    )
    val_slice_dataset = LiberoTrajSlicerDataset(
        base_dataset=val_traj_dataset,
        num_frames=num_frames,
        frameskip=frameskip,
    )

    datasets = {"train": train_slice_dataset, "valid": val_slice_dataset}
    traj_dset = {"train": train_traj_dataset, "valid": val_traj_dataset}
    return datasets, traj_dset



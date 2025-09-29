import torch
import numpy as np
from typing import Optional, Callable, List, Dict

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames
from einops import rearrange

from .traj_dset import TrajDataset


class LiberoDataset(LeRobotDataset):
    """
    Windowed Libero dataset backed by LeRobotDataset.

    Produces fixed-length sequences compatible with this codebase:
      - obs["visual"]: (T, C, H, W) float in [-1, 1] after transform
      - obs["proprio"]: (T, Dp)
      - act: (T, Da) (last step padded with zeros)
      - state: (T, Ds)

    Note: This dataset already returns fixed-length windows, so it is
    used directly by the dataloaders. We also expose a tiny empty
    Trajectory dataset in the loader function to skip open-loop rollouts
    (which expect full episodes).
    """

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        episodes: List[int] | None = None,
        seq_len: int = 20,
        frameskip: int = 1,
        camera_key: str = "observation.images.image",
        state_key: str = "observation.state",
        action_key: str = "action",
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
    ):
        # Build delta timestamps to request a length-T window
        meta_data = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = { # TODO: Ensure the fps is correct
            camera_key: [i * 1/meta_data.fps*frameskip for i in range(seq_len)],
            state_key: [i * 1/meta_data.fps*frameskip for i in range(seq_len)],
            action_key: [i * 1/meta_data.fps*frameskip for i in range(seq_len)],
        }
        self.transform = transform
        self.camera_key = camera_key
        self.state_key = state_key
        self.action_key = action_key
        self.seq_len = seq_len
        self.normalize_action = normalize_action
        self.frameskip = frameskip
        super().__init__(repo_id, episodes=episodes, delta_timestamps=delta_timestamps)

        # Expose dims for trainers
        self.proprio_dim = meta_data.features[state_key]["shape"][-1]
        self.state_dim = meta_data.features[state_key]["shape"][-1]
        self.action_dim = meta_data.features[action_key]["shape"][-1]

        # Default stats (used by planners); will be refined if normalize_action
        self.action_mean = torch.zeros(self.action_dim)
        self.action_std = torch.ones(self.action_dim)
        self.state_mean = torch.zeros(self.state_dim)
        self.state_std = torch.ones(self.state_dim)
        self.proprio_mean = torch.zeros(self.proprio_dim)
        self.proprio_std = torch.ones(self.proprio_dim)

        if self.normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.action_key, meta_data)
            self.state_mean, self.state_std = self.get_data_mean_std(self.state_key, meta_data)
            self.proprio_mean = self.state_mean.clone()
            self.proprio_std = self.state_std.clone()

    def get_data_mean_std(self, data_key, meta_data):
        mean = meta_data.stats[data_key]["mean"]
        std = meta_data.stats[data_key]["std"]
        if isinstance(mean, np.ndarray):
            mean = torch.tensor(mean, dtype=torch.float32)
        if isinstance(std, np.ndarray):
            std = torch.tensor(std, dtype=torch.float32)
        return mean, std

    def _maybe_to_chw(self, imgs: torch.Tensor) -> torch.Tensor:
        # Expect (T, C, H, W). If (T, H, W, C), convert.
        if imgs.ndim != 4:
            raise ValueError(f"Expected 4D tensor for images, got shape {imgs.shape}")
        if imgs.shape[1] in (1, 3):
            return imgs  # already (T, C, H, W)
        # assume (T, H, W, C)
        return rearrange(imgs, "t h w c -> t c h w")

    def _norm_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.normalize_action:
            return actions
        return (actions - self.action_mean) / self.action_std

    def _norm_states(self, states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_action: # TODO: add a new argument for this
            return states
        return (states - self.state_mean) / self.state_std

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        imgs = sample[self.camera_key]
        imgs = self._maybe_to_chw(imgs)
        if self.transform is not None:
            imgs = self.transform(imgs)
        states = sample[self.state_key]
        actions = sample[self.action_key]
        # normalize states and actions
        if self.normalize_action:
            actions = self._norm_actions(actions)
            states = self._norm_states(states)
        obs = {"visual": imgs, "proprio": states}
        act = actions
        state = states
        return obs, act, state


class LiberoTrajectoryDataset(TrajDataset):
    """Episode-level Libero dataset exposing the TrajDataset API.

    Loads full episodes from LeRobotDataset and returns sequences suitable for
    open-loop evaluation utilities (e.g. `Trainer.openloop_rollout`). Images are
    decoded once per episode request and optionally transformed to match the
    training pipeline.
    """

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        episodes: List[int] | None = None,
        frameskip: int = 1,
        camera_key: str = "observation.images.image",
        state_key: str = "observation.state",
        action_key: str = "action",
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        tolerance_s: float = 1e-4,
        base_dataset: LeRobotDataset | None = None,
        meta_data: LeRobotDatasetMetadata | None = None,
    ):
        self.repo_id = repo_id
        self.camera_key = camera_key
        self.state_key = state_key
        self.action_key = action_key
        self.frameskip = max(1, frameskip)
        self.transform = transform
        self.normalize_action = normalize_action

        if base_dataset is not None:
            self.dataset = base_dataset
            dataset_episode_order = (
                list(base_dataset.episodes)
                if base_dataset.episodes is not None
                else list(range(base_dataset.meta.total_episodes))
            )
        else:
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes,
                delta_timestamps=None,
                tolerance_s=tolerance_s,
                download_videos=True,
            )
            dataset_episode_order = (
                list(self.dataset.episodes)
                if self.dataset.episodes is not None
                else list(range(self.dataset.meta.total_episodes))
            )

        self.meta_data = meta_data if meta_data is not None else self.dataset.meta
        self.fps = self.meta_data.fps

        if episodes is None:
            self.episode_ids = dataset_episode_order
        else:
            self.episode_ids = list(episodes)
            if dataset_episode_order:
                missing_eps = [ep for ep in self.episode_ids if ep not in dataset_episode_order]
                if missing_eps:
                    raise ValueError(
                        "Requested episodes are not available in the provided LeRobotDataset: "
                        f"{missing_eps}"
                    )

        dataset_position = {ep_idx: pos for pos, ep_idx in enumerate(dataset_episode_order)}
        self._episode_pos = {ep_idx: dataset_position[ep_idx] for ep_idx in self.episode_ids}

        episode_lengths = [self.meta_data.episodes[ep]["length"] for ep in self.episode_ids]
        self.seq_lengths = [
            len(range(0, length, self.frameskip)) if length > 0 else 0
            for length in episode_lengths
        ]

        feature_shapes = self.meta_data.features
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

        camera_dtype = self.meta_data.features[self.camera_key]["dtype"]
        self._camera_is_video = camera_dtype == "video"

    def _get_feature_stats(self, feature_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.meta_data.stats[feature_key]
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

    def _maybe_to_chw(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim != 4:
            raise ValueError(f"Expected 4D tensor for images, got shape {imgs.shape}")
        if imgs.shape[1] in (1, 3):
            return imgs
        return rearrange(imgs, "t h w c -> t c h w")

    def _load_episode_visuals(
        self, episode_index: int, timestamps: List[float]
    ) -> torch.Tensor:
        video_path = self.dataset.root / self.meta_data.get_video_file_path(
            episode_index, self.camera_key
        )
        frames = decode_video_frames(
            video_path,
            timestamps,
            self.dataset.tolerance_s,
            backend=self.dataset.video_backend,
        )
        frames = frames.to(torch.float32) / 255.0
        frames = self._maybe_to_chw(frames)
        if self.transform is not None:
            frames = self.transform(frames)
        return frames

    def _load_episode_structured(self, start: int, end: int) -> dict:
        indices = list(range(start, end, self.frameskip))
        batch = self.dataset.hf_dataset.select(indices)
        data = {
            "states": torch.stack([item.float() for item in batch[self.state_key]]),
            "actions": torch.stack([item.float() for item in batch[self.action_key]]),
            "timestamps": [ts.item() for ts in batch["timestamp"]],
        }

        if not self._camera_is_video:
            visuals = torch.stack(
                [
                    item.float() / 255.0 if item.dtype == torch.uint8 else item.float()
                    for item in batch[self.camera_key]
                ]
            )
            visuals = self._maybe_to_chw(visuals)
            if self.transform is not None:
                visuals = self.transform(visuals)
            data["visuals"] = visuals

        if data["actions"].shape[0] < data["states"].shape[0]:
            pad = torch.zeros(
                (data["states"].shape[0] - data["actions"].shape[0], self.action_dim),
                dtype=data["actions"].dtype,
            )
            data["actions"] = torch.cat([data["actions"], pad], dim=0)
        elif data["actions"].shape[0] > data["states"].shape[0]:
            data["actions"] = data["actions"][: data["states"].shape[0]]

        return data

    def __len__(self) -> int:
        return len(self.episode_ids)

    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def __getitem__(self, idx: int):
        episode_index = self.episode_ids[idx]
        position = self._episode_pos[episode_index]
        start = self.dataset.episode_data_index["from"][position].item()
        end = self.dataset.episode_data_index["to"][position].item()

        data = self._load_episode_structured(start, end)
        if "visuals" in data:
            visuals = data.pop("visuals")
        else:
            visuals = self._load_episode_visuals(episode_index, data["timestamps"])

        states = data["states"]
        actions = data["actions"]

        if self.normalize_action:
            actions = (actions - self.action_mean) / self.action_std
            states = (states - self.state_mean) / self.state_std

        proprio = states

        obs = {"visual": visuals, "proprio": proprio}
        info = {"episode_index": episode_index}
        return obs, actions, states, info


class _EmptyTrajDataset(TrajDataset):
    """A minimal placeholder so that open-loop rollouts are skipped for Libero.

    val() checks len(train_traj_dset) > 0; we set this to 0.
    """

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Empty trajectory dataset")

    def get_seq_length(self, idx):
        return 0


def load_libero_slice_train_val(
    transform,
    # repo_id: str = "physical-intelligence/libero",
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
    """
    Factory to create Libero windowed datasets for train/val that match the
    codebase's expected interface: returns (datasets, traj_dset).

    Note: traj_dset entries are empty placeholders to disable open-loop
    rollouts (which require full-episode access not provided by
    LeRobotDataset). The training loop will still run standard validation.
    """
    num_frames = num_hist + num_pred
    assert num_frames > 0, "num_hist + num_pred must be > 0"

    # Determine episode split (randomized)
    total_eps = LeRobotDatasetMetadata(repo_id).total_episodes
    n_train = int(split_ratio * total_eps)
    indices = np.arange(total_eps)
    # np.random.shuffle(indices)
    train_eps = indices[:n_train].tolist()
    val_eps = indices[n_train:].tolist()

    meta_data = LeRobotDatasetMetadata(repo_id)
    train_ds = LiberoDataset(
        repo_id,
        episodes=train_eps,
        seq_len=num_frames,
        frameskip=frameskip,
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        transform=transform,
        normalize_action=normalize_action,
    )
    val_ds = LiberoDataset(
        repo_id,
        episodes=val_eps,
        frameskip=frameskip,
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        transform=transform,
        normalize_action=normalize_action,
    )

    train_traj = LiberoTrajectoryDataset(
        repo_id=repo_id,
        episodes=train_eps,
        frameskip=frameskip,
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        transform=transform,
        normalize_action=normalize_action,
        base_dataset=train_ds,
        meta_data=meta_data,
    )

    val_traj = LiberoTrajectoryDataset(
        repo_id=repo_id,
        episodes=val_eps,
        frameskip=frameskip,
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        transform=transform,
        normalize_action=normalize_action,
        base_dataset=val_ds,
        meta_data=meta_data,
    )

    datasets = {"train": train_ds, "valid": val_ds}
    traj_dset = {"train": train_traj, "valid": val_traj}
    # traj_dset = {"train": _EmptyTrajDataset(), "valid": _EmptyTrajDataset()}
    return datasets, traj_dset



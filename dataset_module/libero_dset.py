import torch
import numpy as np
from typing import Optional, Callable, List, Dict

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from einops import rearrange

from .traj_dset import TrajDataset


class LiberoDataset(TrajDataset):
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
        num_frames: int = 20,
        frameskip: int = 1,
        camera_key: str = "observation.images.image",
        state_key: str = "observation.state",
        action_key: str = "action",
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
    ):
        self.transform = transform
        self.camera_key = camera_key
        self.state_key = state_key
        self.action_key = action_key
        self.num_frames = num_frames
        self.normalize_action = normalize_action

        # Build delta timestamps to request a length-T window
        self.meta_data = LeRobotDatasetMetadata(repo_id)
        self.fps = self.meta_data.fps
        delta_timestamps = { # TODO: Ensure the fps is correct
            camera_key: [i * 1/self.fps*frameskip for i in range(num_frames)],
            state_key: [i * 1/self.fps*frameskip for i in range(num_frames)],
            action_key: [i * 1/self.fps*frameskip for i in range(num_frames - 1)],
        }
        self.dataset = LeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
        )

        # Expose dims for trainers
        self.proprio_dim = self.meta_data.features[state_key]["shape"][-1]
        self.state_dim = self.meta_data.features[state_key]["shape"][-1]
        self.action_dim = self.meta_data.features[action_key]["shape"][-1]

        # Default stats (used by planners); will be refined if normalize_action
        self.action_mean = torch.zeros(self.action_dim)
        self.action_std = torch.ones(self.action_dim)
        self.state_mean = torch.zeros(self.state_dim)
        self.state_std = torch.ones(self.state_dim)
        self.proprio_mean = torch.zeros(self.proprio_dim)
        self.proprio_std = torch.ones(self.proprio_dim)

        if self.normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.action_key)
            self.state_mean, self.state_std = self.get_data_mean_std(self.state_key)
            self.proprio_mean = self.state_mean.clone()
            self.proprio_std = self.state_std.clone()

    def get_data_mean_std(self, data_key):
        mean = self.meta_data.stats[data_key]["mean"]
        std = self.meta_data.stats[data_key]["std"]
        if isinstance(mean, np.ndarray):
            mean = torch.tensor(mean, dtype=torch.float32)
        if isinstance(std, np.ndarray):
            std = torch.tensor(std, dtype=torch.float32)
        return mean, std

    # --- TrajDataset API ---
    def get_seq_length(self, idx):
        return self.num_frames

    def __len__(self):
        return len(self.dataset)

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
        sample: Dict[str, torch.Tensor] = self.dataset[idx]

        imgs = sample[self.camera_key]
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        imgs = imgs.to(torch.float32)
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        imgs = self._maybe_to_chw(imgs)
        if self.transform is not None:
            imgs = self.transform(imgs)

        states = sample[self.state_key]
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        states = states.to(torch.float32)

        actions = sample[self.action_key]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.to(torch.float32)
        # actions is (T-1, Da). Pad to (T, Da) by appending zeros on the last step
        if actions.shape[0] == self.num_frames - 1:
            pad = torch.zeros((1, actions.shape[1]), dtype=actions.dtype)
            actions = torch.cat([actions, pad], dim=0)
        actions = self._norm_actions(actions)
        states = self._norm_states(states)

        obs = {"visual": imgs, "proprio": states}
        act = actions
        state = states
        return obs, act, state


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
    repo_id: str = "lerobot/libero_goal_image",
    n_rollout: Optional[int] = None,
    split_ratio: float = 0.9,
    camera_key: str = "observation.images.image",
    state_key: str = "observation.state",
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
    np.random.shuffle(indices)
    train_eps = indices[:n_train].tolist()
    val_eps = indices[n_train:].tolist()

    train_ds = LiberoDataset(
        repo_id=repo_id,
        episodes=train_eps,
        num_frames=num_frames,
        frameskip=frameskip,
        camera_key=camera_key,
        state_key=state_key,
        action_key="action",
        transform=transform,
        normalize_action=normalize_action,
    )

    val_ds = LiberoDataset(
        repo_id=repo_id,
        episodes=val_eps,
        num_frames=num_frames,
        frameskip=frameskip,
        camera_key=camera_key,
        state_key=state_key,
        action_key="action",
        transform=transform,
        normalize_action=normalize_action,
    )

    datasets = {"train": train_ds, "valid": val_ds}
    traj_dset = {"train": _EmptyTrajDataset(), "valid": _EmptyTrajDataset()}
    return datasets, traj_dset



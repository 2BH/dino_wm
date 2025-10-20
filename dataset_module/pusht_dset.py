import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset, WaypointSlicerDataset, WaypointOnlySlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True, # agent's velocity
        use_precomputed_embeddings: bool = False,
        embeddings_subdir: str = "dinov2_embeddings",
        use_waypoints: bool = False,
        waypoints_subdir: str = "waypoints.pkl",
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.embeddings_subdir = embeddings_subdir
        self.use_waypoints = use_waypoints
        self.waypoints_subdir = waypoints_subdir
        
        if self.use_precomputed_embeddings:
            self.embeddings_path = self.data_path / embeddings_subdir
            if not self.embeddings_path.exists():
                raise ValueError(f"Embeddings path {self.embeddings_path} does not exist")
        
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        
        # Load waypoints if enabled
        if self.use_waypoints:
            # Load waypoints from a single dictionary file
            # Format: {episode_id: [list of waypoint indices]}
            waypoints_file = self.data_path / self.waypoints_subdir
            if not waypoints_file.exists():
                raise ValueError(f"Waypoints file {waypoints_file} does not exist")
            
            with open(waypoints_file, 'rb') as f:
                self.waypoints = pickle.load(f)
            
            # Verify that waypoints exist for the episodes we're loading
            for idx in range(n):
                if idx not in self.waypoints:
                    raise ValueError(f"Episode {idx} not found in waypoints dictionary")
            
        else:
            self.waypoints = None
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        obs = {"proprio": proprio}
        
        # Load actual images (always load for decoder loss or display)
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        images = reader.get_batch(frames)  # THWC
        images = images / 255.0
        images = rearrange(images, "T H W C -> T C H W")
        if self.transform:
            images = self.transform(images)
        
        if self.use_precomputed_embeddings:
            # Load pre-computed embeddings
            embeddings_file = self.embeddings_path / f"episode_{idx:03d}.pth"
            embeddings = torch.load(embeddings_file)  # Shape: (T, num_patches, emb_dim)
            embeddings_frames = embeddings[frames]  # Shape: (num_frames, num_patches, emb_dim)
            obs["visual"] = embeddings_frames
            obs["visual_raw"] = images
        else:
            obs["visual"] = images
            obs["visual_raw"] = images
        
        info = {'shape': shape}
        if self.waypoints is not None:
            info['waypoints'] = self.waypoints[idx]
        return obs, act, state, info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0

    


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    use_precomputed_embeddings=False,
    embeddings_subdir="dinov2_embeddings",
    use_waypoints=False,
    waypoints_subdir="waypoints",
    waypoint_mode="frameskip",  # "frameskip" or "waypoint_only"
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        use_precomputed_embeddings=use_precomputed_embeddings,
        embeddings_subdir=embeddings_subdir,
        use_waypoints=use_waypoints,
        waypoints_subdir=waypoints_subdir,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        use_precomputed_embeddings=use_precomputed_embeddings,
        embeddings_subdir=embeddings_subdir,
        use_waypoints=use_waypoints,
        waypoints_subdir=waypoints_subdir,
    )

    num_frames = num_hist + num_pred
    
    if use_waypoints:
        if waypoint_mode == "waypoint_only":
            print("Preparing waypoint only slices")
            # Use only waypoint frames (no intermediate frames, no frameskip)
            train_slices = WaypointOnlySlicerDataset(train_dset, num_hist)
            val_slices = WaypointOnlySlicerDataset(val_dset, num_hist)
            print("Prepared waypoint only slices")
        elif waypoint_mode == "frameskip":
            # Use waypoint-based slicing with frameskip for intermediate frames
            train_slices = WaypointSlicerDataset(train_dset, num_hist, frameskip)
            val_slices = WaypointSlicerDataset(val_dset, num_hist, frameskip)
        else:
            raise ValueError(f"Unknown waypoint_mode: {waypoint_mode}. Use 'frameskip' or 'waypoint_only'")
    else:
        # Use standard trajectory slicing
        train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
        val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset
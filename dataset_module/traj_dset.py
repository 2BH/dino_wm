import abc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange

# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

class TrajSubset(TrajDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.slices = []
        for i in range(len(self.dataset)): 
            T = self.dataset.get_seq_length(i)
            if T - num_frames < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(T - num_frames * frameskip + 1)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        self.slices = np.random.permutation(self.slices)
        
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, _ = self.dataset[i]
        for k, v in obs.items():
            obs[k] = v[start:end:self.frameskip]
        state = state[start:end:self.frameskip]
        act = act[start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    print(
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    )
    return [
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_slices = TrajSlicerDataset(train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip)
    return train, val, train_slices, val_slices


class WaypointOnlySlicerDataset(TrajDataset):
    """
    Dataset that samples ONLY waypoint frames (no intermediate frames).
    
    Uses actual waypoint indices from annotations, ignoring frameskip.
    Example: waypoints=[7, 58, 132, 150], num_hist=3
        → history: [7, 58, 132], target: [150]
        → delta actions: [7→58, 58→132, 132→150]
    """
    def __init__(
        self,
        dataset: TrajDataset,
        num_hist: int,
    ):
        self.dataset = dataset
        self.num_hist = num_hist
        self.slices = []
        
        # Generate slices based on waypoints only
        for traj_idx in range(len(self.dataset)):
            waypoints = self.dataset.waypoints[traj_idx]
            
            # Need at least num_hist + 1 waypoints
            if len(waypoints) < self.num_hist + 1:
                print(f"Ignored trajectory #{traj_idx}: only {len(waypoints)} waypoints, need {self.num_hist + 1}")
                continue
            
            # Create sliding window over waypoints
            # E.g., waypoints=[7,58,132,150], num_hist=3 → [7,58,132]→150
            for i in range(len(waypoints) - self.num_hist):
                history_waypoints = waypoints[i:i + self.num_hist]
                target_waypoint = waypoints[i + self.num_hist]
                
                # Store: (traj_idx, list of history waypoint indices, target waypoint index)
                self.slices.append((traj_idx, history_waypoints, target_waypoint))
        
        print(f"Generated {len(self.slices)} waypoint-only slices from {len(self.dataset)} trajectories")
        print(f"  num_hist={num_hist}, using only waypoint frames")
        
        self.proprio_dim = self.dataset.proprio_dim
        self.state_dim = self.dataset.state_dim
        self.action_dim = 2  # Delta action is 2D (x, y position delta)
    
    def get_seq_length(self, idx: int) -> int:
        return self.num_hist
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        traj_idx, history_waypoints, target_waypoint = self.slices[idx]
        obs, act, state, info = self.dataset[traj_idx]
        
        # Combine history waypoints and target into one array
        all_waypoints = history_waypoints + [target_waypoint]
        waypoints_array = np.array(all_waypoints, dtype=np.int64)
        
        # Vectorized observation and state collection at waypoint indices
        obs_full = {}
        for key in obs.keys():
            obs_full[key] = obs[key][waypoints_array]
        state_full = state[waypoints_array]
        
        # Compute delta actions between consecutive waypoints (vectorized)
        positions = state_full[:, :2]  # (num_hist+1, 2)
        delta_actions = positions[1:] - positions[:-1]  # (num_hist, 2)
        
        return obs_full, delta_actions, state_full


class WaypointSlicerDataset(TrajDataset):
    """
    Dataset that generates slices based on waypoints with teacher forcing.
    
    With frameskip, selects frames at intervals (e.g., T1, T3, T5, T7) and adds waypoint.
    Computes delta actions between consecutive selected frames:
    - T1->T3, T3->T5, T5->T7, T7->waypoint
    
    Each sample returns num_hist+1 frames (history frames + waypoint) and num_hist delta actions.
    """
    def __init__(
        self,
        dataset: TrajDataset,
        num_hist: int,
        frameskip: int = 1,
    ):
        self.dataset = dataset
        self.num_hist = num_hist
        self.frameskip = frameskip
        self.slices = []
        
        # Generate slices based on waypoints
        for traj_idx in range(len(self.dataset)):
            obs, act, state, info = self.dataset[traj_idx]
            traj_len = self.dataset.get_seq_length(traj_idx)
            
            # Check trajectory length
            if self.num_hist * frameskip + 1 > traj_len:
                print(f"Ignored short sequence #{traj_idx}: len={traj_len}, num_hist={num_hist}, frameskip={frameskip}")
                continue
            
            # Check if waypoints exist
            if 'waypoints' not in info or info['waypoints'] is None:
                print(f"Warning: No waypoints for trajectory {traj_idx}, skipping")
                continue
                
            waypoints = info['waypoints']
            if isinstance(waypoints, torch.Tensor):
                waypoints = waypoints.tolist()
            
            # Convert to numpy array once for efficient operations
            waypoints_np = np.array(waypoints)

            for start_t in range(traj_len - self.num_hist * self.frameskip):
                # Calculate the frame indices we would select with frameskip
                # E.g., if start_t=0, num_hist=4, frameskip=2: [0, 2, 4, 6]
                history_frames_idx = [start_t + i * self.frameskip for i in range(self.num_hist)]
                last_hist_frame = history_frames_idx[-1]
                
                # Find the next waypoint after the last history frame (vectorized)
                next_waypoints = waypoints_np[waypoints_np > last_hist_frame]
                if len(next_waypoints) == 0:
                    continue
                next_waypoint_idx = int(next_waypoints.min())  # Convert to int for consistency
                
                # Store: (traj_idx, history_frame_indices, waypoint_idx)
                self.slices.append((traj_idx, history_frames_idx, next_waypoint_idx))
        
        print(f"Generated {len(self.slices)} waypoint slices from {len(self.dataset)} trajectories")
        print(f"  num_hist={num_hist}, frameskip={frameskip}")
        
        self.proprio_dim = self.dataset.proprio_dim
        self.state_dim = self.dataset.state_dim
        # Delta action is 2D (x, y position delta)
        self.action_dim = 2
    
    def get_seq_length(self, idx: int) -> int:
        return self.num_hist
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        traj_idx, history_frames, waypoint_idx = self.slices[idx]
        obs, act, state, info = self.dataset[traj_idx]
        
        # Combine history frames and waypoint into one array for vectorized indexing
        all_frames = history_frames + [waypoint_idx]
        frames_array = np.array(all_frames, dtype=np.int64)  # Ensure int type for indexing
        
        # Vectorized observation and state collection (single indexing operation)
        obs_full = {}
        for key in obs.keys():
            obs_full[key] = obs[key][frames_array]  # Direct indexing with numpy array
        state_full = state[frames_array]  # Direct indexing with numpy array
        
        # Compute delta actions between consecutive frames (vectorized)
        # Extract positions: state[frames, :2] → (num_frames, 2)
        positions = state_full[:, :2]  # (num_hist+1, 2) #TODO: Currently hard coded for only 2D state and PushT Task
        
        # Delta actions: diff between consecutive positions
        # positions[1:] - positions[:-1] → (num_hist, 2)
        delta_actions = positions[1:] - positions[:-1]
        
        return obs_full, delta_actions, state_full
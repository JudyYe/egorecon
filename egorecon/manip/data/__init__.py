"""
Data subpackage for hand and object manipulation datasets.

Contains data loading, processing, and utility functions.
"""

from pytorch3d.structures.meshes import Meshes, join_meshes_as_batch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map


def build_dataset(ds_name, opt, data_cfg, is_train=True, load_obs=False):
    ds_type = opt.get('ds_type', 'w_motion')
    if ds_type == "w_geom":
        from .hand_to_object_dataset_w_geometry import HandToObjectDataset as Dataset
    elif ds_type == "w_motion":
        from .hand2obj_w_geom_motion import HandToObjectDataset as Dataset

    split = data_cfg.trainsplit if is_train else data_cfg.testsplit
    ds = Dataset(
        data_path=data_cfg.data_path,
        is_train=is_train,  # just to decide minimal length
        window_size=opt.datasets.window,
        single_demo=data_cfg.demo_id,
        single_object=data_cfg.target_object_id,
        sampling_strategy="random",
        split=split,
        noise_scheme=data_cfg.get("noise_scheme", "syn"),
        opt=opt,
        t0=300,
        one_window=not is_train,
        data_cfg=data_cfg,
        load_obs=load_obs,
        **opt.datasets.augument,
    )
    ds.set_metadata()
    return ds


def build_dataloader(
    data_cfg, opt, is_train=True, shuffle=None, batch_size=None, num_workers=4, 
    load_obs=False,
):
    ds = build_dataset(data_cfg.name, opt, data_cfg, is_train=is_train, load_obs=load_obs)

    if shuffle is None:
        shuffle = is_train  # train -> shuffle
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_meshes,
    )
    return dataloader, ds


def collate_meshes_fn(batch, *, collate_fn_map=None):
    """
    Collate function specifically for Meshes objects.
    Uses join_meshes_as_batch to combine multiple Meshes into a single batched Meshes object.
    """
    return join_meshes_as_batch(batch)


def collate_meshes(batch):
    """
    Custom collate function that handles Meshes objects properly.
    Uses the collate_fn_map system to register Meshes collation.
    """
    # Create a custom collate function map that includes Meshes handling
    custom_collate_fn_map = default_collate_fn_map.copy()
    custom_collate_fn_map[Meshes] = collate_meshes_fn

    return collate(batch, collate_fn_map=custom_collate_fn_map)

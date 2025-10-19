
"""
Data subpackage for hand and object manipulation datasets.

Contains data loading, processing, and utility functions.
"""
from torch.utils.data import DataLoader
from jutils import mesh_utils


def build_dataset(ds_name, opt, data_cfg, is_train=True):
    if opt.ds_type == 'w_geom':
        from .hand_to_object_dataset_w_geometry import HandToObjectDataset as Dataset
    elif opt.ds_type == 'w_motion':
        from .hand2obj_w_geom_motion import HandToObjectDataset as Dataset

    split = data_cfg.trainsplit if is_train else data_cfg.testsplit
    ds = Dataset(
        data_path=data_cfg.data_path,
        is_train=is_train,  # just to decide minimal length 
        window_size=opt.model.window,
        single_demo=data_cfg.demo_id,
        single_object=data_cfg.target_object_id,
        sampling_strategy='random',
        split=split,
        noise_scheme=data_cfg.get('noise_scheme', 'syn'),
        opt=opt,
        t0=300,
        one_window=not is_train,
        data_cfg=data_cfg,
        **opt.datasets.augument,
    )
    ds.set_metadata()
    return ds


def build_dataloader(data_cfg, opt, is_train=True, shuffle=None, batch_size=None, num_workers=None):
    ds = build_dataset(data_cfg.name, opt, data_cfg, is_train=is_train)

    if shuffle is None:
        shuffle = is_train  # train -> shuffle
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mesh_utils.collate_meshes,
    )
    return dataloader, ds




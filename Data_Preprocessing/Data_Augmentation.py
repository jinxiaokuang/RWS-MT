import numpy as np
from copy import deepcopy
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import SpatialTransform, GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.3,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 2,
    "num_cached_per_thread": 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)


def get_default_augmentation(dataloader_train, dataloader_val, patch_size, params=default_2D_augmentation_params,
                             border_val_seg=-1, pin_memory=False,
                             seeds_train=None, seeds_val=None):
    tr_transforms = []

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"), seeds=seeds_train,
                                                  pin_memory=pin_memory)

    batchgenerator_val = None
    if dataloader_val is not None:
        val_transforms = []
        val_transforms.append(RenameTransform('seg', 'target', True))
        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)

        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"), seeds=seeds_val,
                                                    pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val






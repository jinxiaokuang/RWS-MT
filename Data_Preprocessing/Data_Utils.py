import numpy as np
import nibabel as nib
from medpy.metric import dc, hd
from sklearn.model_selection import KFold
from collections import OrderedDict


def convert_to_one_hot(seg):    # (slices, width, height)
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)    # (channels, slices, width, height)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    # res = np.moveaxis(res, 0, 1)    # convert to (slices, channels, width, height)
    return res


def normalize_img(img, eps=1e-8):
    m = np.mean(img)
    std = np.std(img)
    return (img - m + eps) / (std + eps)


def load_nii(img_path, reorient=False):
    nimg = nib.load(img_path)
    if reorient:
        nimg = nib.as_closest_canonical(nimg)
    return nimg.get_fdata(), nimg.affine, nimg.header    # method get_data() has been abandoned


def get_orientation(affine):
    return nib.aff2axcodes(affine)


def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def windwo_transform(ct_arry,wind_width,wind_center):
    min_window = float(wind_center) - 0.5 * float(wind_width)
    max_window = float(wind_center) + 0.5 * float(wind_width)
    ct_arry = np.clip(ct_arry,min_window,max_window)
    return ct_arry

def normalize_img_after_windowtransform(img,window_center,window_width,eps=1e-8):
    min_window = float(window_center) - 0.5 * float(window_width)
    max_window = float(window_center) + 0.5 * float(window_width)
    img_mask = img[np.where((img > min_window)&(img < max_window))]
    m = np.mean(img_mask)
    std = np.std(img_mask)
    return (img - m + eps) / (std + eps)

def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volpred-volgt]

    return res


def split_data(patient_list, K=5, shuffle=False):
    """
    :param patient_list:
    :param K: K-fold cross-validation
    :return: 5 train-val pairs
    """
    splits = []
    # sort patient_list to ensure the splited data unchangeable every time.
    patient_list.sort()
    # k-fold, I think it doesn't matter whether the shuffle is true or not
    kfold = KFold(n_splits=K, shuffle=shuffle, random_state=12345)
    for i, (train_idx, test_idx) in enumerate(kfold.split(patient_list)):
        train_keys = np.array(patient_list)[train_idx]
        test_keys = np.array(patient_list)[test_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    return splits


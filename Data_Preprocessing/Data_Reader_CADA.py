import os
import numpy as np
import json
from Config.Data_Config import ori_data_path, ori_label_path, train_list_path, val_list_path #, xls_path, img_name, gt_name
from Data_Preprocessing.Data_Utils import convert_to_one_hot, normalize_img, load_nii, get_orientation,windwo_transform,normalize_img_after_windowtransform

# def get_labeled_data(norm=True, one_hot=False):
#     paths = os.listdir(ori_data_path)
#     paths.sort()
#     # calculate statistics
#     imgs = []
#     for path in paths:
#         img_path = os.path.join(ori_data_path, path)
#         img, img_affine, img_header = load_nii(img_path)   
#         img = np.transpose(img, (2, 0, 1)) 
#         imgs.append(img)
        
#     imgs_data = np.vstack(imgs)
#     clip_min = np.percentile(imgs_data, 0.5)
#     clip_max = np.percentile(imgs_data, 99.5)
#     print(clip_max)
#     print(clip_min)
    
#     # read center file
#     with open(center_file_path, 'r') as f:
#         js_cont = json.load(f)
#         center_dict = {}
#         for item in js_cont:
#             store_id = '{}_orig.nii.gz'.format(item['dataset_id'])
#             if store_id not in center_dict.keys():
#                 center_dict[store_id] = []
#             center_dict[store_id].append(item['position_in_index_coordinates'])  # x,y,s
#     # read train / val list file
#     with open(train_list_path, 'r') as f:
#         train_list = f.readlines()
#         train_list = [x.rstrip('\n') for x in train_list]
#     with open(val_list_path, 'r') as f:
#         val_list = f.readlines()
#         val_list = [x.rstrip('\n') for x in val_list]
#     dataset_train = {}
#     dataset_val = {}
#     for i, path in enumerate(paths):
#         img_path = os.path.join(ori_data_path, path)
#         gt_path = os.path.join(ori_label_path, path.replace('orig', 'masks'))
#         img = imgs[i]
#         # clip 
#         np.clip(img, clip_min, clip_max)
#         gt, gt_affine, gt_header = load_nii(gt_path)   
#         gt = np.transpose(gt, (2, 0, 1))  
        
#         if path in train_list:
#             dataset_train[path] = {}
#             dataset_train[path]['center'] = center_dict[path]
#             dataset_train[path]['img'] = normalize_img(img) if norm else img
#             dataset_train[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
#             dataset_train[path]['nii_'] = [gt_affine, gt_header]
#         elif path in val_list:
#             dataset_val[path] = {}
#             dataset_val[path]['center'] = center_dict[path]
#             dataset_val[path]['img'] = normalize_img(img) if norm else img
#             dataset_val[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
#             dataset_val[path]['nii_'] = [gt_affine, gt_header]
#         else:
#             raise Exception('sample ID not in train_list nor val_list')
        
#     return dataset_train, dataset_val

def get_labeled_data(norm=True, one_hot=False,window_width=700,window_center=80):
    paths = os.listdir(ori_data_path)
    paths.sort()
    paths = paths[:14]       # only choose 14 paths, in order to train/test faster
    # calculate statistics
    imgs = []
    for path in paths:
        img_name = '{}.nii.gz'.format(path)
        img_path = os.path.join(ori_data_path, path, img_name)
        img, img_affine, img_header = load_nii(img_path)   
        img = np.transpose(img, (2, 0, 1)) 
        imgs.append(img)
        
    imgs_data = np.vstack(imgs)
    clip_min = np.percentile(imgs_data, 0.5)
    clip_max = np.percentile(imgs_data, 99.5)
    print(clip_max)
    print(clip_min)
    
    # read train / val list file
    with open(train_list_path, 'r') as f:
        train_list = f.readlines()
        train_list = [x.rstrip('\n') for x in train_list]
    with open(val_list_path, 'r') as f:
        val_list = f.readlines()
        val_list = [x.rstrip('\n') for x in val_list]
    dataset_train = {}
    dataset_val = {}
    for i, path in enumerate(paths):
        img_name = '{}.nii.gz'.format(path)
        img_path = os.path.join(ori_data_path, path, img_name)
        gt_name = '{}_gt.nii.gz'.format(path)
        gt_path = os.path.join(ori_label_path, path, gt_name)
        img = imgs[i]
        # clip 
        # np.clip(img, clip_min, clip_max)
        #window_transform
        img = windwo_transform(img,window_width,window_center)
        gt, gt_affine, gt_header = load_nii(gt_path)   
        gt = np.transpose(gt, (2, 0, 1))  
        
        if path in train_list:
            dataset_train[path] = {}
            dataset_train[path]['center'] = [[128,128,128]]
            dataset_train[path]['img'] = normalize_img_after_windowtransform(img,window_center,window_width) if norm else img
            dataset_train[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
            dataset_train[path]['nii_'] = [gt_affine, gt_header]
        elif path in val_list:
            dataset_val[path] = {}
            dataset_val[path]['center'] = [[128,128,128]]
            dataset_val[path]['img'] = normalize_img_after_windowtransform(img,window_center,window_width) if norm else img
            dataset_val[path]['gt'] = convert_to_one_hot(gt) if one_hot else gt
            dataset_val[path]['nii_'] = [gt_affine, gt_header]
        else:
            # raise Exception('sample ID not in train_list nor val_list')
            continue
        
    return dataset_train, dataset_val

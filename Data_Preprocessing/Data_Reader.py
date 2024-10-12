import os
import numpy as np
import xlrd
from Config.Data_Config import labeled_ori_data_path, unlabeled_ori_data_path, xls_path, img_name, gt_name
from Data_Preprocessing.Data_Utils import convert_to_one_hot, normalize_img, load_nii, get_orientation


def get_data_description():
    xls_data = xlrd.open_workbook(xls_path)
    table = xls_data.sheet_by_name(xls_data.sheet_names()[0])
    data_description = {}
    for row in range(1, table.nrows):
        content = table.row_values(row)
        data_description[content[0]] = {}
        data_description[content[0]]['Vendor'] = content[1]
        data_description[content[0]]['Centre'] = int(content[2])
        data_description[content[0]]['ED'] = int(content[3])
        data_description[content[0]]['ES'] = int(content[4])
    return data_description


def get_data_description_by_vendor(vendor='A'):
    xls_data = xlrd.open_workbook(xls_path)
    table = xls_data.sheet_by_name(xls_data.sheet_names()[0])
    data_description = {}
    for row in range(1, table.nrows):
        content = table.row_values(row)
        # print(content)  # ['A0S9V9', 'A', 1.0, 0.0, 9.0]
        if content[1] == vendor:
            data_description[content[0]] = {}
            data_description[content[0]]['Vendor'] = content[1]
            data_description[content[0]]['Centre'] = int(content[2])
            data_description[content[0]]['ED'] = int(content[3])
            data_description[content[0]]['ES'] = int(content[4])
    return data_description


def get_labeled_data(norm=True, one_hot=False):
    data_description = get_data_description()
    paths = os.listdir(labeled_ori_data_path)
    paths.sort()
    dataset = {}
    dataset['A'] = {}   # vendor A
    dataset['B'] = {}   # vendor B
    for path in paths:
        img_path = os.path.join(labeled_ori_data_path, path, img_name.format(path))
        gt_path = os.path.join(labeled_ori_data_path, path, gt_name.format(path))
        img, img_affine, img_header = load_nii(img_path)    # (216, 256, 13, 25)
        img = np.transpose(img, (3, 2, 0, 1))   # (25, 13, 216, 256)
        gt, gt_affine, gt_header = load_nii(gt_path)    # (216, 256, 13, 25)
        gt = np.transpose(gt, (3, 2, 0, 1))   # (25, 13, 216, 256)
        # print(img.shape, gt.shape)  # (25, 13, 216, 256) (25, 13, 216, 256)

        vendor = data_description[path]['Vendor']
        centre = data_description[path]['Centre']
        ed = data_description[path]['ED']
        es = data_description[path]['ES']
        dataset[vendor][path] = {}
        dataset[vendor][path]['Centre'] = centre
        dataset[vendor][path]['ED'] = normalize_img(img[ed]) if norm else img[ed]
        dataset[vendor][path]['ED_GT'] = convert_to_one_hot(gt[ed]) if one_hot else gt[ed]
        dataset[vendor][path]['ES'] = normalize_img(img[es]) if norm else img[es]
        dataset[vendor][path]['ES_GT'] = convert_to_one_hot(gt[es]) if one_hot else gt[es]
        # print('**', dataset[vendor][path]['ED'].shape, np.mean(dataset[vendor][path]['ED']), np.mean(img[ed]), dataset[vendor][path]['ED_GT'].shape, np.unique(dataset[vendor][path]['ED_GT']), dataset[vendor][path]['ES'].shape)
        # ** (13, 216, 256) 5.056206e-08 172.60858 (13, 4, 216, 256) [0. 1.] (13, 216, 256)
    return dataset


def get_labeled_data_by_vendors(vendor='A', norm=True, one_hot=True):
    data_description = get_data_description()
    paths = os.listdir(labeled_ori_data_path)
    paths.sort()
    dataset = {}
    for path in paths:
        img_path = os.path.join(labeled_ori_data_path, path, img_name.format(path))
        gt_path = os.path.join(labeled_ori_data_path, path, gt_name.format(path))
        img, img_affine, img_header = load_nii(img_path)    # (216, 256, 13, 25)
        img = np.transpose(img, (3, 2, 0, 1))   # (25, 13, 216, 256)
        gt, gt_affine, gt_header = load_nii(gt_path)    # (216, 256, 13, 25)
        gt = np.transpose(gt, (3, 2, 0, 1))   # (25, 13, 216, 256)
        # print(img.shape, gt.shape)  # (25, 13, 216, 256) (25, 13, 216, 256)

        cur_vendor = data_description[path]['Vendor']
        if cur_vendor == vendor:
            centre = data_description[path]['Centre']
            ed = data_description[path]['ED']
            es = data_description[path]['ES']
            dataset[path] = {}
            dataset[path]['Centre'] = centre
            dataset[path]['ED'] = normalize_img(img[ed]) if norm else img[ed]
            dataset[path]['ED_GT'] = convert_to_one_hot(gt[ed]) if one_hot else gt[ed]
            dataset[path]['ES'] = normalize_img(img[es]) if norm else img[es]
            dataset[path]['ES_GT'] = convert_to_one_hot(gt[es]) if one_hot else gt[es]
    return dataset


def get_labeled_data_by_patient_list(vendor='A', patient_list=None, norm=True, one_hot=False):
    data_description = get_data_description()
    paths = os.listdir(labeled_ori_data_path)
    paths.sort()
    dataset = {}
    for pat in patient_list:
        img_path = os.path.join(labeled_ori_data_path, pat, img_name.format(pat))
        gt_path = os.path.join(labeled_ori_data_path, pat, gt_name.format(pat))
        img, img_affine, img_header = load_nii(img_path)    # (216, 256, 13, 25)
        img = np.transpose(img, (3, 2, 0, 1))   # (25, 13, 216, 256)
        gt, gt_affine, gt_header = load_nii(gt_path)    # (216, 256, 13, 25)
        gt = np.transpose(gt, (3, 2, 0, 1))   # (25, 13, 216, 256)
        # print(img.shape, gt.shape)  # (25, 13, 216, 256) (25, 13, 216, 256)

        cur_vendor = data_description[pat]['Vendor']
        if cur_vendor == vendor:
            centre = data_description[pat]['Centre']
            ed = data_description[pat]['ED']
            es = data_description[pat]['ES']
            dataset[pat] = {}
            dataset[pat]['Centre'] = centre
            dataset[pat]['ED'] = normalize_img(img[ed]) if norm else img[ed]
            dataset[pat]['ED_GT'] = convert_to_one_hot(gt[ed]) if one_hot else gt[ed]
            dataset[pat]['ES'] = normalize_img(img[es]) if norm else img[es]
            dataset[pat]['ES_GT'] = convert_to_one_hot(gt[es]) if one_hot else gt[es]
    return dataset







# data_description = get_data_description()
# print(data_description['A6D5F9']) # {'ES': 11, 'Vendor': 'A', 'Centre': 1, 'ED': 0}

# dataset = get_labeled_data_by_vendors('A', True, True)
# print(len(dataset.keys()))
# print(len(dataset['A'].keys()))
# print(dataset['A']['Q3R9W7']['centre'])
# print(dataset['A']['Q3R9W7']['ED'].shape)
# print(dataset['A']['Q3R9W7']['ED_GT'].shape)
# print(dataset['A']['Q3R9W7']['ES'].shape)
# print(dataset['A']['Q3R9W7']['ES_GT'].shape)
#
# print(len(dataset['B'].keys()))
# print(dataset['B']['Q7V1Y5']['centre'])
# print(dataset['B']['Q7V1Y5']['ED'].shape)
# print(dataset['B']['Q7V1Y5']['ED_GT'].shape)
# print(dataset['B']['Q7V1Y5']['ES'].shape)
# print(dataset['B']['Q7V1Y5']['ES_GT'].shape)


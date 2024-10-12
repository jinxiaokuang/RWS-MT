from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import random_crop_3D_image_batched, pad_nd_image
import numpy as np
import random

class DataGenerator2D(SlimDataLoaderBase):
    def __init__(self, data, vendor='A', patch_size=(224, 224), batch_size=1, num_classes=4):
        super(DataGenerator2D, self).__init__(data, batch_size)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.vendor = vendor
        if self.vendor is None:
            self.patients = list(self._data.keys())
        else:
            self.patients = list(self._data[self.vendor].keys())  # patients' ids (list type)

    def generate_train_batch(self):
        data = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.uint8)
        selected_pats = np.random.choice(self.patients, self.batch_size, True, None)
        id = 0
        for pat in selected_pats:
            selected_phase = np.random.choice(['ED', 'ES'])
            if self.vendor is None:
                shp = self._data[pat][selected_phase].shape
            else:
                shp = self._data[self.vendor][pat][selected_phase].shape
            choosen_slice = np.random.choice(range(shp[0]))
            if self.vendor is None:
                img_instance = self._data[pat][selected_phase][choosen_slice]  # (width, height)
                seg_instance = self._data[pat][selected_phase + '_GT'][choosen_slice]  # (num_classes, width, height)
            else:
                img_instance = self._data[self.vendor][pat][selected_phase][choosen_slice]  # (width, height)
                seg_instance = self._data[self.vendor][pat][selected_phase + '_GT'][choosen_slice]  # (num_classes, width, height)

            # expand dimensions
            img_instance = np.expand_dims(np.expand_dims(img_instance, axis=0), axis=0)  # (1, 1, width, height)
            seg_instance = np.expand_dims(seg_instance, axis=0)  # (1, num_classes, width, height)
            # stack data and seg
            stacked_img_seg = np.concatenate((img_instance, seg_instance), axis=-3)     # (1, num_classes+1, width, height)
            # pad img in case the shape is smaller than patch_size
            padded_instance = pad_nd_image(stacked_img_seg, self.patch_size)    # (1, num_classes+1, *, *)
            # randomly crop
            cropped_instance = random_crop_2D_image_batched(padded_instance, self.patch_size)  # (1, num_classes+1, *self.patch_size)

            data[id, 0] = cropped_instance[0, 0]
            seg[id, 0] = np.argmax(cropped_instance[0, 1:], axis=0)
            id += 1
        return {'data': data, 'seg': seg}

class DataGenerator3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size=(128, 128, 128), batch_size=1):
        super(DataGenerator3D, self).__init__(data, batch_size, None)
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())

    def generate_train_batch(self):
        data = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.uint8)
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        for i, key in enumerate(selected_keys):
            img_instance = self._data[key]['img']
            seg_instance = self._data[key]['gt']
            # expand dimensions
            img_instance = np.expand_dims(np.expand_dims(img_instance, axis=0), axis=0)  # (1, 1, slices, width, height)
            seg_instance = np.expand_dims(seg_instance, axis=0)  # (1, num_classes, slices, width, height)
            
            # stack data and seg
            stacked_img_seg = np.concatenate((img_instance, seg_instance), axis=1)     # (1, num_classes+1, slices, width, height)
            # pad img in case the shape is smaller than patch_size
            padded_instance = pad_nd_image(stacked_img_seg, self.patch_size)    # (1, num_classes+1, *, *)
            # randomly crop around the center point
            center_pt = random.choice(self._data[key]['center']) # some may contain more than one centers
            lt_x = max(0, center_pt[0] - 64 + random.randint(-30, 30))
            lt_y = max(0, center_pt[1] - 64 + random.randint(-30, 30))
            lt_s = max(0, center_pt[2] - 64 + random.randint(-30, 30))
            rb_x = min(padded_instance.shape[3], lt_x + 128)
            rb_y = min(padded_instance.shape[4], lt_y + 128)
            rb_s = min(padded_instance.shape[2], lt_s + 128)

            cropped_instance = padded_instance[:, :, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]
            cropped_instance = pad_nd_image(cropped_instance, self.patch_size)
            # cropped_instance = random_crop_3D_image_batched(padded_instance, self.patch_size)  # (1, num_classes+1, *self.patch_size)
            
            data[i, 0] = cropped_instance[0, 0]
            seg[i, 0] = np.argmax(cropped_instance[0, 1:], axis=0)
        
        return {'data':data, 'seg':seg, 'keys': selected_keys}



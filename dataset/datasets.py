import torch.utils.data as data
import os
import os.path
from glob import glob
from PIL import Image as Image
import numpy as np


class GL3D(data.Dataset):
    def __init__(self, args):
        # gl3d/scenes/images1
        if  args.train == True:
            self.root = args.traindata_list  # ./datasets/traindata
        else:
            self.root = args.testdata_list


        self.images_dirs = args.dataset_root
        self.images_list = glob(os.path.join(self.root, 'images', '*.txt'))
        self.label = os.path.join(self.root, 'labels')

    def __getitem__(self, index):
        images_dir = self.images_list[index]
        file_name = images_dir.split('/')[-1].split('_')[1]
        # print(file_name)
        part_name = images_dir.split('/')[-1].split('_')[2].split('.')[0]
        # print(file_name)
        with open(images_dir, 'r') as f:
            image_name = f.readlines()

        img_list = []
        for image in image_name:
            img = Image.open(os.path.join( self.images_dirs, file_name, 'undist_images', image[0:-1]))
            img = img.resize((224, 224))
            img = np.array(img) / 255.0
            img = img.transpose((2, 0, 1))
            img_list.append(img)

        label_dir = os.path.join(self.label, 'label_{}_{}.npy'.format(file_name, part_name))
        label = np.load(label_dir)

        return img_list, label

    def __len__(self):
        return len(self.images_list)



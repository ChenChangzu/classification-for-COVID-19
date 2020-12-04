from torch.utils.data import Dataset
from config import config
from dataset.propressing import multiDimNormal
import random
import numpy as np
import pandas as pd
import os
import torch
import nibabel as nib
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 设置随机种子
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# 创建读取nii文件的Dataset
class CreateMultiPETDataset(Dataset):
    def __init__(self, label_list, train=True, test=False):
        self.test = test
        self.train = train
        imgs = []
        if self.test:
            pass
        else:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"]))
            self.imgs = imgs

    def __getitem__(self, index):
        if self.test:
            pass
        else:
            filename, label = self.imgs[index]
            img_contents = []
            for item in os.listdir(filename):
                img_contents.append(multiDimNormal(nib.load(filename+os.sep+item).get_data()))
            img_fdata = np.asarray(img_contents, dtype=float)
            #img_fdata = img_fdata[np.newaxis, :]
            #print(img_fdata.shape)
            img_tensor = torch.from_numpy(img_fdata)
            img_tensor = img_tensor.type(torch.FloatTensor)
            return img_tensor, label
    def __len__(self):
        return len(self.imgs)
# 创建读取nii文件的Datase
class CreateNiiDataset(Dataset):
    def __init__(self, label_list, train=True, test=False):
        self.test = test
        self.train = train
        imgs = []
        if self.test:
            pass
        else:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"]))
            self.imgs = imgs

    def __getitem__(self, index):
        if self.test:
            pass
        else:
            filename, label = self.imgs[index]
            img_contents = nib.load(filename)
            img_fdata = np.asarray(img_contents.get_data())
            img_fdata = img_fdata[np.newaxis, :]
            img_tensor = torch.from_numpy(img_fdata)
            img_tensor = img_tensor.type(torch.FloatTensor)
            return img_tensor, label
    def __len__(self):
        return len(self.imgs)
# 创建读取mat文件的Dataset
class Create3DMatDataset(Dataset):
    def __init__(self, label_list, train=True, test=False):
        self.test = test
        self.train = train
        imgs = []
        if self.test:
            pass
        else:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"]))
            self.imgs = imgs

    def __getitem__(self, index):
        if self.test:
            pass
        else:
            filename, label = self.imgs[index]
            img_contents = loadmat(filename)["data"]
            img_contents = img_contents[np.newaxis, :]
            img_tensor = torch.from_numpy(img_contents)
            img_tensor = img_tensor.type(torch.FloatTensor)
            return img_tensor, label
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root, class_list, mode ):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename": files})
        return files
    elif mode != "test": 
        # for train and val
        # print("loading " + mode + " dataset")
        all_data_path, labels = [], []
        for class_dir in os.listdir(root):
            if class_dir in class_list:
                class_path = os.path.join(root, class_dir)
                for file_dir in os.listdir(class_path):  #tqdm(list(class_path)):  #os.listdir(class_path):
                    all_data_path.append(os.path.join(class_path, file_dir))
                    for class_index in range(0, len(class_list)):
                        if class_dir == class_list[class_index]:
                            labels.append(class_index)
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        if mode == "train":
            all_files = shuffle(all_files)
        return all_files
    else:
        print("check the mode please!")

# 按百分比划分train和val
def random_split_ratio(root, class_list, split_rate):
    x, y = [], []
    for class_dir in os.listdir(root):
        if class_dir in class_list:
            class_path = os.path.join(root, class_dir)
            for sub_dir in os.listdir(class_path):
                x.append(os.path.join(class_path, sub_dir))
                y.append(class_list.index(class_dir))
    train_data_path, val_data_path, train_labels, val_labels = train_test_split(x, y, test_size=split_rate, random_state=config.seed)
    train_files = pd.DataFrame({"filename": train_data_path, "label": train_labels})
    val_files = pd.DataFrame({"filename": val_data_path, "label": val_labels})
    return shuffle(train_files), val_files

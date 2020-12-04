import os
import math
import random
import numpy as np
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
import shutil
import cv2
from tqdm import tqdm
from PIL import Image
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def split_slices(data_list, class_list, slice_type, plane, start, end):
    print(slice_type + ' slices...')
    save_path = './dataset/data/' + slice_type
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for dir in class_list:
        class_path = save_path + '/' + dir
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        else:
            remove(class_path)
    path_list = list(data_list["filename"])
    # print(path_list)
    for i, path in enumerate(tqdm(path_list)):
        # print(path)
        label = data_list["label"][i]
        # print(label)
        save_dir = save_path + '/' + class_list[label]
        temp = path.split('/')
        temp = temp[-1].split('_')
        slice_name = temp[1] + '_' + temp[2] + '_' + temp[3] + '_'
        image_contents = nib.load(path)  # 读取nii
        img_fdata = image_contents.get_fdata()  # 获取nii_file数据
        for j in range(start, end):  # (img_fdata.shape[0]):
            if plane == "sagittal":
                data = img_fdata[j, :, :]  # 选择哪个方向的切片都可以[矢状面、冠状面、轴位面][[][55,95][40,72]]
            elif plane == "coronal":
                data = img_fdata[:, j, :]  # 选择哪个方向的切片都可以[矢状面、冠状面、轴位面][[][55,95][40,72]]
            else:
                data = img_fdata[:, :, j]  # 选择哪个方向的切片都可以[矢状面、冠状面、轴位面][[][55,95][40,72]]
            imageio.imwrite(os.path.join(save_dir, '{}.png'.format(slice_name + str(j))), np.asarray(data*255, dtype='uint8'))


def get_slices_data(source_path, save_path, temp_dir, plane, start, end):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    sets = ['train', 'verify', 'test']
    for i in range(3):
        print('正在处理' + sets[i] + '集...\n')
        file_dir = os.path.join(source_path, sets[i])
        save_dir = os.path.join(save_path, sets[i])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # temp_save_dir = os.path.join(save_dir, 'entropy_temp')
        # os.mkdir(temp_save_dir)
        for label in os.listdir(file_dir):
            class_save_dir = os.path.join(save_dir, label)
            if not os.path.exists(class_save_dir):
                os.mkdir(class_save_dir)
            class_dir = os.path.join(file_dir, label)
            for f_dir in os.listdir(class_dir):
                image_contents = nib.load(os.path.join(class_dir, f_dir))  # 读取nii
                img_fdata = image_contents.get_fdata()  # 获取nii_file数据
                print(img_fdata.shape)
                # 保存切片
                temp = f_dir.split('_')
                slice_name = temp[1]+'_'+temp[2]+'_'+temp[3]+'_'
                print('正在切片'+slice_name+'...')
                for j in range(start, end):  #(img_fdata.shape[0]):
                    if plane == "sagittal":
                        data = img_fdata[j, :, :]  # 选择哪个方向的切片都可以[矢状面、冠状面、轴位面][[][55,95][40,72]]
                    elif plane == "coronal":
                        data = img_fdata[:, j, :]  # 选择哪个方向的切片都可以[矢状面、冠状面、轴位面][[][55,95][40,72]]
                    else:
                        data = img_fdata[:, :, j]  # 选择哪个方向的切片都可以[矢状面、冠状面、轴位面][[][55,95][40,72]]
                    imageio.imwrite(os.path.join(temp_dir, '{}.png'.format(slice_name + str(j))), data)
                # print('正在统计'+sets[i]+'/'+label+'/'+slice_name+'信息熵...')
                # getImgEntropy(temp_dir, temp_save_dir)
            # print('正在裁剪填充' + label + '图像...\n')
        #     cope_padding(temp_save_dir, class_save_dir, 160, 160)  #裁剪填充数据cope_padding(source_path, save_path, width=160, height=160)
        #     remove(temp_save_dir)   #清空数据
        # os.rmdir(temp_save_dir)

        #     cope_padding(temp_dir, class_save_dir, 160, 160)  # 裁剪填充数据cope_padding(source_path, save_path, width=160, height=160)
        #     remove(temp_dir)  # 清空数据

                move(temp_dir, class_save_dir)

def move(temp_dir, save_dir):
    for f in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, f), save_dir)

def remove(temp_dir):
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))  # 删除文件

def getImgEntropy(temp_dir, save_dir):
    file_list = []
    total = 0
    for img_f in os.listdir(temp_dir):
        img_dir = os.path.join(temp_dir, img_f)
        img = cv2.imread(img_dir, 0)
        img = np.array(img)
        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k = float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if (tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        # print(res)
        file_list.append([img_dir, res])
        total += 1
    print(file_list)
    list = sorted(file_list, key=lambda x: x[1])
    print(list)
    for i in range(total):
        # os.remove(list[i][0])  # 删除文件
        if(i >= total-28):
            shutil.move(list[i][0], save_dir)
        elif os.path.exists(list[i][0]):  # 如果文件存在
                os.remove(list[i][0])  # 删除文件

# if __name__ == '__main__':
#     plane = "coronal"  # sagittal/coronal/axial
#     start = 85
#     end = 105
#     root = '/home/image/PyCharmProjects/AD_Classification_Pytorch/dataset/data/sgm_'+plane+'_'+str(start)+'-'+str(end)
#     get_slices_data('/home/image/DataDisk/Data/ADNI_sets/smwp1', root, root+'/temp', plane, start, end)
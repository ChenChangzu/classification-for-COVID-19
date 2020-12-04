import csv
import os
import numpy as np
import shutil
import scipy.io as scio
import nibabel as nib
from tqdm import tqdm
from sklearn import preprocessing

# 将数据分类到所属的标签文件夹中
def moveClassFile(data_root, save_path, csv_path):
    # data_root = "E:\MEGAsync\FDG-PET"
    # save_path = "F:\ADNI\PET"
    # csv_path = "E:\MEGAsync\FDG-PET_3_05_2020.csv"
    dataList = {}
    with open(csv_path, 'r') as f:
        resultList = list(csv.reader(f))
        sub = [result[1] for result in resultList[1:]]
        label = [result[2] for result in resultList[1:]]
        for i in range(len(sub)):
            dataList[sub[i]] = label[i]
        for f in set(label):
            if os.path.exists(save_path+os.sep+f)==False:
                os.mkdir(save_path+os.sep+f)
    total = 1
    num = len(set(sub))
    for sub in os.listdir(data_root):
        total += 1
        print("已搬运"+str(total/num*100)+"% ...")
        shutil.copytree(data_root+os.sep+sub, save_path+os.sep+dataList[sub]+os.sep+sub)

# 将PET数据的帧结合成同一帧
def combinPetFrames(data_root, save_path):
    # data_root = "F:\ADNI\PET"
    # save_path = "F:\ADNI\PETCombin"
    # print("正在读取数据...")
    for cls in os.listdir(data_root):
        class_path = data_root+os.sep+cls
        save_cls_path = save_path+os.sep+cls
        if not os.path.exists(save_cls_path):
            os.mkdir(save_cls_path)
        # print(class_path)
        pbar = tqdm(os.listdir(class_path))
        for sub in pbar:
            pbar.set_description(cls+"->"+sub)
            data_path = class_path+os.sep+sub
            dataList = []
            for i in os.listdir(data_path):
                if i.find("register") != -1:
                    img_contents = nib.load(data_path+os.sep+i)
                    dataList.append(multiDimNormal(np.asarray(img_contents.get_data())))
            data = np.zeros((91, 109, 91))
            # total = 1
            for i in range(91):
                for j in range(109):
                    for k in range(91):
                        # total += 1
                        # if total>=10000 and total%10000 == 0:
                        #     print("已处理"+str(total/(91*109*91)*100)+"%")
                        voxel = []
                        for n in range(len(dataList)):
                            voxel.append(dataList[n][i][j][k])
                        p = np.poly1d(np.polyfit(range(1, len(voxel)+1, 1), voxel, len(voxel)-1))
                        coe = p.coeffs
                        weight = [1-i/10.0 for i in range(len(coe))]
                        for c in range(len(coe)):
                            data[i, j, k] = coe[c]*weight[c]
            scio.savemat(save_cls_path+os.sep+sub+".mat", {'data': data})
# 归一化三维数据
def multiDimNormal(imgs):
    for i, brain_slice in enumerate(imgs):
        brain_slice = (brain_slice - np.mean(brain_slice)) / np.std(brain_slice)
        # 下面的if...else很关键，如果没有这个叠加操作，你会发现for循环结束后imgs里面的数据还是未归一化的数据
        if i == 0:
            imgs = np.reshape(brain_slice, [1, brain_slice.shape[0], brain_slice.shape[1]])
        else:
            imgs = np.concatenate((imgs, np.reshape(brain_slice, [1, brain_slice.shape[0], brain_slice.shape[1]])), axis=0)
    return imgs

if __name__ == '__main__':
    combinPetFrames(r"D:\ADNI\PET", "F:\ADNI\PETCombin")
import csv
import os
import numpy as np
import shutil
import scipy.io as scio
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

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
        pbar = tqdm(os.listdir(class_path))
        for sub in pbar:
            pbar.set_description(cls+"->"+sub)
            data_path = class_path+os.sep+sub
            dataList = []
            for i in os.listdir(data_path):
                if i.find("register") != -1:
                    img_contents = nib.load(data_path+os.sep+i)
                    dataList.append(multiDimNormal(np.asarray(img_contents.get_data())))  # 归一化三位矩阵数据
            data = np.zeros((91, 109, 91))
            for i in range(91):
                for j in range(109):
                    for k in range(91):
                        voxel = []
                        for n in range(len(dataList)):
                            voxel.append(dataList[n][i][j][k])
                        p = np.poly1d(np.polyfit(range(1, len(voxel)+1, 1), voxel, len(voxel)-1))  # 拟合数据点
                        coe = p.coeffs
                        weight = [1-i/10.0 for i in range(len(coe))]
                        for c in range(len(coe)):
                            data[i, j, k] = coe[c]*weight[c]  # 根据衰减，将拟合得到的相关系数加权，实现多个数据点合并为一个数据点
            scio.savemat(save_cls_path+os.sep+sub+".mat", {'data': data})

# 归一化三维矩阵数据
def multiDimNormal(imgs):
    for i, brain_slice in enumerate(imgs):
        brain_slice = (brain_slice - np.mean(brain_slice)) / np.std(brain_slice)
        # 下面的if...else很关键，如果没有这个叠加操作，你会发现for循环结束后imgs里面的数据还是未归一化的数据
        if i == 0:
            imgs = np.reshape(brain_slice, [1, brain_slice.shape[0], brain_slice.shape[1]])
        else:
            imgs = np.concatenate((imgs, np.reshape(brain_slice, [1, brain_slice.shape[0], brain_slice.shape[1]])), axis=0)
    return imgs

# 调用sh文件执行预处理流，如FreeSurfer或FSL
def processing(shFIle, root, save, temp):
    if not os.path.exists(save):
        os.mkdir(save)
    if not os.path.exists(temp):
        os.mkdir(temp)
    for label in os.listdir(root):
        label_path = root+os.sep+label
        save_path = save+os.sep+label
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        pbar = tqdm(os.listdir(label_path))
        for sub in pbar:
            pbar.set_description(label + "->" + sub)
            cmd = "bash " + shFIle \
                  + " " + label_path+os.sep+sub \
                  + " " + save_path+os.sep+sub \
                  + " " + temp + " " + ">/dev/null 2>&1"
            os.system(cmd)

# 批量裁减NII数据无用区域，重采样输出到指定尺寸
def cropNII(root, save, size, padding):
    if not os.path.exists(save):
        os.mkdir(save)
    for label in os.listdir(root):
        label_path = root+os.sep+label
        save_path = save+os.sep+label
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        pbar = tqdm(os.listdir(label_path))
        for sub in pbar:
            pbar.set_description(label + "->" + sub)
            img = sitk.ReadImage(label_path+os.sep+sub)
            data = sitk.GetArrayFromImage(img)
            # plt.imshow(data[:, 110, :], cmap='gray')  # 冠状面、横断面、矢状面
            # plt.show()
            for dim in range(3):
                flag = False
                index = 0
                datat = np.swapaxes(data, 0, dim)
                temp = np.zeros((datat.shape[1], datat.shape[2]))
                for i in range(datat.shape[0]):
                    if not flag and not (datat[i, :, :] == temp).all():
                        index = i-padding
                        flag = True
                    if flag and (datat[i, :, :] == temp).all():
                        data = datat[index:i+padding, :, :]
                        flag = False
                        break
            data = np.swapaxes(data, 0, 1)  # 维度恢复
            data = np.swapaxes(data, 1, 2)
            # data = data[:, :, ::-1]  # 翻转矫正
            sitk.WriteImage(imgResizeResample(sitk.GetImageFromArray(data), size), save_path+os.sep+sub)

# 重采样到指定尺寸
def imgResizeResample(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

if __name__ == '__main__':
    shFIle = "/home/image/PycharmProjects/preprocessing-freesurfer.sh"
    root = "/home/image/MEGAsync/MRI_MPRAGE_1.2mm_1.5T_T1/ADNI"
    save = "/home/image/MEGAsync/MRI_MPRAGE_1.2mm_1.5T_T1/temp"
    temp = "/home/image/MEGAsync/temp"
    processing(shFIle, root, save, temp)

#     combinPetFrames(r"D:\ADNI\PET", "F:\ADNI\PETCombin")
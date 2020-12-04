import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset.dataloader import CreateNiiDataset
from models.Attention_3D_model import Attention_3D_Model_2_4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
num_classes = 3  #类别数
best_model = "./checkpoints/best_model/"
model_name = "attention_2-4_simple_sgm_3Class"
fold = 0
data_path = "/home/image/DataDisk/Data/ADNI1_Complete_1Yr_1.5T_Class_GM/smwp1/CN/smwp1ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.nii"

test_data = pd.DataFrame({"filename": [data_path], "label": [0]})
model = Attention_3D_Model_2_4(num_classes=num_classes)
test_loader = DataLoader(CreateNiiDataset(test_data, train=False), batch_size=1, shuffle=False, pin_memory=False)

best_model = torch.load(best_model + model_name + os.sep + str(fold) + os.sep + "model_best.pth.tar")
model.load_state_dict(best_model["state_dict"])
model.cuda()
model.eval()
label = ''
probability = []
with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        input = Variable(input).cuda()
        target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
        # 2.2.1 计算输出
        output = model(input)
        smax = nn.Softmax(1)
        smax_out = smax(output)
        probability = smax_out.cpu().data.tolist()
        probability = probability[0]
        label = np.argmax(probability)
print("AD/NC/MCI分类概率：")
print(probability)
if label==0:
    print("最终归类为：AD")
elif label==1:
    print("最终归类为：NC")
else:
    print("最终归类为：MCI")
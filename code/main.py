import os
import random
import xlwt
import warnings
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
from config import config
from collections import OrderedDict
from utils.pytorchtools import EarlyStopping
from utils.utils import *
from utils.progress_bar import *
from dataset.dataloader import *
from timeit import default_timer as timer
from models.resnet import *

# 1. 设置random.seed和cudnn性能
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus  # 根据序号选定要使用的GPU
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# 2. 评估函数
def evaluate(test_loader, model, criterion, fold, epoch):
    # 2.1 初始化统计
    losses = AverageMeter()
    top1 = AverageMeter()
    test_progressor = ProgressBar(mode="Test ", epoch=epoch, total_epoch=config.epochs,
                                 model_name=config.model_name,
                                 path=config.logs + config.model_name + os.sep + str(fold) + os.sep,
                                 total=len(test_loader))
    # 2.2 切换到评估模式，确认型号已转移到cuda
    model.cuda()
    model.eval()
    # 2.3 测试数据
    label_list = []
    Y_pred = []
    target_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            test_progressor.current = i
            target_list.append(target[0])
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            # 2.3.1 计算输出
            output = model(input)
            loss = criterion(output, target)
            y_pred = output.cpu().data.tolist()
            label_list.append(np.argmax(y_pred[0]))  # 取出y中元素最大值所对应的索引
            Y_pred.append(y_pred[0])
            # 2.3.2 测量accuracy和记录loss
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            test_progressor.current_loss = losses.avg
            test_progressor.current_top1 = top1.avg
            test_progressor()
        test_progressor.done()

    return [losses.avg, top1.avg.cpu().data.item()], target_list, label_list, Y_pred

# 3. 更多详细操作，以建立main函数
def main():
    fold = str(config.fold)
    # 3.1 创建必要的文件夹
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(config.submit + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.submit + config.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(config.logs + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.logs + config.model_name + os.sep + str(fold) + os.sep)

    # 3.2 获取模型和优化器，并初始化损失函数
    # model = resnet18(num_classes=len(config.class_list))
    # model = seresnet18()
    model = resnet18()
    model.cuda()
    # 初始化正则化
    # if config.weight_decay > 0:
    #     reg_loss = Regularization(model, config.weight_decay, p=1).cuda()  # L1/L2正则
    # else:
    #     print("no regularization")
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)
    
    # 3.4 重新启动训练过程
    criterion = nn.CrossEntropyLoss().cuda()
    start_epoch = 0
    best_precision1 = 0
    resume = False
    if resume:
        checkpoint = torch.load(config.best_models + config.model_name + os.sep + str(fold) + "/model_best.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 3.5 获取文件和分割数据集
    train_data_list, val_data_list = random_split_ratio(config.data_root, config.class_list, split_rate=0.2)
    # print(len(val_data_list))

    # 3.6 加载数据为DataLoader
    train_dataloader = DataLoader(CreateImgDataset(train_data_list), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(CreateImgDataset(val_data_list, train=False), batch_size=1, shuffle=True, collate_fn=collate_fn, pin_memory=False, num_workers=4)

    # 4.1 初始化学习率调整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # 4.2 定义指标
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf, 0, 0]
    model.train()

    # 5. 训练模块
    start = timer()
    train_list = []
    valid_list = []
    label_list = []
    y_pred = []
    target_list = []
    # 5.1 初始化早停止
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    for epoch in range(start_epoch, config.epochs):
        # 5.2 学习率调整
        if get_learning_rate(optimizer) > 1e-8:
           scheduler.step(epoch)
        # 5.3 全局迭代
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=config.epochs,
                                       model_name=config.model_name,
                                       path=config.logs + config.model_name + os.sep + str(fold) + os.sep,
                                       total=len(train_dataloader))
        for batch, (input, target) in enumerate(train_dataloader):
            train_progressor.current = batch
            # 5.4 数据输入网络训练
            model.train()
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(input)
            # 5.5 计算训练损失
            loss = criterion(output, target)
            # if config.weight_decay > 0:
            #     loss = loss + reg_loss(model)
            # 5.6 计算准确率
            precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0], input.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            # 5.7 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_progressor()

        train_progressor.done()
        train_list.append([train_losses.avg, train_top1.avg.cpu().data.item()])

        # 6 评估每个epoch
        valid, target_list_t, label_list_t, y_pred_t = evaluate(val_dataloader, model, criterion, fold, epoch)
        valid_list.append(valid)
        # 6.1 保存最优模型
        is_best = valid[1] > best_precision1
        best_precision1 = max(valid[1], best_precision1)
        if is_best:
            target_list = target_list_t
            label_list = label_list_t
            y_pred = y_pred_t
        save_checkpoint({
                    "epoch": epoch + 1,
                    "model_name": config.model_name,
                    "state_dict": model.state_dict(),
                    "best_precision1": best_precision1,
                    "optimizer": optimizer.state_dict(),
                    "fold": fold,
                    "valid_loss": valid[0],
        }, is_best, fold)

        # 6.2 根据验证损失判断早停止
        early_stopping(-valid[1], model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
    print("训练用时：" + time_to_str((timer() - start), 'min'))
    # 6.3 保存最优模型评估结果到excel
    Y_pred = np.asarray(y_pred)
    f = xlwt.Workbook()
    train_sheet = f.add_sheet(u'train', cell_overwrite_ok=True)  # 创建sheet
    val_sheet = f.add_sheet(u'verify', cell_overwrite_ok=True)
    result_sheet = f.add_sheet(u'result', cell_overwrite_ok=True)
    for i, t in enumerate(train_list):
        train_sheet.write(i, 0, t[0])
        train_sheet.write(i, 1, t[1])
    for i, v in enumerate(valid_list):
        val_sheet.write(i, 0, v[0])
        val_sheet.write(i, 1, v[1])
    for i, r in enumerate(target_list):
        result_sheet.write(i, 0, r)
        result_sheet.write(i, 1, int(label_list[i]))
        for c in range(0, len(config.class_list)):
            result_sheet.write(i, 2+c, Y_pred[i, c])
    f.save(config.logs + config.model_name + os.sep + str(fold) + os.sep + config.model_name + '.xlsx')
    # 6.4 画训练过程曲线与结果图
    plot_training(train_list, valid_list, fold, config.dpi)
    plot_result(config.class_list, fold, best_precision1, target_list, label_list, Y_pred, config.dpi)
if __name__ =="__main__":
    main()



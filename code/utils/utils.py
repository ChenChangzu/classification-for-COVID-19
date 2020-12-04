import torch
import torch.nn as nn
import sys
import os
from config import config
import numpy as np
from PIL import Image
import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
import xlrd
import shutil
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MultipleLocator
from scipy import interp
from itertools import cycle
from dataset.dataloader import *

# 保存训练模型
def save_checkpoint(state, is_best, fold):
    filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = config.best_models + config.model_name + os.sep + str(fold) + os.sep + 'model_best.pth.tar'
        print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message))
        with open("%s%s.txt" % (config.logs + config.model_name + os.sep + str(fold) + os.sep, config.model_name), "a") as f:
            print("Get Better top1 : %s saving weights to %s" % (state["best_precision1"], message), file=f)
        shutil.copyfile(filename, message)
# 计算平均值的类
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# 自定义学习率衰减方法
def adjust_learning_rate(optimizer, epoch):
    """将学习速率设置为每3个epoch衰减10倍"""
    lr = config.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# 自定义学习率变化方法
def schedule(current_epoch, current_lrs, **logs):
        lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
        epochs = [0, 1, 6, 8, 12]
        for lr, epoch in zip(lrs, epochs):
            if current_epoch >= epoch:
                current_lrs[5] = lr
                if current_epoch >= 2:
                    current_lrs[4] = lr * 1
                    current_lrs[3] = lr * 1
                    current_lrs[2] = lr * 1
                    current_lrs[1] = lr * 1
                    current_lrs[0] = lr * 0.1
        return current_lrs
# 计算准确率
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# 获取当前学习率
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
       lr += [param_group['lr']]
    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]
    return lr
# 输出格式化时间
def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t)/60
        hr = t//60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t//60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError

# 画训练过程图
def plot_training(train_list, valid_list, fold, dpi):
    train_loss = [example[0] for example in train_list]
    train_acc = [example[1] for example in train_list]
    valid_loss = [example[0] for example in valid_list]
    valid_acc = [example[1] for example in valid_list]
    plt.style.use("ggplot")  # matplotlib的美化样式

    plt.figure(1)
    plt.plot(np.arange(0, len(train_loss)), train_loss, label="train_loss")
    plt.plot(np.arange(0, len(valid_loss)), valid_loss, label="test_loss")
    plt.title("Loss of " + config.model_name)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.savefig(config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_loss.svg", dpi=dpi)
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(0, len(train_acc)), train_acc, label="train_acc")
    plt.plot(np.arange(0, len(valid_acc)), valid_acc, label="test_acc")
    plt.title("Accuracy of " + config.model_name)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend(loc="best")
    plt.savefig(config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_all_acc.svg", dpi=dpi)
    plt.show()

    plt.figure(3)
    plt.plot(np.arange(0, len(valid_acc)), valid_acc, label="val_acc")
    plt.title("Accuracy of " + config.model_name)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend(loc="best")
    plt.savefig(config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_val_acc.svg", dpi=dpi)
    plt.show()
# 画单个模型二分类和多分类的ROC曲线
def plot_result(class_list, fold, top1, target_list, label_list, Y_pred, dpi):
    nb_classes = len(class_list)
    plt.figure(figsize=(8, 6))
    plt.style.use("ggplot")  # matplotlib的美化样式

    # Binarize the output
    output_list = label_binarize(label_list, classes=[i for i in range(nb_classes)])
    Y_valid = label_binarize(target_list, classes=[i for i in range(nb_classes)])
    print("best Accuracy:", top1)
    # Top1 = Accuracy
    # plot ROC
    if nb_classes == 2:
        precision = metrics.precision_score(Y_valid, output_list, average='binary')
        recall = metrics.recall_score(Y_valid, output_list, average='binary')
        f1_score = metrics.f1_score(Y_valid, output_list, average='binary')
        accuracy_score = metrics.accuracy_score(Y_valid, output_list)
        print("Precision_score:", precision)
        print("Recall_score:", recall)
        print("F1_score:", f1_score)
        print("Accuracy_score:", accuracy_score)

        result = "Top1="+str(top1)+"\nPrecision_score=" + str(precision) + "\nRecall_score=" + str(recall) + \
                 "\nF1_score=" + str(f1_score) + "\nAccuracy_score=" + str(accuracy_score)
        with open("%s%s.txt" % (config.logs + config.model_name + os.sep + str(fold) + os.sep, config.model_name),
                  "a") as f:
            print(result, file=f)
        fpr, tpr, threshold = metrics.roc_curve(Y_valid, Y_pred[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.3f)" % roc_auc)
    else:
        correct_num = 0
        for i, l in enumerate(label_list):
            if l == target_list[i]:
                correct_num += 1
        accuracy_score = correct_num/len(label_list)
        print("Accuracy_score:", accuracy_score)
        result = "Top1="+str(top1)+"\nAccuracy_score=" + str(accuracy_score)
        with open("%s%s.txt" % (config.logs + config.model_name + os.sep + str(fold) + os.sep, config.model_name), "a") as f:
            print(result, file=f)

        # roc_curve：真阳性率（True Positive Rate , TPR）或灵敏度（sensitivity）
        # 横坐标：假阳性率（False Positive Rate , FPR）
        # 计算每个类别的ROC曲线和ROC面积
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(nb_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(Y_valid[:, i], Y_pred[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # 计算micro-average的ROC曲线和ROC面积
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_valid.ravel(), Y_pred.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # 计算macro-average的ROC曲线和ROC面积
        # 首先，汇总所有的假阳性率（false positive rates）
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

        # 然后，插值所有的ROC曲线在这一点
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(nb_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # 最后，求平均值并计算AUC
        mean_tpr /= nb_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # 绘制所有ROC曲线
        # plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC Curve (AUC = {0:0.3f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC Curve (AUC = {0:0.3f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(nb_classes), colors):
            plt.plot(fpr[i], tpr[i],  # color=color, lw=2,
                     label='ROC Curve of {0} (AUC = {1:0.3f})'.format(class_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC')
    plt.legend(loc="lower right")
    plt.savefig(config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_ROC.svg", dpi=dpi)
    plt.show()
    # plot Confusion Matrix
    matrix = confusion_matrix(target_list, label_list)
    plot_CM(class_list, matrix, config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_CM.svg", dpi)
# 画混淆矩阵
def plot_CM(classes, matrix, savname, dpi):
    # Normalize by row
    cm = matrix
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str(cm[i, i])+' ('+str('%.2f' % (matrix[i, i] * 100))+'%)', va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    # save
    plt.savefig(savname, dpi=dpi)

# 画同一个模型三个二分类的ROC
def plot_multiple_ROC(model_list, fold_list, save_name, dpi):
    plt.figure(figsize=(8, 6))
    plt.style.use("ggplot")  # matplotlib的美化样式

    for i, model_dir in enumerate(model_list):
        classification = model_dir.split('_')[-1].split('vs')
        data = xlrd.open_workbook('./logs/'+model_dir+os.sep+str(fold_list[i])+os.sep+model_dir+'.xlsx')
        # 通过索引顺序或名称获取sheet页
        table = data.sheet_by_name(u'result')  # 通过名称获取
        target_list = table.col_values(0)
        y_pred = []
        for j in range(table.nrows):  # table.nrows获取数据行数
            # 获取整行和整列的值（返回数组）
            row = table.row_values(j)
            y_pred.append(row[2:])
        Y_pred = np.asarray(y_pred)
        Y_valid = label_binarize(target_list, classes=[i for i in range(2)])
        # plot ROC
        fpr, tpr, threshold = metrics.roc_curve(Y_valid, Y_pred[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC Curve of "+classification[0]+" vs. "+classification[1]+" (AUC = %0.3f)" % roc_auc, lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Test ROC')
    plt.legend(loc="lower right")
    plt.savefig(config.submit + save_name +"_multiple_2Class_ROC.svg", dpi=dpi)
    plt.show()

# 画单个模型二分类和多分类的ROC曲线
def plot_ROC(model, val_data_list, class_list, fold, dpi):
    nb_classes = len(class_list)
    plt.figure(figsize=(8, 6))
    plt.style.use("ggplot")  # matplotlib的美化样式

    val_dataloader = DataLoader(CreateNiiDataset(val_data_list, train=False), batch_size=1, shuffle=False, pin_memory=False)

    best_model = torch.load(config.best_models + config.model_name + os.sep + str(fold) + os.sep + "model_best.pth.tar")
    model.load_state_dict(best_model["state_dict"])

    label_list = []
    score_list = []
    Y_pred = []
    target_list = []
    top1 = AverageMeter()

    # 切换到评估模式，确认型号已转移到cuda
    model.cuda()
    model.eval()
    for j, (input, target) in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            image_var = Variable(input).cuda()
            output = model(image_var)
            y_pred = output.cpu().data.tolist()
        label_list.append(np.argmax(y_pred[0]))  # 取出y中元素最大值所对应的索引
        score_list.append(max(y_pred[0]))
        Y_pred.append(y_pred[0])
        target_list.append(int(target))
        precision1, precision2 = accuracy(output, Variable(torch.from_numpy(np.array(target)).long()).cuda(), topk=(1, 2))
        top1.update(precision1[0], input.size(0))
    print("Top1="+str(top1.avg.cpu().data.item()))
    # Binarize the output
    output_list = label_binarize(label_list, classes=[i for i in range(nb_classes)])
    Y_valid = label_binarize(target_list, classes=[i for i in range(nb_classes)])
    Y_pred = np.asarray(Y_pred)
    # plot ROC
    if nb_classes == 2:
        precision = metrics.precision_score(Y_valid, output_list, average='binary')
        recall = metrics.recall_score(Y_valid, output_list, average='binary')
        f1_score = metrics.f1_score(Y_valid, output_list, average='binary')
        accuracy_score = metrics.accuracy_score(Y_valid, output_list)
        print("Precision_score:", precision)
        print("Recall_score:", recall)
        print("F1_score:", f1_score)
        print("Accuracy_score:", accuracy_score)

        result = "Top1="+str(top1.avg.cpu().data.item())+"\nPrecision_score=" + str(precision) + "\nRecall_score=" + str(recall) + \
                 "\nF1_score=" + str(f1_score) + "\nAccuracy_score=" + str(accuracy_score)
        with open("%s%s.txt" % (config.logs + config.model_name + os.sep + str(fold) + os.sep, config.model_name),
                  "a") as f:
            print(result, file=f)
        fpr, tpr, threshold = metrics.roc_curve(target_list, score_list)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.3f)" % roc_auc)
    else:
        # micro：多分类　　
        # weighted：不均衡数量的类来说，计算二分类metrics的平均
        # macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
        # precision = metrics.precision_score(Y_valid, output_list, average='micro')
        # recall = metrics.recall_score(Y_valid, output_list, average='micro')
        # f1_score = metrics.f1_score(Y_valid, output_list, average='micro')
        correct_num = 0
        for i, l in enumerate(label_list):
            if l == target_list[i]:
                correct_num += 1
        accuracy_score = correct_num/len(label_list)
        # accuracy_score = metrics.accuracy_score(Y_valid, output_list)
        # print("Precision_score:", precision)
        # print("Recall_score:", recall)
        # print("F1_score:", f1_score)
        print("Accuracy_score:", accuracy_score)

        # result = "\nPrecision_score="+str(precision) + "\nRecall_score="+str(recall) + \
        #          "\nF1_score="+str(f1_score) + "\nAccuracy_score="+str(accuracy_score)
        result = "Top1="+str(top1.avg.cpu().data.item())+"\nAccuracy_score=" + str(accuracy_score)
        with open("%s%s.txt" % (config.logs + config.model_name + os.sep + str(fold) + os.sep, config.model_name), "a") as f:
            print(result, file=f)

        # roc_curve：真阳性率（True Positive Rate , TPR）或灵敏度（sensitivity）
        # 横坐标：假阳性率（False Positive Rate , FPR）

        # 计算每个类别的ROC曲线和ROC面积
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(nb_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(Y_valid[:, i], Y_pred[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # 计算micro-average的ROC曲线和ROC面积
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_valid.ravel(), Y_pred.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # 计算macro-average的ROC曲线和ROC面积

        # 首先，汇总所有的假阳性率（false positive rates）
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

        # 然后，插值所有的ROC曲线在这一点
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(nb_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # 最后，求平均值并计算AUC
        mean_tpr /= nb_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # 绘制所有ROC曲线
        # plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC Curve (AUC = {0:0.3f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC Curve (AUC = {0:0.3f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(nb_classes), colors):
            plt.plot(fpr[i], tpr[i],  # color=color, lw=2,
                     label='ROC Curve of {0} (AUC = {1:0.3f})'.format(class_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation ROC')
    plt.legend(loc="lower right")
    plt.savefig(config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_ROC.svg", dpi=dpi)
    plt.show()
    # plot Confusion Matrix
    matrix = confusion_matrix(target_list, label_list)
    plot_CM(class_list, matrix, config.submit + config.model_name + os.sep + str(fold) + os.sep + config.model_name + "_CM.svg", dpi)

# 画热力图可视化
def draw_CAM(model, img_path, save_path):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载
    img_contents = nib.load(img_path)
    img_fdata = np.asarray(img_contents.get_data())
    img = img_fdata
    img_fdata = img_fdata[np.newaxis, :]
    img_fdata = img_fdata[np.newaxis, :]
    img_tensor = torch.from_numpy(img_fdata)
    img_tensor = img_tensor.type(torch.FloatTensor)
    input = Variable(img_tensor).cuda()

    # 获取模型输出的feature/score
    model.cuda()
    model.eval()
    features = get_features(model, input)
    # print(features.shape)
    output = features.view(features.size()[0], -1)
    output = model.classifier(output)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度
    print(grads.shape)

    pooled_grads = torch.nn.functional.adaptive_avg_pool3d(grads, 1)
    print(pooled_grads.shape)
    print(features.shape)

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    # plt.matshow(heatmap)
    # plt.show()

    heatmap = np.resize(heatmap, (121, 145, 121))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = heatmap[:, :, 65]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    # plt.matshow(heatmap)
    # plt.show()
    img = np.uint8(255 * img[:, :, 65])
    # plt.matshow(img)
    # plt.show()
    img = np.repeat(img[..., np.newaxis], 3, 2)
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    plt.matshow(superimposed_img)
    plt.show()
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
def get_features(model, x):

    x1 = model.stage1(x)
    x1 = model.maxpool(x1)
    xs1 = model.conv1x1_1(x1)

    x2 = model.stage2(x1)
    x_1_2 = xs1 + x2
    x_1_2 = model.maxpool(x_1_2)
    xs2 = model.conv1x1_2(x_1_2)

    x3 = model.stage3(x_1_2)
    x_2_3 = xs2 + x3
    x_2_3 = model.maxpool(x_2_3)
    xs3 = model.conv1x1_3(x_2_3)

    x4 = model.stage4(x_2_3)
    x_3_4 = xs3 + x4
    x_3_4 = model.maxpool(x_3_4)
    xs4 = model.conv1x1_4(x_3_4)

    x5 = model.stage5(x_3_4)
    x_4_5 = xs4 + x5
    x_4_5 = model.maxpool(x_4_5)

    return x_4_5

# 读取txt文件为矩阵
def file2array(path, delimiter):
    fp = open(path, 'r', encoding='utf-8')
    content = fp.read()  # content现在是一行字符串，该字符串包含文件所有内容
    fp.close()
    rowlist = content.splitlines()  # 按行转换为一维表，splitlines默认参数是‘\n’
    # 逐行遍历
    # 结果按分隔符分割为行向量
    recordList = [row.strip().split(delimiter) for row in rowlist]
    w = len(recordList[0])
    h = len(recordList)
    result = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            result[i, j] = float(recordList[i][j].strip())
    return result

# 自定义正则化Regularization类
class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=2为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
    #     print("---------------regularization weight---------------")
    #     for name, w in weight_list:
    #         print(name)
    #     print("---------------------------------------------------")
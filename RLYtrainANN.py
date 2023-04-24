from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import helper_code as helper
from helper_code import *
import numpy as np, os, sys
import random

from torch.optim.lr_scheduler import StepLR


def load_mat2list_data(data_folder, patient_id, name, flag):
    # Define file location.
    patient_EEC_file_path = os.path.join(data_folder, patient_id, name + '.mat')
    patient_EEC_file = sp.io.loadmat(patient_EEC_file_path)
    if flag == 1:
        mat2list = patient_EEC_file['val'].tolist()
    else:
        mat2list = patient_EEC_file['val'].astype(numpy.float32)
    # 同时load cpc
    print(patient_EEC_file_path)
    return mat2list  # ,cpc


def GoodPoor(prediction):
    if prediction <= 1.0:
        return 0
    else:
        return 1


# 读取你想训练哪个小时的数据,同时用trainrate来控制训练集的比例 #最牛逼的要用
def build_traina_test_sets_labels(trainrate, testrate, data_folder, patient_ids):
    # 定义一个空的narray变量，用来存放本次训练的所有数据,和测试集
    traindata = np.array([])
    testdata = np.array([])

    # 定义一个空的narray变量，用来存放本次训练的标签,和测试集的标签
    trainlabel = []
    testlabel = []

    # Load the metadata.
    train_num_patients = int(len(patient_ids) * trainrate)
    test_num_patients = int(len(patient_ids) * testrate)
    # hour = "_" + str(hour)
    # 构建训练集和标签
    for i in range(train_num_patients):
        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # reco_meta读tsv字符串
        # 利用tsv访问EEC,利用函数把record列的名字留下，为后面制作路径铺垫,同时把nan去除，加快速度
        EEC_namelist = get_column(recording_metadata, 'Record', str)
        EEC_namelist = np.delete(EEC_namelist, np.where(EEC_namelist == ['nan']), axis=0)
        flag = 0
        # print(EEC_namelist)

        for name in EEC_namelist:  # name是一个字符串
            random_num = np.random.normal(0, 1, 1)  # 定义一个随机数，该随机数满足正态分布
            if random_num > 0.6:  # 如果这个字符串不是nan，就把这个字符串作为文件名，读取数据
                mat2list = load_mat2list_data(data_folder, patient_id, name, 0)  # 1就是list，其他就是narray
                flag = 1
                # 已经在help中把mat2list变成float32类型了 把这一次的mat2list加到traindata中
                if traindata.size == 0:
                    traindata = mat2list
                else:
                    traindata = np.concatenate((traindata, mat2list), axis=0)

                # 读取cpc值
                cpc = get_variable(patient_metadata, 'CPC', float) - 1
                trainlabel.append(cpc)

    t = i  # 记录一下训练集的最后一个病人的序号
    print("训练集准备完毕共", train_num_patients, "个病人", traindata.shape, "个数据")
    print("训练标签准备完毕共", train_num_patients, "个病人", len(trainlabel), "个数据")

    # 构建测试集
    for i in range(t + 1, t + 1 + test_num_patients):
        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # reco_meta读tsv字符串
        # 利用tsv访问EEC,利用函数把record列的名字留下，为后面制作路径铺垫,同时把nan去除，加快速度
        EEC_namelist = get_column(recording_metadata, 'Record', str)
        EEC_namelist = np.delete(EEC_namelist, np.where(EEC_namelist == ['nan']), axis=0)

        # print(EEC_namelist)
        for name in EEC_namelist:  # name是一个字符串
            random_num = np.random.normal(0, 1, 1)  # 定义一个随机数，该随机数满足正态分布
            if random_num > 0.6:
                # if hour in name:  # 如果这个字符串不是nan，就把这个字符串作为文件名，读取数据
                mat2list = load_mat2list_data(data_folder, patient_id, name, 0)  # 1就是list，其他就是narray

                # 已经在help中把mat2list变成float32类型了 把这一次的mat2list加到traindata中
                if testdata.size == 0:
                    testdata = mat2list
                else:
                    testdata = np.concatenate((testdata, mat2list), axis=0)

                cpc = get_variable(patient_metadata, 'CPC', float) - 1
                testlabel.append(cpc)

    print("测试标签准备完毕共", test_num_patients, "个病人", len(testlabel), "个数据")
    print("测试集准备完毕共", test_num_patients, "个病人", testdata.shape, "个数据")
    return traindata, testdata, np.array(trainlabel, int), np.array(testlabel, int)


def use_dataset(data_folder, patient_ids):
    testdata = np.array([])
    testlabel = []

    # Load the metadata.

    i = random.randint(0, 600)
    # Load data.
    patient_id = patient_ids[i]
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # reco_meta读tsv字符串
    # 利用tsv访问EEC,利用函数把record列的名字留下，为后面制作路径铺垫,同时把nan去除，加快速度
    EEC_namelist = get_column(recording_metadata, 'Record', str)
    EEC_namelist = np.delete(EEC_namelist, np.where(EEC_namelist == ['nan']), axis=0)
    # 随机在EEC_namelist中选取一个字符串出来，作为name
    name = random.choice(EEC_namelist)
    # if hour in name:  # 如果这个字符串不是nan，就把这个字符串作为文件名，读取数据
    mat2list = load_mat2list_data(data_folder, patient_id, name, 0)  # 1就是list，其他就是narray
    # 已经在help中把mat2list变成float32类型了 把这一次的mat2list加到traindata中
    if testdata.size == 0:
        testdata = mat2list
    else:
        testdata = np.concatenate((testdata, mat2list), axis=0)

    # 读取cpc值
    cpc = get_variable(patient_metadata, 'CPC', float) - 1
    testlabel.append(cpc)

    return testdata, np.array(testlabel, int), patient_id


# 构建神经网络模型输入为病人的第24小时的数据18*30000，输出为outcome值1*1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(18 * 30000, 100)
        self.fc2 = nn.Linear(100, 5)  ########RLY20.04 4.23改5输出

    def forward(self, x):
        x = x.view(-1, 18 * 30000)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# 训练神经网络模型，输入为病人的第24小时的数据18*30000，输出为outcome值1*1
def train_model(model, train_data, train_labels, test_data, test_labels, num_epochs, learning_rate):
    # Convert the data to torch tensors.
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    # Define the loss function.
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数20.10.4.23改
    # Define the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # 每10个epoch学习率衰减为原来的0.1倍
    losslist = []

    # Train the model.
    for epoch in range(num_epochs):
        y_pred = model(train_data)

        # Compute and print loss.
        loss = loss_fn(y_pred, train_labels)
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        losslist.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # 每个epoch都调整学习率

    # print(type(y_pred), type(test_labels))
    # print(y_pred, test_labels)

    # Test the model.
    with torch.no_grad():
        y_pred = model(test_data)
        loss = loss_fn(y_pred, test_labels)
        # 输出y_pred和test_labels之间的准确率
        print('Test loss: {}'.format(loss.item()))


# 主函数
if __name__ == '__main__':
    data_folder = 'D:/brain/i-care-international-cardiac-arrest-research-consortium-database-1.0/training'
    output_folder = 'C:/Users/13102/Desktop/brain_wave/1.txt'
    patient_ids = find_data_folders(data_folder)

    traindata, testdata, trainlabel, testlabel = build_traina_test_sets_labels(0.1, 0.025, data_folder, patient_ids)
    print(torch.from_numpy(trainlabel).to(torch.int64))

    trainlabel = nn.functional.one_hot(torch.from_numpy(trainlabel).to(torch.int64), 5).float()
    testlabel = nn.functional.one_hot(torch.from_numpy(testlabel).to(torch.int64), 5).float()

    # trainlabel = np.array(trainlabel)
    # testlabel = np.array(testlabel)
    model = Net()
    train_model(model, traindata, trainlabel, testdata, testlabel, num_epochs=200, learning_rate=0.01)
    torch.save(model.state_dict(), 'model.sav')  # 保存模型
    print('model saved!!!')

    usemodel = model.load_state_dict(torch.load('model.sav'))  # 加载模型
    print('model loaded!!!')

    data, label, patient_id = use_dataset(data_folder, patient_ids)
    data = torch.from_numpy(data)
    label = nn.functional.one_hot(torch.from_numpy(label).to(torch.int64), 5).float()
    print(data)
    print(label)
    with torch.no_grad():
        output = model(data)
    print(output)
    probs = torch.softmax(output, dim=1)
    print(probs)
    max_probs, max_indices = torch.max(probs, dim=1)
    probbility = float(max(max_probs.tolist()))  # 输出概率
    print(probbility)
    predictions = float(probs.argmax(dim=1) + 1)  # 输出预测值
    print(predictions)
    outcome = GoodPoor(predictions)  # GOOD POOR
    print(outcome)
    S = helper.save_challenge_outputs(output_folder, patient_id, outcome, probbility, predictions)

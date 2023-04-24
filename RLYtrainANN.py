import numpy
from sklearn.metrics import accuracy_score

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import scipy
import torch

def load_mat2list_data(data_folder, patient_id, name, flag):
    # Define file location.
    patient_EEC_file_path = os.path.join(data_folder, patient_id, name + '.mat')
    patient_EEC_file = sp.io.loadmat(patient_EEC_file_path)
    if flag == 1:
        mat2list = patient_EEC_file['val'].tolist()
    else:
        mat2list = patient_EEC_file['val'].astype(numpy.float32)
    #同时load cpc
    print(patient_EEC_file_path)
    return mat2list#,cpc

def GoodPoor(prediction):
    if prediction <=1.0 :
        return 0
    else:
        return 1


# 读取你想训练哪个小时的数据,同时用trainrate来控制训练集的比例 #最牛逼的要用
def build_traina_test_sets_labels(hour, trainrate, testrate,data_folder,patient_ids):
    # 定义一个空的narray变量，用来存放本次训练的所有数据,和测试集
    traindata = np.array([])
    testdata = np.array([])

    # 定义一个空的narray变量，用来存放本次训练的标签,和测试集的标签
    trainlabel = []
    testlabel = []

    # Load the metadata.
    train_num_patients = int(len(patient_ids) * trainrate)
    test_num_patients = int(len(patient_ids) * testrate)
    hour = "_" + str(hour)
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
            if hour in name:  # 如果这个字符串不是nan，就把这个字符串作为文件名，读取数据
                mat2list = load_mat2list_data(data_folder, patient_id, name, 0)  # 1就是list，其他就是narray
                flag = 1
                # 已经在help中把mat2list变成float32类型了 把这一次的mat2list加到traindata中
                if traindata.size == 0:
                    traindata = mat2list
                else:
                    traindata = np.concatenate((traindata, mat2list), axis=0)

                # 读取cpc值
                cpc = get_variable(patient_metadata, 'CPC', float)-1
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
            if hour in name:  # 如果这个字符串不是nan，就把这个字符串作为文件名，读取数据
                mat2list = load_mat2list_data(data_folder, patient_id, name, 0)  # 1就是list，其他就是narray

                # 已经在help中把mat2list变成float32类型了 把这一次的mat2list加到traindata中
                if testdata.size == 0:
                    testdata = mat2list
                else:
                    testdata = np.concatenate((testdata, mat2list), axis=0)

                cpc = get_variable(patient_metadata, 'CPC', float)-1
                testlabel.append(cpc)

    print("测试标签准备完毕共", test_num_patients, "个病人", len(testlabel), "个数据")
    print("测试集准备完毕共", test_num_patients, "个病人", testdata.shape, "个数据")
    return traindata, testdata, np.array(trainlabel, int), np.array(testlabel, int)



# 构建数据集，选n个病人（如果没有第24小时数据，则选下一个病人），将数据集分为训练集和测试集
# def build_dataset(patient_ids, folder, n):
#     # Load the data.
#     data_24 = get_challenge_data(patient_ids, folder)   #data_24是一个list，每个元素是一个病人的第24小时的数据
#     # Load the labels.
#     cpcs = get_challenge_labels(patient_ids, folder)        #cpcs是一个list，每个元素是一个病人的cpc值
#     outcomes = get_challenge_outcomes(patient_ids, folder)  #outcomes是一个list，每个元素是一个病人的outcome值
#     # Select n patients.
#     data_24 = data_24[:n]   #data_24是一个list，每个元素是一个病人的第24小时的数据
#     cpcs = cpcs[:n]
#     outcomes = outcomes[:n]     #outcomes是一个list，每个元素是一个病人的outcome值
#     # Split the data into training and test sets.
#     train_data_24 = data_24[:int(n*0.8)]
#     test_data_24 = data_24[int(n*0.8):]
#     train_cpcs = cpcs[:int(n*0.8)]
#     test_cpcs = cpcs[int(n*0.8):]
#     train_outcomes = outcomes[:int(n*0.8)]
#     test_outcomes = outcomes[int(n*0.8):]
#     return train_data_24, test_data_24, train_cpcs, test_cpcs, train_outcomes, test_outcomes


# 构建神经网络模型输入为病人的第24小时的数据18*30000，输出为cpc值1*1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import helper_code as helper


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


# class MyDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.labels[index]
#         return x, y


# 训练神经网络模型，输入为病人的第24小时的数据18*30000，输出为outcome值1*1
def train_model(model, train_data, train_labels, test_data, test_labels, num_epochs, learning_rate):
    # Convert the data to torch tensors.
    train_data = torch.from_numpy(train_data)
    # train_labels = torch.from_numpy(train_labels).to(torch.float32)
    test_data = torch.from_numpy(test_data)
    # test_labels = torch.from_numpy(test_labels)
    # 将训练数据集封装为Dataset对象
    # train_dataset = MyDataset(train_data, train_labels)
    # 使用DataLoader类加载数据集，并设置batch size大小和shuffle参数
    # trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Define the loss function.
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数20.10.4.23改
    # Define the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losslist = []
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # Train the model.
    for epoch in range(num_epochs):
        # for inputs, labels in train_loader:
        # Forward pass: Compute predicted y by passing x to the model.

        y_pred = model(train_data)

        # Compute and print loss.
        loss = loss_fn(y_pred, train_labels)
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        losslist.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(type(y_pred), type(test_labels))
    # print(y_pred, test_labels)

    # Test the model.
    with torch.no_grad():
        y_pred = model(test_data)
        loss = loss_fn(y_pred, test_labels)
        # 输出y_pred和test_labels之间的准确率
        print('Test loss: {}'.format(loss.item()))


    # #预测ICARE_0334_24的outcome值
    # with torch.no_grad():
    #     y_pred = model(test_data[0:18][:])
    #     print(y_pred)
    #     print(test_labels[0])
    #     print(torch.max(y_pred, 1)[1])



    # Plot the loss.
    # plt.plot(losslist)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()


# 主函数
if __name__ == '__main__':

    data_folder = 'D:/brain/i-care-international-cardiac-arrest-research-consortium-database-1.0/training'
    output_folder='C:/Users/13102/Desktop/brain_wave/1.txt'
    patient_ids = find_data_folders(data_folder)

    traindata, testdata, trainlabel, testlabel = build_traina_test_sets_labels(24, 0.008, 0.002, data_folder,patient_ids)
    print(torch.from_numpy(trainlabel).to(torch.int64))

    trainlabel = nn.functional.one_hot(torch.from_numpy(trainlabel).to(torch.int64), 5).float()
    testlabel = nn.functional.one_hot(torch.from_numpy(testlabel).to(torch.int64), 5).float()

    # trainlabel = np.array(trainlabel)
    # testlabel = np.array(testlabel)
    model = Net()
    train_model(model, traindata, trainlabel, testdata, testlabel, num_epochs=50, learning_rate=0.001)
    torch.save(model.state_dict(), 'model.sav') # 保存模型
    print('model saved!!!')

    usemodel=model.load_state_dict(torch.load('model.sav'))     #加载模型
    print('model loaded!!!')

    k=1  #预测第k+1个病人的outcome值
    data=torch.from_numpy(testdata[0*18*k:18+18*k][:])
    with torch.no_grad():
        output = model(data)
    print(output)
    probs = torch.softmax(output, dim=1)
    print(probs)
    max_probs, max_indices = torch.max(probs, dim=1)
    probbility =float( max(max_probs.tolist())  )  #输出概率
    print(probbility)
    predictions = float(probs.argmax(dim=1)+1  )  #输出预测值
    print(predictions)
    outcome=GoodPoor(predictions)   #GOOD POOR
    print(outcome)
    S=helper.save_challenge_outputs(output_folder,patient_ids[k],outcome,probbility, predictions)



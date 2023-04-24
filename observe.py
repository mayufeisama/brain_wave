import numpy

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import scipy
import torch

data_folder = 'D:/brain/i-care-international-cardiac-arrest-research-consortium-database-1.0/training'
patient_ids = find_data_folders(data_folder)
num_patients = len(patient_ids)
print(patient_ids)


# mat2list为病人的脑电图数据，其中包含多个病人的文件，每个文件包含多个18*30000的矩阵，每个矩阵代表一个EEC的数据
# 定义卷积神经网络处理EEC数据
# 输入x为18*30000的矩阵，
class cnn_eec(torch.nn.Module):
    def __init__(self):
        super(cnn_eec, self).__init__()
        # 先做第一次卷积，25个卷积核，每个卷积核大小为1*12，步长为3，输出为25*18*9997
        self.conv1 = torch.nn.Conv2d(1, 25, kernel_size=(1, 12), stride=3)
        # 第一次卷积后的输出经过空间滤波器卷积，25个卷积核，每个卷积核大小为25*18，步长为1，输出为25*9997
        self.conv2 = torch.nn.Conv2d(25, 25, kernel_size=(25, 18), stride=1)
        # 得到特征图后，进行最大池化，池化核大小为1*3，步长为1，输出为25*9995
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 第二次卷积，50个卷积核，每个卷积核大小为25*14，步长为3,输出为50*3328
        self.conv3 = torch.nn.Conv2d(25, 50, kernel_size=(25, 14), stride=3)
        # relu激活函数
        self.relu = torch.nn.ReLU()
        # 池化，池化核大小为1*3，步长为1，输出为50*3326
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 100个卷积核，每个卷积核大小为50*14，步长为3，输出为100*1105
        self.conv4 = torch.nn.Conv2d(50, 100, kernel_size=(50, 14), stride=3)
        # relu激活函数
        self.relu = torch.nn.ReLU()
        # 池化，池化核大小为1*3，步长为1，输出为100*1103
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 卷积，100个卷积核，每个卷积核大小为100*14，步长为3，输出为100*367
        self.conv5 = torch.nn.Conv2d(100, 100, kernel_size=(100, 14), stride=3)
        # relu激活函数
        self.relu = torch.nn.ReLU()
        # 池化，池化核大小为1*3，步长为1，输出为100*365
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 卷积，100个卷积核，每个卷积核大小为100*14，步长为3，输出为100*118
        self.conv6 = torch.nn.Conv2d(100, 100, kernel_size=(100, 14), stride=3)
        # relu激活函数
        self.relu = torch.nn.ReLU()
        # 池化，池化核大小为1*3，步长为1，输出为100*116
        self.pool5 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 卷积，100个卷积核，每个卷积核大小为100*14，步长为3，输出为100*35
        self.conv7 = torch.nn.Conv2d(100, 100, kernel_size=(100, 14), stride=3)
        # relu激活函数
        self.relu = torch.nn.ReLU()
        # 池化，池化核大小为1*3，步长为1，输出为100*33
        self.pool6 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 卷积，100个卷积核，每个卷积核大小为100*12，步长为3，输出为100*10
        self.conv8 = torch.nn.Conv2d(100, 100, kernel_size=(100, 12), stride=3)
        # relu激活函数
        self.relu = torch.nn.ReLU()
        # 池化，池化核大小为1*3，步长为1，输出为100*8
        self.pool7 = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=1)
        # 全连接层，输入为100*8，输出为1
        self.fc1 = torch.nn.Linear(100 * 8, 1)
        # sigmoid激活函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        # 第一次卷积
        X = self.conv1(X)
        # 第一次卷积后的输出经过空间滤波器卷积
        X = self.conv2(X)
        # 得到特征图后，进行最大池化
        X = self.pool1(X)
        # 第二次卷积
        X = self.conv3(X)
        # relu激活函数
        X = self.relu(X)
        # 池化
        X = self.pool2(X)
        # 第三次卷积
        X = self.conv4(X)
        # relu激活函数
        X = self.relu(X)
        # 池化
        X = self.pool3(X)
        # 第四次卷积
        X = self.conv5(X)
        # relu激活函数
        X = self.relu(X)
        # 池化
        X = self.pool4(X)
        # 第五次卷积
        X = self.conv6(X)
        # relu激活函数
        X = self.relu(X)
        # 池化
        X = self.pool5(X)
        # 第六次卷积
        X = self.conv7(X)
        # relu激活函数
        X = self.relu(X)
        # 池化
        X = self.pool6(X)
        # 第七次卷积
        X = self.conv8(X)
        # relu激活函数
        X = self.relu(X)
        # 池化
        X = self.pool7(X)
        # 将X转换为1*800的tensor
        X = X.view(1, 800)
        # 全连接层
        X = self.fc1(X)
        # sigmoid激活函数
        X = self.sigmoid(X)
        # 将X转换为1*1的tensor
        X = X.view(1, 1)
        # 将X转换为numpy
        X = X.numpy()
        # 将X转换为float
        X = X.astype(float)
        # 返回X
        return X


model = cnn_eec()

# 定义损失函数
loss_fn = torch.nn.MSELoss()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


# 梯度下降
# def train():
#     #定义损失函数
#     loss_fn=torch.nn.MSELoss()
#     #定义优化器
#     optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
#     #定义训练次数
#     for t in range(10000):
#         #将模型的参数梯度清零
#         optimizer.zero_grad()
#         #将模型的输出赋值给y_pred
#         y_pred=cnn_eec(eec_data)
#         #计算损失
#         loss=loss_fn(y_pred,y
#         #反向传播
#         loss.backward()
#         #更新参数
#         optimizer.step()
#         #打印损失
#         print(t,loss.item())


def get_patients_i_n_hours(i, n_hours):
    patient_id = patient_ids[i]
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
    # print(type(patient_metadata),type(recording_metadata),type(recording_data))
    # print(str(patient_metadata)+'&&&&'+str(recording_metadata)+'&&&&'+str(recording_data)) reco_meta读tsv字符串
    # 利用tsv访问EEC,利用函数把record列的名字留下，为后面制作路径铺垫
    EEC_namelist = get_column(recording_metadata, 'Record', str)
    # print(EEC_namelist)
    for name in EEC_namelist:
        # 如果EEC文件名中包含'_'+n_hours，取出该文件
        if name.find('_' + n_hours) != -1:
            mat2list = load_mat2list_data(data_folder, patient_id, name, 0)
    return mat2list


def train_n_epochs(n_epochs, n_hours):
    Num_patients = n_epochs
    n_hours = n_hours
    p = 0
    i = 0
    while i < Num_patients:
        # 取出第i个样本的第24小时的EEC数据
        if get_patients_i_n_hours(i + p, n_hours) is not None:
            eec_data = get_patients_i_n_hours(i, n_hours)
            loss_fn = torch.nn.MSELoss()
            y_pred = cnn_eec(eec_data)
            # y是第i+P个病人的cpc值
            y = load_challenge_data(data_folder, patient_ids[i + p])[0]['CPC']
            # cpc1，2为1，cpc3，4，5为0
            if y == 1 or y == 2:
                y = 1
            else:
                y = 0
            y = y.view(1, 1)
            loss = loss_fn(y_pred, y)
            loss.backward()
            i = i + 1
        else:
            p = p + 1
            continue


# 训练1000个样本，每个样本的EEC数据为24小时
# train_n_epochs(1000, '24')

for i in range(num_patients):

    # Load data.
    patient_id = patient_ids[i]
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
    # reco_meta读tsv字符串
    # 利用tsv访问EEC,利用函数把record列的名字留下，为后面制作路径铺垫
    EEC_namelist = get_column(recording_metadata, 'Record', str)
    # print(EEC_namelist)
    for name in EEC_namelist:
        if name != 'nan':
            mat2list = load_mat2list_data(data_folder, patient_id, name, 0)  # 1就是list，其他就是narray
            # 尝试把mat2list转化为float
            print(mat2list)
            print(type(mat2list))
            # mat2list.astype(numpy.float)

            print(mat2list.dtype)

            # 输入一个数据到模型中
            y_pred = model(mat2list)    # 传入的是一个numpy数组 1*1*100*100

            print(y_pred)
            # print(mat2list)
            # print(type(mat2list), len(mat2list), len(mat2list[0]))
            # print(mat2list[0][0])
            # print(type(mat2list[0][0]))
            # print(mat2list[0][0][0])
            # print(type(mat2list[0][0][0]))

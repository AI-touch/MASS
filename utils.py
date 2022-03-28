import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def visualize_accuracy(acc, title, name):
    x = range(len(acc))
    plt.plot(x, acc, label = 'Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    f = os.path.join(name, '{}.png'.format(title))
    plt.savefig(f, dpi=400)
    plt.cla()
    # pylab.show()


def result_save(name, item, title):
    f = os.path.join(name, '{}.npy'.format(title))
    np.save(f, item)
    print('Saved result:', f)

def Data(path='E:/Jupyter/CNN/cutout_new_data_48/', human=['CL'], emotion=['angry', 'pleasure', 'sad', 'calm']):
    label_ges = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    maxvalue = [1024.0, 805.3422196964673, 915.8821055611188, 756.1536631436337, 959.7490196078434, 1024.0,
                801.2727021202508, 995.8901960784314,
                920.6683212697227, 1015.9686274509804, 1024.0, 811.838518250555, 799.121568627451, 843.4264549192859,
                915.4542956197772]
    gesture_labels = np.array(label_ges)
    emotion_labels = np.array(emotion)

    num_gesture = len(gesture_labels)
    num_emotion = len(emotion_labels)

    gesture_id_to_name = dict(zip(range(len(gesture_labels)), gesture_labels))
    gesture_name_to_id = dict(zip(gesture_labels, range(len(gesture_labels))))

    emotion_id_to_name = dict(zip(range(len(emotion_labels)), emotion_labels))
    emotion_name_to_id = dict(zip(emotion_labels, range(len(emotion_labels))))

    label_ges = []
    sequences = []
    label_emo = []
    for i in range(len(human)):
        #     path_human = os.listdir(path + human[i])
        label_ges_person = []
        sequences_person = []
        label_emo_person = []
        for j in range(len(emotion)):
            path_emo = os.listdir(path + human[i] + '/' + emotion[j])
            for m in range(len(path_emo)):
                case_path = path + human[i] + '/' + emotion[j] + '/' + path_emo[m]
                df = pd.read_csv(case_path, header=None)
                values = df.values
                values = values.reshape(48, 10, 10)
                values = np.true_divide(values, maxvalue[i])
                values = np.expand_dims(values, 0)
                sequences_person.append(values)
                label_ges_person.append(gesture_name_to_id[path_emo[m][0]])
                label_emo_person.append(emotion_name_to_id[emotion[j]])
        sequences.append(sequences_person)
        label_ges.append(label_ges_person)
        label_emo.append(label_emo_person)
    return sequences, label_ges, label_emo


def train_test_split_dataloader(data,label,rate = 0.2,BATCH_SIZE = 32):
    ss = StratifiedShuffleSplit(n_splits=1, test_size=rate, train_size=1-rate, random_state=0)  # 分成5组，测试比例为0.25，训练比例是0.75
    for train_index, test_index in ss.split(data, label):
        data_train, test_person_data = np.array(data)[train_index], np.array(data)[test_index]  # 训练集对应的值
        lable_train, test_person_label = np.array(label)[train_index], np.array(label)[test_index]  # 类别集对应的值

    lable_train = lable_train.reshape((len(lable_train),))
    print(lable_train.shape)

    trainX = torch.from_numpy(np.array(data_train))  # 将数组转化为张量，并且二者共享内存，trainX改变，Xtr也会改变
    trainy = torch.from_numpy(np.array(lable_train))
    print(trainX.shape)

    testX = torch.from_numpy(np.array(test_person_data))
    testy = torch.from_numpy(np.array(test_person_label))

    train_dataset = torch.utils.data.TensorDataset(trainX, trainy)
    test_dataset = torch.utils.data.TensorDataset(testX, testy)  # 把数据放在数据库中

    # 打乱训练集数据索引，将索引划分为训练集索引和验证集索引
    num_train = len(train_dataset)
    indices = list(range(num_train))  # 数组，从1到num排列 [1,2,3,4...,6244]
    np.random.shuffle(indices)  # 打乱indices顺序

    # 创建训练集，验证集，测试集的数据loader，每次调用返回一个batch数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # ,
    # sampler=train_sampler)
    print('读取训练数据集合完成')
    # valid_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE,
    #                                            sampler=valid_sampler)
    # print('读取验证数据集合完成')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    print('测试训练数据集合完成', testX.size())
    return train_loader, test_loader

def Dataloader(data_path = '/home/ps/LYK/MyMSDA4/data/',train_human= ['CL'],BATCH_SIZE = 32):
    data_loader = []
    for i in range(len(train_human)):
        data, target, _ = Data(path=data_path,
                             human=[train_human[i]],
                             emotion=['afraid', 'angry', 'astonished', 'calm', 'content', 'depress',
                                      'disgust', 'excited', 'pleasure', 'relaxed', 'sad', 'tired',
                                      'rest'])
        data = np.squeeze(np.array(data), axis=0)
        target = np.transpose(np.array(target),(1,0))
        np.random.seed(10)
        shuffle_ix = np.random.permutation(len(data))
        shuffle_ix = list(shuffle_ix)
        data = data[shuffle_ix, :]
        target = target[shuffle_ix, :]
        target = np.squeeze(target, axis=1)
        trainX = torch.from_numpy(data)  # 将数组转化为张量，并且二者共享内存，trainX改变，Xtr也会改变
        trainy = torch.from_numpy(target)
        dataset = torch.utils.data.TensorDataset(trainX, trainy)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(str(train_human[i])+'数据集合完成', trainX.size(), trainy.size())
        data_loader.append(loader)
    return data_loader



def step_lr(optimizer, learning_rate, epoch, gamma):
    lr = learning_rate
    if (epoch % 10==0) :#&(epoch ==200):
        lr = learning_rate * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
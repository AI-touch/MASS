from __future__ import print_function
import argparse
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import utils
import numpy as np
import models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def MASS_train(model, index, domain_num, optimizer, scheduler, iteration, device,log_interval, save_path):

    target_iter = iter(target_train_loader)
    correct = 0
    source_iter = []
    for idx in range(len(source_loader)):
        source_iter.append(iter(source_loader[idx]))

    acc = []
    time_all = []
    for i in range(1, iteration + 1):
        print('Epoch: {}'.format(index))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        model.train()
        for idx in range(len(source_loader)):
            try:
                source_data, source_label = source_iter[idx].next()
            except Exception as err:
                source_iter[idx] = iter(source_loader[idx])
                source_data, source_label = source_iter[idx].next()
            try:
                target_data, __ = target_iter.next()
            except Exception as err:
                target_iter = iter(target_train_loader)
                target_data, __ = target_iter.next()
            if device:
                source_data, source_label = source_data.type(torch.FloatTensor).cuda(), source_label.cuda()
                target_data = target_data.type(torch.FloatTensor).cuda()
            else:
                source_data, source_label = source_data.type(torch.FloatTensor), source_label
                target_data = target_data.type(torch.FloatTensor)
            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)
            optimizer.zero_grad()
            t0 = time.time()
            cls_1, cls_loss, src_loss, trt_loss, l1_loss = model(source_data, target_data, source_label.long(), mark=idx)
            alpha = 1
            beta = (2 / (1 + math.exp(-10 * (i) / (iteration))) - 1)/2
            lamuda = 1
            loss = cls_1 + alpha*cls_loss + beta * (src_loss + trt_loss) + lamuda*l1_loss
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print(
                    'Source: {:.6f}\tTrain source iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tsrc_Loss: {:.6f}\ttrt_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                        idx, i, 100. * i / iteration, loss.item(), cls_loss.item(), src_loss.item(), trt_loss.item(), l1_loss.item()))
            t1 = time.time() - t0
            time_all.append(t1)

        if i % (log_interval) == 0:
            t_correct = valid(model,domain_num,device)
            if t_correct > correct:
                correct = t_correct
                resultdir = os.path.join(save_path, str(human[index]))
                os.makedirs(resultdir, exist_ok=True)
                model_save(resultdir, model, 'model_best')
            print('Subject-',human,  "%s max correct:" % human[index],
                  correct.item(), "\n")
            acc.append(t_correct)
        scheduler.step()
    resultdir = os.path.join(save_path, str(human[index]))
    os.makedirs(resultdir, exist_ok=True)
    utils.visualize_accuracy(acc, 'acc', resultdir)
    utils.result_save(resultdir, acc, 'test_acc')
    utils.result_save(resultdir, time_all, 'time')

def valid(model,domain_num,device):
    model.eval()
    test_loss = 0
    correct = 0
    correct_per = np.zeros((1, domain_num+1))
    with torch.no_grad():
        for data, target in target_test_loader:
            if device:
                data, target = data.type(torch.FloatTensor).cuda(), target.long().cuda()
            data, target = Variable(data), Variable(target)
            pred_all = model(data, alpha = 0)
            pred_all_onehot = []
            for i in range(len(pred_all)):
                pred_per = torch.nn.functional.softmax(pred_all[i], dim=1)
                pred = pred_per.data.max(1)[1]  # get the index of the max log-probability
                correct_per[0][i] += pred.eq(target.data.view_as(pred)).cpu().sum()
                pred_all_onehot.append(pred_per.cpu().numpy())
            pred = np.mean(np.array(pred_all_onehot), axis=0)
            pred = torch.from_numpy(pred)
            pred = pred.cuda()
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print('DX', '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource accnum {}'.format(correct_per))
    return 100. * correct / len(target_test_loader.dataset)

def model_save(name,item,title):
    f = os.path.join(name, '{}.pt'.format(title))
    torch.save(item,f)
    print('Saved model:', f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MASS parameters')
    parser.add_argument('--batch_size', type=int, default=32, help='size for one batch, integer')
    parser.add_argument('--iteration', type=int, default=2000, help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', type=float, default=True, help='using CUDA or not')
    parser.add_argument('--seed', type=float, default=8, help='random seed')
    parser.add_argument('--log_interval', type=float, default=10, help='log_interval')
    parser.add_argument('--class_num', type=float, default=10, help='number of class')
    parser.add_argument('--domain_num', type=float, default=14, help='number of source domains')
    args = parser.parse_args()

    human = ['CL', 'DX', 'HRZ','LQ','LY','LYK','SZH','WYX','XMD','YM','YTH','YZ','YZZ','ZSK','ZYJ']
    data_path = '/home/ps/LYK/MyMSDA4/data/'
    dataloader = utils.Dataloader(data_path, human, args.batch_size)
    # data save path
    date = '2022-3-28'
    hourandmin = '0849'
    save_path = './result/' + date + '/' +hourandmin
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    #leave-one-subject-out split
    for j in range(len(human)):
        net = models.MASS(num_classes=args.class_num,num_domains=args.domain_num)
        optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        if args.cuda:
            net.cuda()
        num = list(range(len(human)))
        num.pop(j)
        target_train_loader, target_test_loader = dataloader[j], dataloader[j]
        source_loader = []
        for k in num:
            source_loader.append(dataloader[k])
        MASS_train(net,
                   j,
                   args.domain_num,
                   optimizer,
                   scheduler,
                   args.iteration,
                   args.cuda,
                   args.log_interval,
                   save_path)
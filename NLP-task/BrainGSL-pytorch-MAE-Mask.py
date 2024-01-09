import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from sklearn import metrics
import model1
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)






        

# BD load  导入BD_sig.npy
print('loading MDD data...')
# X = np.load('../data/correlation&label/BD_sig.npy')
# Y = np.load('../data/correlation&label/BD_label.npy')

#导入BD train load
X = np.load('../data/correlation&label/MDD_sig.npy')
Y = np.load('../data/correlation&label/MDD_label.npy')
print(X.shape)

# Data = scio.loadmat('./data/ABIDE/data/correlation/pcc_correlation_871_cc116_.mat')
# # print(cc116.keys()) # connectivity
# X = Data['connectivity']
# # print(X[0][0])
# # print(cc116.shape) # 871 116 116
# Y = np.loadtxt('./data/ABIDE/data/labels/871_label_cc116.txt')

where_are_nan = np.isnan(X)  # 找出数据中为nan的
where_are_inf = np.isinf(X)  # 找出数据中为inf的
for bb in range(0,392):
    for i in range(0, 116):
        for j in range(0, 116):
            if where_are_nan[bb][i][j]:
                X[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X[bb][i][j] = 1

print('---------------------')
print('X', X.shape) # N M M
print('Y', Y.shape)
print('---------------------')

epochs_rec = 30 # 20 675 # 10 684 # 0 663 # 30 666 # 15 675 5 666
epochs = 50 + epochs_rec # 116 671

batch_size = 32 # 64 0.660
dropout = 0.5#欠拟合，变小 0.5
lr = 0.005
decay = 0.01#欠拟合，变小 0.01
result = []
acc_final = 0
result_final = []
ts_result=[['kfold_index','prec','recall','acc','F1','auc']]
from sklearn.model_selection import KFold
for ind in range(1):
    setup_seed(ind)
    # Masked
    nodes_number = 75
    nums = np.ones(116) # 制作mask模板
    nums[:116-nodes_number] = 0 # 根据设置的nodes number 决定多少是mask 即mask比例
    np.random.seed(1)
    np.random.shuffle(nums) # 116 75%1 25%0 shuffle打散
    # print(nums)
    # print('nums----------')
    Mask = nums.reshape(nums.shape[0], 1) * nums # 116 116
    # print('X before ', X.shape)
    Masked_X = X * Mask # 将部分转换为 0（masked）
    # print('X after ', X.shape)
    X0=X
    Masked_X_rest = X - Masked_X
    # print('Masked_X_rest ', Masked_X_rest[0][])
    J = nodes_number # J 拷贝出一份
    for i in range(0, J):
        ind = i
        # ind = nums.shape[0] - 1 - i
        if nums[ind] == 0:
            for j in range(J, 116):
                if nums[j] == 1:
                    Masked_X[:, [ind, j], :] = Masked_X[:, [j, ind], :]
                    Masked_X[:, :, [ind, j]] = Masked_X[:, :, [j, ind]]
                    Masked_X_rest[:, [ind, j], :] = Masked_X_rest[:, [j, ind], :]
                    Masked_X_rest[:, :, [ind, j]] = Masked_X_rest[:, :, [j, ind]]
                    X0[:, [ind, j], :] = X0[:, [j, ind], :]
                    X0[:, :, [ind, j]] = X0[:, :, [j, ind]]
                    J = j + 1
                    break
                
            # Masked_X = np.delete(Masked_X, ind, axis=1)
            # Masked_X = np.delete(Masked_X, ind, axis=2)
    # print(Masked_X[0,1,:])
    # print(Masked_X_rest[0,1,:]) # 只有后面的有值
    X_0 = Masked_X # X_0 是残缺的 只有前面unmasked的有值
    Masked_X = Masked_X[:, :nodes_number, :nodes_number]
    unMasked_X = Masked_X[:, nodes_number:, nodes_number:] # 这个是被mask的那些pcc

    X = Masked_X 
    X_unmasked = Masked_X_rest  

    acc_all = 0
    kf = KFold(n_splits=10, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X, Y):
        kfold_index += 1
        print('kfold_index:', kfold_index)
        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]

        X_trainval_masked_rest = X_unmasked[trainval_index]
        X_test_masked_rest = X_unmasked[test_index]

        X_trainval_0, X_test_0 = X0[trainval_index], X0[test_index]
        for train_index, val_index in kf.split(X_trainval, Y_trainval):
            # 取消验证集
            X_train, X_val = X_trainval[:], X_trainval[:]
            Y_train, Y_val = Y_trainval[:], Y_trainval[:]

            X_train_masked_rest = X_trainval_masked_rest[:]
            X_val_masked_rest = X_trainval_masked_rest[:]

            X_train_0 = X_trainval_0[:] # 完整的A
            X_val_0 = X_trainval_0[:]
        print('X_train', X_train.shape)
        print('X_val', X_val.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_val', Y_val.shape)
        print('Y_test', Y_test.shape)
        
        # train dataset average
        X_train_number = X_train.shape[0]
        X_avg = X_train_0.sum(axis=0)
        # X_avg = X_avg.sum(axis=2)
        # print('------X avg-------', X_avg.shape)
        X_avg = X_avg / X_train_number

        for k in range(X_train_0.shape[0]):
            for i in range(X_train_0.shape[1]):
                for j in range(X_train_0.shape[2]):
                    if abs(X_train_0[k][i][j] < 0.2):
                        X_train_0[k][i][j] = 0
                    else:
                        X_train_0[k][i][j] = 1
        for k in range(X_test_0.shape[0]):
            for i in range(X_test_0.shape[1]):
                for j in range(X_test_0.shape[2]):
                    if abs(X_test_0[k][i][j] < 0.2): # 0.3 679 0.4 684 0.45 661 0.6 671 0.2 675
                        X_test_0[k][i][j] = 0
                    else:
                        X_test_0[k][i][j] = 1

        for k in range(X_avg.shape[0]):
            for i in range(X_avg.shape[1]):
                if abs(X_avg[k][i] < 0.15): # 0.2 675  0.4 646 0.3 677 0.1 675 0.15 696 ！！！！！
                    X_avg[k][i] = 0
                else:
                    X_avg[k][i] = 1
        
        # for k in range(X_avg_test.shape[0]):
        #     for i in range(X_avg_test.shape[1]):
        #         if abs(X_avg_test[k][i] < 0.5):
        #             X_avg_test[k][i] = 0
        #         else:
        #             X_avg_test[k][i] = 1

        #print(X_avg)
        X_avg = X_avg.reshape(1, X_avg.shape[0], X_avg.shape[1]) #  
        X_avg = np.repeat(X_avg, X_train.shape[0], 0)
        X_avg_test = np.repeat(X_avg, X_test.shape[0], 0)

        X_train_0 = X_train_0
        X_test_0 = X_test_0

        X_train_0 = X_avg
        # X_test_0 = X_avg_test
        # print(X_test_0)

        X_masked = np.zeros([X_train.shape[0], 116-nodes_number, 48])
        X_masked_test = np.zeros([X_test.shape[0], 116-nodes_number, 48])
        for i in range(X_masked.shape[0]):
            for j in range(X_masked.shape[1]):
                X_masked[i][j] = np.random.normal(loc=0.0, scale=1.0, size=48)
        for i in range(X_masked_test.shape[0]):
            for j in range(X_masked_test.shape[1]):
                X_masked_test[i][j] = np.random.normal(loc=0.0, scale=1.0, size=48)

        # model
        model = model1.Model(dropout=dropout, num_class=2)
        model.to(device)
        # for p in model.parameters():
        #     if p.requires_grad:
        #         print(p.name, p.data.shape)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        optimizer2 = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
         #lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 116, 250], gamma=0.8)
        loss_fn = nn.CrossEntropyLoss()
        loss_rec = nn.MSELoss()
        # loss_fn = nn.MSELoss()
        
        best_val = 0
        
        
        # train
        for epoch in range(1, epochs+1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            num_batch = X_train.shape[0] // int(batch_size)
            
            loss_train = 0
            for bn in range(num_batch):
                if bn == num_batch - 1:
                    batch = idx_batch[bn * int(batch_size):]
                else:
                    batch = idx_batch[bn * int(batch_size) : (bn+1) * int(batch_size)]
                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]
                train_data_batch_A = X_train_0[batch]
                train_data_batch_rest = X_train_masked_rest[batch]
                train_data_batch_maskedX = X_masked[batch]
                # print(train_data_batch[0])

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)
                train_data_batch_A_dev = torch.from_numpy(train_data_batch_A).float().to(device)
                train_data_batch_rest_dev = torch.from_numpy(train_data_batch_rest).float().to(device)
                train_data_batch_maskedX_dev = torch.from_numpy(train_data_batch_maskedX).float().to(device)

                optimizer.zero_grad()
                # x:未被遮掩的结点，A：全部的结点，masked_x:被遮掩的结点
                print('train_data_batch_dev',train_data_batch_dev)
                print('train_data_batch_dev', train_data_batch_dev.shape)
                print('train_data_batch_A_dev',train_data_batch_A_dev)
                print('train_data_batch_A_dev', train_data_batch_A_dev.shape)
                print('train_data_batch_maskedX_dev',train_data_batch_maskedX_dev)
                print('train_data_batch_maskedX_dev', train_data_batch_maskedX_dev.shape)
                outputs,rec= model( train_data_batch_dev,train_data_batch_A_dev ,train_data_batch_maskedX_dev)
                #, rec,_,rec
                loss1 = loss_fn(outputs, train_label_batch_dev)
                # print(train_data_batch_rest_dev[0][0])
                loss2 = loss_rec(rec, train_data_batch_rest_dev)

                if epoch <= epochs_rec:#每次交叉验证时，前epochs_rec个epoch只优化自监督模型
                    loss = loss2
                    loss_train += loss
                    loss.backward()
                    optimizer.step()
                else:
                   loss = loss1 # + loss2#之后的50个优化自监督模型和下游模型
                   loss_train += loss
                   loss.backward()
                   optimizer2.step()

                
            
            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
                
            # val
            if epoch % 80 == 0 and epoch > epochs_rec:
                # model.eval()
                
                # val_data_batch_dev = torch.from_numpy(X_val).float().to(device)
                # val_label_batch_dev = torch.from_numpy(Y_val).long().to(device)
                # outputs, rec = model(val_data_batch_dev)
                # loss1 = loss_fn(outputs, val_label_batch_dev)
                # loss2 = loss_rec(rec, val_data_batch_dev)
                # loss = loss1 + loss2
                # _, indices = torch.max(outputs, dim=1)
                # preds = indices.cpu()
                # # print(preds)
                # acc_val = metrics.accuracy_score(preds, Y_val)
                if True: #acc_val > best_val:
                    # best_val = acc_val
                    model.eval()
                    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                    test_data_batch_A_dev = torch.from_numpy(X_test_0).float().to(device)
                    test_data_batch_maskedX_dev = torch.from_numpy(X_masked_test).float().to(device)
                    #outputs,rec= model( train_data_batch_dev,train_data_batch_A_dev ,train_data_batch_maskedX_dev)
                    outputs, _ = model(test_data_batch_dev ,test_data_batch_A_dev, test_data_batch_maskedX_dev)
                    _, indices = torch.max(outputs, dim=1)
                    preds = indices.cpu()
                    # print(preds)
                    acc = metrics.accuracy_score(preds, Y_test)
                    # print('Test acc', acc)
                    auc=0
                    try:
                        auc=metrics.roc_auc_score(preds, Y_test, average='macro', sample_weight=None)  ## y_true=ground_truth
                    except ValueError:
                        pass
                    # test_result = {'kfold_index':kfold_index,
                    #                'prec': metrics.precision_score(preds, Y_test),
                    #                'recall': metrics.recall_score(preds, Y_test),
                    #                'acc': metrics.accuracy_score(preds, Y_test),
                    #                'F1': metrics.f1_score(preds, Y_test),
                    #                'matrix': metrics.confusion_matrix(preds, Y_test),
                    #                'auc': auc
                    #                }
                    # print(test_result)
                    test_result=[kfold_index,metrics.precision_score(preds, Y_test),metrics.recall_score(preds, Y_test), metrics.accuracy_score(preds, Y_test),
                                 metrics.f1_score(preds, Y_test),auc]

                    ts_result.append(test_result)
                # print('Test acc', acc_val)

        # if epoch % 1 == 0:
        #每一次交叉验证后，保存模型，共保存10次
        torch.save(model.state_dict(), './models/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        print(result)
        acc_all += acc
    temp = acc_all / 10
    # acc_final += temp
    # result_final.append(temp)
    # print(result)

ACC = temp
print('ACC:',ACC)

# 684
import xlwt
f = xlwt.Workbook('encoding = utf-8') #设置工作簿编码
sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表

for j in range(len(ts_result)):
 for i in range(len(ts_result[j])):
    sheet1.write(j,i,ts_result[j][i]) #写入数据参数对应 行, 列, 值

f.save('MDD_BrainGSL_pytorch_MAE_Mask_results.xls')#保存.xls到当前工作目录


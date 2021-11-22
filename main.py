import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from Model.models import *
from Train.train import *
from Utils.data import *


parser = argparse.ArgumentParser()
parser.add_argument('--target_area', type=str, default='Urban-core',
                    help='study area')
parser.add_argument('--seq_dim', type=int, default=12,
                    help='history steps')
parser.add_argument('--pred_len', type=int, default=12,
                    help='time steps to predict')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--num_adj', type=int, default=3,
                    help='number of neighbors to consider')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--out_features', type=int, default=4,
                    help='the number of output features')
parser.add_argument('--out_channels', type=int, default=4,
                    help='the number of output features')
parser.add_argument('--train_ratio', type=float, default=21/30,
                    help='training set ratio')
parser.add_argument('--val_ratio', type=float, default=2/30,
                    help='validation set ratio')
args = parser.parse_args()


if args.target_area == "Urban-core":
    # urban 1 dataset
    df1 = pd.read_csv("/media/hdd1/AAAI/urban1/urban1_1.csv", header=None)
    df2 = pd.read_csv("/media/hdd1/AAAI/urban1/urban1_2.csv", header=None)
    df3 = pd.read_csv("/media/hdd1/AAAI/urban1/urban1_3.csv", header=None)

    df1 = np.array(df1)
    df2 = np.array(df2)
    df3 = np.array(df3)

    tmp_df = np.concatenate((df1, df2, df3), axis=1)

    tmp_limit = tmp_df[:, 4]
    tmp_limit = tmp_limit.reshape((tmp_df.shape[0], 1))
    df4 = np.multiply(tmp_df[:, 7:], tmp_limit)
    adj_direc='urban1_adj' # Expanded_adj & urban1_adj
    num_epochs = 500

elif args.target_area == "Urban-mix":
    # Expanded Area dataset
    df1 = pd.read_csv("/media/hdd1/AAAI/ExpandedArea/ExpandedArea.csv", header=None)
    df1 = np.array(df1)

    df4 = df1[:, 7:]
    adj_direc = 'Expanded_adj'
    num_epochs = 500

speed_matrix = np.transpose(df4)
speed_matrix = pd.DataFrame(speed_matrix)

del df1

n_links = speed_matrix.shape[1]
hidden_dim = n_links*2
input_dim = n_links * args.out_channels

W = load_weight_matrix(num_adj=args.num_adj, direc=adj_direc, plain=1, speed_limit=1,
                       speedCat=0, speedChange=0, dist=0, dist_og=0, angle=0)
# dist = length
# dist_og = distance between links

print("Load weight matrix for {} ranks, and of size [{}, {}, {}] for each rank".format(
    len(W), W[0].shape[0], W[0].shape[1], W[0].shape[2]))

# idx = []
# for i in range(num_adj):
#     for j in range(W[0].shape[2]):
#         tmp_idx = np.where(W[i][:, :, j] != 0)
#         idx.append(tmp_idx)

# Data loader: defined batch size is for train data loader only.
#              For valid and test dataset, batch size is set to 8 for memory limitation
train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset(
    speed_matrix,
    BATCH_SIZE=args.batch_size,
    seq_len=args.seq_dim,
    pred_len=args.pred_len,
    train_propotion=args.train_ratio,
    valid_propotion=args.val_ratio,
    z_norm=True)

# print('before model:\t{}'.format(np.around(torch.cuda.memory_allocated()/1024/1024)))
for i in range(len(W)):
    W[i] = torch.from_numpy(W[i]).float().cuda()
    # W[i] = torch.from_numpy(W[i]).float().to(device)
model = GCLSTM_model(input_dim, hidden_dim, args.out_features, args.out_channels,
                     adj=W, output_dropout=0.3, GCN=True, bias=True) # 359*12, 1200, 2, 359, #12, 4

# print('after model:\t{}'.format(np.around(torch.cuda.memory_allocated()/1024/1024)))

if torch.cuda.is_available():
    model.cuda()
    # model.to(device)
# print('after model to cuda:\t{}'.format(np.around(torch.cuda.memory_allocated()/1024/1024)))

###### TRAINING ######
loss_MSE = torch.nn.MSELoss()
loss_L1 = torch.nn.L1Loss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=5e-4) # 5e-4
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
# scheduler=None

trainedModel, loss_list, valid_loss_list, avg_time, num_iter = trainModel(
    model, train_dataloader, valid_dataloader, loss_MSE, loss_L1,
    optimizer, scheduler=scheduler, num_epochs=num_epochs)

del train_dataloader, valid_dataloader, model, W

print('=================================================\nTEST RESULT FOR PRED_LEN: {}'
      '\n================================================='.format(args.pred_len))

test_loss, test_link_RMSE, preds_test = testModel(trainedModel, test_dataloader, loss_MSE, loss_L1)

print("Average computation time for 1 epoch in training : {:0.3f}sec".format(avg_time))
# sys.exit()

plt.plot(loss_list)
plt.plot(valid_loss_list)
plt.ylim([0, 75])
plt.title("Train & Valid Loss for MW-TGC\n" + args.target_area)
plt.legend(["Train", "Valid"])
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.grid()
plt.show()

plt.plot(test_link_RMSE[5, :], 'o')
plt.xlabel("Link No.")
plt.ylabel("RMSE")
plt.title("Prediction error on each link on " + args.target_area + "dataset \non 30min forecast (MW-TGC)")
plt.show()
"""
np.savetxt("/home/user/PycharmProjects/MWTGC_revision2/final_results/link_RMSE_" + args.target_area
             + "_MW-TGC.csv", test_link_RMSE, delimiter=',')
# torch.save(trainedModel.state_dict(), "./urban_MW-TGC")
np.savetxt("/home/user/PycharmProjects/MWTGC_revision2/final_results/train_loss_" +
           args.target_area + "_MW-TGC.csv", loss_list, delimiter=',')
np.savetxt("/home/user/PycharmProjects/MWTGC_revision2/final_results/valid_loss_" +
           args.target_area + "_MW-TGC.csv", valid_loss_list, delimiter=',')

# np.savetxt('/home/user/PycharmProjects/MWTGC_revision2/1hrpred_MW-TGC_' + args.target_area + '.csv',
#            preds_test, delimiter=',')

for i in range(preds_test.shape[1]):
    pd.DataFrame(preds_test[:, i, :]).to_csv('/home/user/PycharmProjects/MWTGC_revision2/final_results/' +
                                             str(i) + 'pred_MW-TGC_' + args.target_area + '.csv',
                                             header=None, index=False)

filePATH = "/home/user/whatever-Net/torch/output2/LSTM_linkwise_"
fileNAME = 'epoch' + str(num_epochs) + '_batch' + str(batch_size) + '_' + '_lr' + str(learning_rate) \
    + '_num_adj' + str(num_adj) + '_' + args.target_area + '.csv'
np.savetxt(filePATH + fileNAME, test_link_RMSE, delimiter=',')
"""
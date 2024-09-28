import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
import utils
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pickle

class GAFM( torch.nn.Module ):

    def __init__( self,n_entitys, n_users, n_items, dim):

        super( GAFM, self ).__init__( )

        self.items = nn.Embedding( n_items, dim, max_norm = 1 )
        self.users_df = nn.Embedding( n_entitys, dim, max_norm = 1 )
        self.users_like = nn.Embedding( n_users, dim, max_norm = 1 )
        self.users_dislike = nn.Embedding( n_users, dim, max_norm = 1 )

        self.query = nn.Linear(dim, dim)
        self.key1 = nn.Linear(dim, dim)
        self.value1 = nn.Linear(dim, dim)

        self.key2 = nn.Linear(dim, dim)
        self.value2 = nn.Linear(dim, dim)

        self.f1 = nn.Linear(dim, 520)
        self.f2 = nn.Linear(520, 1)
        self.f3 = nn.Linear(dim, 520)
        self.f4 = nn.Linear(520, 1)

        self.fc1 = nn.Linear(dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def FMaggregator(self, feature_embs):
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        output = square_of_sum - sum_of_square
        return output
    def __getEmbeddingByNeibourIndex( self, orginal_indexes, nbIndexs, aggEmbeddings ):
        new_embs = []
        # print("orginal_indexes:",orginal_indexes)
        for v in orginal_indexes:
            embs = aggEmbeddings[ torch.squeeze( torch.LongTensor( nbIndexs.loc[v].values )) ]
            new_embs.append( torch.unsqueeze( embs, dim = 0 ) )

        return torch.cat( new_embs, dim = 0 )

    def gnnForward( self, adj_lists):
        n_hop = 1
        for df in adj_lists:
            if n_hop == 1:
                entity_embs = self.users_df( torch.LongTensor( df.values ).to(device) )
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex( df.values, neighborIndexs, aggEmbeddings )
            target_embs = self.users_df( torch.LongTensor( df.index).to(device) )
            aggEmbeddings = self.FMaggregator( entity_embs )

            if n_hop < len( adj_lists ):
                neighborIndexs = pd.DataFrame( range( len( df.index ) ), index = df.index )

            aggEmbeddings =   aggEmbeddings + target_embs
            n_hop +=1
        # [ batch_size, dim ]
        return aggEmbeddings

    def gnnForwardpositive(self, adj_lists):
        n_hop = 1
        # print(adj_lists)
        # exit(0)
        for df in adj_lists:
            if n_hop == 1:

                entity_embs = self.users_like(torch.LongTensor(df.values-69).to(device))
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)

            weights = []
            for i in range(len(df.index)):
                if len(df.values[0]) ==3:
                    weights.append([G2.edges[(df.values[i][0], df.index[i])]['weight'],G2.edges[(df.values[i][1], df.index[i])]['weight'],G2.edges[(df.values[i][2], df.index[i])]['weight']])
                if len(df.values[0]) ==5:
                    if df.index[i] == df.values[i][0]:
                        weights.append([0,0,0,0,0])
                    else:
                        weights.append([G2.edges[(df.values[i][0], df.index[i])]['weight'],G2.edges[(df.values[i][1], df.index[i])]['weight'],G2.edges[(df.values[i][2], df.index[i])]['weight'],G2.edges[(df.values[i][3], df.index[i])]['weight'],G2.edges[(df.values[i][4], df.index[i])]['weight']])

            # print(weights)
            weights = torch.Tensor(weights).to(device)
            weights = weights.view(len(df.index),len(df.values[0]),1)

            target_embs = self.users_like(torch.LongTensor(df.index-69).to(device))

            aggEmbeddings = weights* entity_embs
            aggEmbeddings = torch.sum(aggEmbeddings,dim=1)
            aggEmbeddings = aggEmbeddings + target_embs

            if n_hop < len(adj_lists):
                neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)

            n_hop += 1

        return aggEmbeddings
    def gnnForwardnegative(self, adj_lists):
        n_hop = 1

        for df in adj_lists:
            if n_hop == 1:

                entity_embs = self.users_dislike(torch.LongTensor(df.values - 69).to(device))
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)

            weights = []
            for i in range(len(df.index)):
                if df.index[i] == df.values[i][0]:
                    weights.append([0,0,0,0,0])
                else:
                    weights.append([G3.edges[(df.values[i][0], df.index[i])]['weight'],
                                G3.edges[(df.values[i][1], df.index[i])]['weight'],
                                G3.edges[(df.values[i][2], df.index[i])]['weight'],
                                G3.edges[(df.values[i][3], df.index[i])]['weight'],
                                G3.edges[(df.values[i][4], df.index[i])]['weight']])

            # print(weights)
            weights = torch.Tensor(weights).to(device)
            weights = weights.view(len(df.index), len(df.values[0]), 1)

            target_embs = self.users_dislike(torch.LongTensor(df.index - 69).to(device))

            aggEmbeddings = weights * entity_embs
            aggEmbeddings = torch.sum(aggEmbeddings, dim=1)
            aggEmbeddings = aggEmbeddings + target_embs
            # print("aggEmbeddings:", aggEmbeddings.shape)
            if n_hop < len(adj_lists):
                neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
                # print(neighborIndexs)
                # exit(0)
            n_hop += 1
        # [ batch_size, dim ]
        # exit(0)
        return aggEmbeddings
    def attentionPositive_Negative(self, users_like, users_dislike, users_df):

        Q = self.query(users_df)
        users_like_k = self.key1(users_like)
        users_like_v = self.value1(users_like)
        users_dislike_k = self.key2(users_dislike)
        users_dislike_v = self.value2(users_dislike)

        score1 = users_like_k * Q
        score2 = users_dislike_k * Q

        score1 = self.f1(score1)
        score1 = self.relu(score1)
        score1 = torch.sigmoid(self.f2(score1))
        score2 = self.f3(score2)
        score2 = self.relu(score2)
        score2 = torch.sigmoid(self.f4(score2))

        user1 = score1 * users_like_v
        user2 = score2 * users_dislike_v
        user = user1 + user2

        return user
    def forward( self,u, i, adj_lists_G1,adj_lists_G2,adj_lists_G3):
        items = self.items(i.to(device))
        users_df = self.gnnForward(adj_lists_G1)
        users_positive = self.gnnForwardpositive(adj_lists_G2)
        users_negative = self.gnnForwardnegative(adj_lists_G3)
        users=self.attentionPositive_Negative(users_positive,users_negative,users_df)
        uv = torch.cat((users, items), dim=1)
        uv = self.fc1(uv)
        uv = self.relu(uv)
        uv = F.dropout(uv, p=0.5, training=self.training)
        uv = self.fc2(uv)
        uv = self.relu(uv)
        uv = F.dropout(uv, p=0.5, training=self.training)
        uv = self.fc3(uv)
        uv = torch.squeeze(uv)
        logit = torch.sigmoid(uv)
        return logit
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            print("保存模型！")
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), 'BOOK_Model/model.pth')
        self.val_loss_min = val_loss
@torch.no_grad()
def doEva(net, d):
    net.eval()
    criterion = torch.nn.BCELoss()
    d = torch.LongTensor(d).to(device)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    rr = r.float()
    u_index = u.detach().cpu().numpy()
    adj_lists_G1 = utils.graphSage4RecAdjType(G1, u_index, [2])
    adj_lists_G2 = utils.graphSage4RecAdjType2(G2, u_index)
    adj_lists_G3 = utils.graphSage4RecAdjType2(G3, u_index)
    out = net(u, i, adj_lists_G1, adj_lists_G2, adj_lists_G3)
    loss = criterion(out, rr)
    y = np.array([i for i in out.cpu().detach().numpy()])
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out.cpu().detach().numpy()])
    y_true = r.cpu().detach().numpy()
    auc_score = roc_auc_score(y_true, y)

    return auc_score, loss
def train(epoch=50, batchSize=128, lr=0.00001, eva_per_epochs=1):


    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for e in range(epoch):
        net.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_data, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy()).to(device)
            optimizer.zero_grad()
            u_index = u.detach().cpu().numpy()
            adj_lists_G1 = utils.graphSage4RecAdjType(G1, u_index, [2])
            adj_lists_G2 = utils.graphSage4RecAdjType2(G2, u_index)
            adj_lists_G3 = utils.graphSage4RecAdjType2(G3, u_index)
            logits = net(u.to(device),i.to(device),adj_lists_G1,adj_lists_G2,adj_lists_G3)
            loss = criterion(logits,r)
            all_lose+=loss
            loss.backward()
            optimizer.step()

        print(f'epoch {e}')
        net.eval()
        with torch.no_grad():
            auc_score, val_loss = doEva(net, val_data)
        print("val_loss:", val_loss)
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
def test():
    print("Testing....")
    model_path = f'BOOK_Model/model.pth'
    net.load_state_dict(torch.load(model_path))
    auc_score, loss = doEva(net, test_data)
    print('test:  auc_score{:.4f} | loss{:.4f} '.format(auc_score, loss))
if __name__ == '__main__':
    print('Reading triplet data...')
    user_set, item_set, train_data = utils.getdata(
        "Data/BookCrossing/train_set.csv")
    _, _, test_data = utils.getdata("Data/BookCrossing/test_data.csv")  # print(len(train_set))
    _, item_set2, val_data = utils.getdata("Data/BookCrossing/val_data.csv")  # print(len(train_set))

    print('Reading graphs data...')
    with open('Data/BookCrossing/G1.pkl', 'rb') as f:
        G1 = pickle.load(f)
    with open('Data/BookCrossing/G2.pkl', 'rb') as f:
        G2 = pickle.load(f)
    with open('Data/BookCrossing/G3.pkl', 'rb') as f:
        G3 = pickle.load(f)
    print("Model Training...")
    if not os.path.exists("BOOK_Model"):
        os.makedirs("BOOK_Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GAFM(max(user_set) + 1,len(user_set), max(item_set2) + 1, 128).to(device)
    # train()
    test()

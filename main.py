import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import data_process as dp
from model import Linear
import train

def parse():
    parse = argparse.ArgumentParser('Dataset Select')
    parse.add_argument('--dataset',help='choose the dataset you want to analyze',default= None,type=str)
    parse.add_argument('--all_influ_elem',default=True,type=bool)
    return parse.parse_args()

def DataSet(train_data,device,batchsize=4320,drop=False):
    x,influ_elem,y=train_data
    x = torch.from_numpy(x).type(torch.Tensor).to(device)
    influ_elem = torch.from_numpy(influ_elem).type(torch.Tensor).to(device)
    y = torch.from_numpy(y).type(torch.Tensor).to(device)

    data = TensorDataset(x,influ_elem,y)#参数必须是tensor
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=False,drop_last=drop)

    return data_loader

if __name__ == '__main__':
    args = parse()
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    train_data,test_data = dp.Data_process(args.dataset,args.all_influ_elem)
    train_data = DataSet(train_data,device)
    x_test,test_influence,y_test = test_data

    x_test =  torch.from_numpy(x_test).type(torch.Tensor).to(device)
    test_influence = torch.from_numpy(test_influence).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

    model = Linear.Linnear()

    train.train(model,train_data)

    
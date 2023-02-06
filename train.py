import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_detial(y_train_pred,y_train,hist = [1,2,3]):
    sns.set_style("darkgrid")

    predict = pd.DataFrame(y_train_pred.cpu().detach().numpy())
    original = pd.DataFrame(y_train.cpu().detach().numpy())


    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = range(3804), y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction", color='tomato')

    ax.set_title('Parking Rate', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Rate", size = 14)
    ax.set_xticklabels('', size=10)


    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(18)
    plt.show()

def absloss(pred,true):
    loss = np.sum(abs(pred-true)/pred)/504*100
    print("预测的偏差为：",round(loss,2),"%")


def train(model,train_data,warm_epochs = 100,num_epochs = 630):
    hist = []
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for t in range(warm_epochs):
        print("Warming Epoch ", t, ": ")
        for step,data in enumerate(train_data):
            x,w,y=data
            y_train_pred = model(x,w)
            loss = criterion(y_train_pred, y)
            print("Warming batch ", step, "MSE: ", loss.item())
            hist .append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,verbose=True)

    for t in range(num_epochs):
        print("Cosine Epoch ", t, ": ")
        for step,data in enumerate(train_data):
            x,w,y=data
            y_train_pred = model(x,w)
            loss = criterion(y_train_pred, y)
            print("Cosine batch ", step, "MSE: ", loss.item())
            hist .append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        CosineLR.step()

    training_time = time.time() - start_time
    draw_detial(y_train_pred,y,hist)
    print("Training time: {}".format(training_time))



import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from RNN_Series1 import SimpleRNN, SimpleLSTM, SimpleGRU, SimpleBILSTM
from torch.utils.data import Dataset, DataLoader
import string
from sklearn.preprocessing import LabelEncoder
import time

letters = string.ascii_letters + " .,;'"
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

def load_data():
    data = pd.read_csv('name_classfication.txt', sep='\t', names = ['name', 'country'])
    X = data[['name']]
    lb = LabelEncoder()
    y = data['country']
    y = lb.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


class create_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.length = len(self.X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.zeros(10, len(letters), dtype=torch.float, device=device)
        for i, letter in enumerate(self.X.iloc[idx, 0]):
            if i == 10:
                break
            data[i, letters.index(letter)] = 1
        label = torch.tensor(self.y[idx], dtype=torch.long, device=device)
        return data, label


def train_rnn():
    X_train, X_test, y_train, y_test = load_data()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_list = []
    acc_list = []
    val_acc_list = []
    val_loss_list = []
    epochs = 10
    my_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_test, y_test)
    my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(y_test), shuffle=True)
    my_rnn = SimpleRNN(len(letters), 128, 2)
    my_rnn.to(device)
    optimizer = torch.optim.Adam(my_rnn.parameters(), lr=0.001)
    start_time = time.time()

    for epoch in range(epochs):
        my_rnn.train()
        total_loss = 0
        total_acc = 0
        total_sample = 0
        for i, (X, y) in enumerate(my_dataloader):
            output, hidden = my_rnn(X, my_rnn.init_hidden(batch_size=len(y)))
            total_sample += len(y)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediction = output.argmax(dim=1)
            acc_num = torch.sum(prediction == y).item()
            total_acc += acc_num
        loss_list.append(total_loss / total_sample)
        acc_list.append(total_acc / total_sample)

        my_rnn.eval()
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(val_dataloader):
                output, hidden = my_rnn(X_val, my_rnn.init_hidden(batch_size=len(y_test)))
                loss = criterion(output, y_val)
                prediction = output.argmax(dim=1)
                acc_num = torch.sum(prediction == y_val).item()
                val_acc_list.append(acc_num / len(y_val))
                val_loss_list.append(loss.item() / len(y_val))
                print(
                    f'epoch: {epoch + 1}, train_loss: {total_loss / total_sample:.2f}, train_acc: {total_acc / total_sample:.2f}, val_loss: {loss.item() / len(y_val):.2f}, val_acc: {acc_num / len(y_val):.2f}, time: {time.time() - start_time : .2f}')
    torch.save(my_rnn.state_dict(), 'rnn.pt')
    plt.plot(np.arange(1, 11), loss_list, label='Training Loss')
    plt.plot(np.arange(1, 11), val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, 11))
    plt.title('Loss')
    plt.legend()
    plt.savefig('logg.png')
    plt.show()
    plt.plot(np.arange(1, 11), acc_list, label='Training Accuracy')
    plt.plot(np.arange(1, 11), val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1, 11))
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()


def train_bilstm():
    X_train, X_test, y_train, y_test = load_data()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_list = []
    acc_list = []
    val_acc_list = []
    val_loss_list = []
    epochs = 10
    my_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_test, y_test)
    my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(y_test), shuffle=True)
    my_rnn = SimpleBILSTM(len(letters), 128, 2)
    my_rnn.to(device)
    optimizer = torch.optim.Adam(my_rnn.parameters(), lr=0.001)
    start_time = time.time()

    for epoch in range(epochs):
        my_rnn.train()
        total_loss = 0
        total_acc = 0
        total_sample = 0
        for i, (X, y) in enumerate(my_dataloader):
            hidden, c0 = my_rnn.init_hidden(batch_size=len(y))
            output, hidden, c = my_rnn(X, hidden, c0)
            total_sample += len(y)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediction = output.argmax(dim=1)
            acc_num = torch.sum(prediction == y).item()
            total_acc += acc_num
        loss_list.append(total_loss / total_sample)
        acc_list.append(total_acc / total_sample)

        my_rnn.eval()
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(val_dataloader):
                hidden, c0 = my_rnn.init_hidden(batch_size=len(y_val))
                output, hidden, c = my_rnn(X_val, hidden, c0)
                loss = criterion(output, y_val)
                prediction = output.argmax(dim=1)
                acc_num = torch.sum(prediction == y_val).item()
                val_acc_list.append(acc_num / len(y_val))
                val_loss_list.append(loss.item() / len(y_val))
                print(
                    f'epoch: {epoch + 1}, train_loss: {total_loss / total_sample:.2f}, train_acc: {total_acc / total_sample:.2f}, val_loss: {loss.item() / len(y_val):.2f}, val_acc: {acc_num / len(y_val):.2f}, time: {time.time() - start_time : .2f}')

    torch.save(my_rnn.state_dict(), 'bilstm.pt')
    plt.plot(np.arange(1, 11), loss_list, label='Training Loss')
    plt.plot(np.arange(1, 11), val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, 11))
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    plt.plot(np.arange(1, 11), acc_list, label='Training Accuracy')
    plt.plot(np.arange(1, 11), val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1, 11))
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

if __name__ == '__main__':
    train_bilstm()
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(train_loss,test_loss):
    ep = np.arange(len(train_loss)) + 1
    plt.plot(ep, train_loss, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_loss, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    
def plot_acc(train_acc,test_acc):
    ep = np.arange(len(train_acc)) + 1
    plt.plot(ep, train_acc, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_acc, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title('Normalized Discounted Cumulative Gain')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

def plot_error(train_error,test_error):
    ep = np.arange(len(train_error)) + 1
    plt.plot(ep, train_error, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_error, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title('Mean Absolute Error')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(loc='upper right')

def plot_R2(train_acc,test_acc):
    ep = np.arange(len(train_acc)) + 1
    plt.plot(ep, train_acc, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_acc, color="red",  linewidth=1, linestyle="-", label="Test")
    plt.title('R-Squared Score')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
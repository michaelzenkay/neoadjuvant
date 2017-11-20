import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


def s6(loc,kfold_splits,epochs):
    accs = np.zeros([epochs,kfold_splits])
    losss = np.zeros([epochs,kfold_splits])
    val_accs = np.zeros([epochs,kfold_splits])
    val_losss = np.zeros([epochs,kfold_splits])

    for i in range(kfold_splits):
        with open(os.path.join(loc,'history' + str(i) + '.pkl'), 'rb') as f:
            history = pickle.load(f)
        # summarize history for accuracy
        accs[:,i] = history['acc']
        val_accs[:,i] = history['val_acc']
        plt.plot(history['acc'], lw=1, alpha=0.3 )#,label='Accuracy fold %d' % i)
        plt.plot(history['val_acc'], lw=1, alpha=0.3) #,label='Validation Accuracy fold %d' % i)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')


    #mean_acc =
    plt.plot(np.mean(accs,axis=1),label='Mean Accuracy ', lw=2, alpha=.8)
    plt.plot(np.mean(val_accs,axis=1), label='Mean Validation Accuracy ', lw=2, alpha=.8)
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(loc,'acc.png'))
    plt.close()


    for i in range(kfold_splits):
        with open(os.path.join(loc, 'history' + str(i) + '.pkl'), 'rb') as f:
            history = pickle.load(f)
        # summarize history for loss

        losss[:,i] = history['loss']
        val_losss[:,i] = history['val_loss']

        plt.plot(history['loss'], lw=1, alpha=0.3)#,label='Loss fold %d' % i)
        plt.plot(history['val_loss'], lw=1, alpha=0.3)#,label='Validation Loss fold %d' % i)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

    plt.plot(np.mean(losss, axis=1), label='Mean Loss ', lw=2, alpha=.8)
    plt.plot(np.mean(val_losss, axis=1), label='Mean Validation Loss ', lw=2, alpha=.8)
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(loc, 'loss.png'))
    plt.close()


loc    = '/data/breast/neo/results/vgg4block1dense30'
s6(loc,5, 1000)
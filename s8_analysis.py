import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
from scipy import interp

def s8(n_classes,results,kfold_splits):
    Sensitivity = np.empty(kfold_splits)
    Specificity = np.empty(kfold_splits)
    Accuracy    = np.empty(kfold_splits)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for n in range(n_classes):
        for i in range(kfold_splits):
            y_pred = np.loadtxt(os.path.join(results,'yscore' + str(i) + '.csv'),delimiter=",")
            y_test = np.loadtxt(os.path.join(results,'ytest' + str(i) + '.csv'),delimiter=",")

            # Calculate sensitivity, specificity, AUC accuracy
            FP , FN, TP, TN = (0,0,0,0)

            ## Count number of True/False Positive/Negative
            for p in range (len(y_test)):
                pred=np.argmax(y_pred[p,:])
                test=np.argmax(y_test[p,:])
                if   pred == test and pred != n:
                    TN = TN + 1
                elif pred == test and pred == n:
                    TP = TP + 1
                elif pred != test and pred != n:
                    FN = FN + 1
                elif pred != test and pred == n:
                    FP = FP + 1

            Sensitivity[i] = TP / (TP + FN)
            Specificity[i] = TN / (TN + FP)
            Accuracy[i] = (TP + TN) / (TP + TN + FP + FN)
            print('fold ' + str(i) + ' Specificity ' + str(Specificity[i]) + ' Sensitivity ' + str(Sensitivity[i]) + ' Accuracy ' + str(Accuracy[i]))

            fpr, tpr, thresholds= roc_curve(y_test[:,n], y_pred[:,n])

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label='Mean ROC Class %d (AUC = %0.2f $\pm$ %0.2f)' % (n, mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results, 'roc' + str(n) + '.png'))
        plt.close()
        print('Specificity ' + str(np.mean(Specificity)) + ' Sensitivity ' + str(np.mean(Sensitivity))+ ' Accuracy ' + str(np.mean(Accuracy)))

results    = '/data/breast/neo/results/vgg4block1dense30'
s8(3,results,5)
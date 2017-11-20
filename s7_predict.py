import keras
import numpy as np
import os

def s7(n_classes,results,kfold_splits):

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(kfold_splits):
        x_test = np.load(os.path.join(results,'x_test' + str(i) + '.npy'))
        y_test = np.load(os.path.join(results,'y_test' + str(i) + '.npy'))

        # use model to predict train/test, plot ROC curve
        model = keras.models.load_model(os.path.join(results,'model' + str(i) + '.h5'))
        a = model.predict(x_test)
        (a == a.max(axis=1, keepdims=True)).astype(int)
        np.savetxt(os.path.join(results,'yscore' + str(i) + '.csv'),a,delimiter=",")
        np.savetxt(os.path.join(results,'ytest' + str(i) + '.csv'),y_test,delimiter=",")
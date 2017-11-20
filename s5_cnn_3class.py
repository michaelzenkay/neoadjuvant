import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers
import pickle
from keras import backend as K


def s5(cuda,results,npydata,kfold_splits,batch_size,epochs):
    os.environ["CUDA_VISIBLE_DEVICES"]=cuda
    ogdata = np.load(npydata)
    num_classes = 3

    if os.path.isdir(results):
        print('already exists')
    else:
        os.mkdir(results)

    ### Shuffle and parse data into x and y
    np.random.shuffle(ogdata)
    X        = ogdata[:,:,:,1:]
    y        = ogdata[:,1,1,0]
    y=y-1

    ###KFOLDSPLITS - insert y, get test[] and train[] for different folds
    y = y[np.newaxis].T
    y = to_categorical(y, num_classes)
    train =[[] for i in range(kfold_splits)]
    test =[[] for i in range(kfold_splits)]

    # Go through each class
    for cla in range (num_classes):
        splitlist = []
        # Grab all examples from class and shuffle
        shuffledclass = np.squeeze(np.nonzero(y[:, cla]))
        np.random.shuffle(shuffledclass)
    #   print('class ' + str(cla) + ' ' + str(shuffledclass))

        #distribute shuffled examples of class evenly into each fold
        for fold in range(kfold_splits):
            startindex=round(fold*(shuffledclass.shape[0]/kfold_splits))
            endindex =round(((fold + 1) * shuffledclass.shape[0] / kfold_splits))
            splitlist.append(shuffledclass[startindex:endindex])
    #       print(shuffledclass[startindex:endindex])

        #Concatenate examples of class cla with previous examples
        for fold in range(kfold_splits):
            test[fold] = np.append(np.asarray(splitlist[fold]), test[fold])
            train[fold] = np.append(np.setdiff1d(shuffledclass, np.asarray(splitlist[fold])),train[fold])
    #       print('fold ' + str(fold) + str(np.asarray(splitlist[fold])))

    for kf in range(kfold_splits):

        print("Running fold", kf+1, "/", kfold_splits)

        ### Get indices of train/test data for current fold
        kf_train = train[kf].astype(int)
        kf_test  = test[kf].astype(int)
        x_train, x_test = X[kf_train], X[kf_test]
        y_train, y_test = y[kf_train], y[kf_test]

        ### Check that our classes are evenly distributed in training/testing
        print ('ytrain0 ' + str(sum(y_train[:, 0])))
        print ('ytrain1 ' + str(sum(y_train[:, 1])))
        print ('ytrain2 ' + str(sum(y_train[:, 2])))
        print ('ytest0 ' + str(sum(y_test[:, 0])))
        print ('ytest1 ' + str(sum(y_test[:, 1])))
        print ('ytest2 ' + str(sum(y_test[:, 2])))

        # Save test (x,y) data
        x_test_loc = os.path.join(results, 'x_test' + str(kf) + '.npy')
        x_train_loc = os.path.join(results, 'x_train' + str(kf) + '.npy')
        y_test_loc = os.path.join(results, 'y_test' + str(kf) + '.npy')
        y_train_loc = os.path.join(results, 'y_train' + str(kf) + '.npy')

        np.save(x_test_loc, x_test)
        #np.save(x_train_loc, x_train)
        np.save(y_test_loc, y_test)
        #np.save(y_train_loc, y_train)

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        data_augmentation = True
        model = Sequential()

        # Block 1
        model.add(Conv2D(64, 3, activation='relu', padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))
        model.add(Conv2D(64, 3, activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #        model.add(Dropout(0.25))

        # Block 2
        model.add(Conv2D(128, 3, activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, 3, activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #       model.add(Dropout(0.25))

        # Block 3
        model.add(Conv2D(256, 3, activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, 3, activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, 3, activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #      model.add(Dropout(0.25))

        # Block 4
        model.add(Conv2D(512, 3, activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, 3, activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, 3, activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #     model.add(Dropout(0.25))

        model.add(Flatten())
        #        model.add(Dense(4096, activation='relu'))
        #        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        ### Choice of optimizer
        adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        nadam = optimizers.nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0004)
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])

        ### Data Augmentation
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                      validation_data=(x_test, y_test), shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,
                shear_range=0.1,
                zoom_range=0.1)  # randomly flip images

        ###Run the model
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=epochs, validation_data=(x_test, y_test))

        ###Save Results

        # Dump history data into history(kf).pkl file
        with open(os.path.join(results, 'history' + str(kf) + '.pkl'), 'wb') as f:
            pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

        # Save Model
        model.save(os.path.join(results, 'model' + str(kf) + '.h5'))
        K.clear_session()
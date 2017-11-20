# neoadjuvant
s1 - sort
s2 - mask
s3 - slice
  Input 3D Nii
  Keep slices with ROI's larger than 'threshold' value
  Create a square patch around the ROI (matrix size is a factor of 32)
  scale square patch to 64x64
  Output 64x64 sized 2D NPY files 

s4 - load
  Define how many dynamics we want (max = 3 dynamics)
  Load and normalize 2D npy arrays of patients that have desired number of dynamics
  Save all 2D npy arrays into a large 4D npy array (n,x,y,dynamics)
  
s5 - CNN
  Run CNN with data augmentation and 5 fold cross validation
        # Block 1
        Conv2D(64 filters, 3x3 kernel, activation='relu', padding='same', name='block1_conv2')
        Conv2D(64 filters, 3x3 kernel, activation='relu', padding='same', name='block1_conv2')
        MaxPooling2D(pool_size=(2, 2)

        # Block 2
        Conv2D(128 filters, 3x3 kernel, activation='relu', padding='same', name='block2_conv1')
        Conv2D(128 filters, 3x3 kernel, activation='relu', padding='same', name='block2_conv2')
        MaxPooling2D(pool_size=(2, 2))

        # Block 3
        Conv2D(256 filters, 3x3 kernel, activation='relu', padding='same', name='block3_conv1')
        Conv2D(256 filters, 3x3 kernel, activation='relu', padding='same', name='block3_conv2')
        Conv2D(256 filters, 3x3 kernel, activation='relu', padding='same', name='block3_conv3')
        MaxPooling2D(pool_size=(2, 2)))

        # Block 4
        Conv2D(512 filters, 3x3 kernel, activation='relu', padding='same', name='block4_conv1')
        Conv2D(512 filters, 3x3 kernel, activation='relu', padding='same', name='block4_conv2')
        Conv2D(512 filters, 3x3 kernel, activation='relu', padding='same', name='block4_conv3')
        MaxPooling2D(pool_size=(2, 2)))

        Flatten()
        Dense(4096, activation='relu')
        Dropout(0.5)
        Dense(num_classes, activation='softmax')
s6 - Create Accuracy and Loss Figure
s7 - Generate Predictions
s8 - Generate ROC Curve


Visualize 0 layer feature maps
![features](/fig0.png?raw=true "1st Conv Layer")

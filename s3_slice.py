import nibabel as nib
import numpy as np
from math import ceil, floor
import glob
import os
from scipy.misc import imresize
import numpy
from shutil import copyfile
def s3(percent,mask,npypath,neofile):
    a = 0
    neo = np.loadtxt(neofile,delimiter=",")

    # Find masked files
    maskfiles = glob.glob(mask)
    for mask in maskfiles:

        # Parse Filenames for patient #, Oncotype DX Score, Dynamic #
        filenamein = os.path.basename(mask)
        pathname = os.path.dirname(mask)
        parsed = filenamein.split('_')
        PAT = parsed[0]
        DYN = parsed[2][0]
        pat = int(PAT)
        # Work on files that are not masks
        if DYN!='L' and pat in neo[:,0]:

            #Load Data
            data = nib.load(mask).get_data()

            #what is the neo score
            SCORE = np.where(neo[:,0]==pat)
            # Array->Scalar (flatten) ->int (get rid of .)
            SCORE = str(int(np.asscalar(neo[SCORE,1])))
            print(SCORE)
            # initialize x/y min/max variables
            xmin = np.empty(data.shape[2])
            xmax = np.empty(data.shape[2])
            xsiz = np.empty(data.shape[2])
            ymin = np.empty(data.shape[2])
            ymax = np.empty(data.shape[2])
            ysiz = np.empty(data.shape[2])

            # find mask start/end on each slice
            slices = []
            numslices = 0

            #Iterate through slices
            for i in range(0, data.shape[2]):

                # Find the masked data from slice
                data[data<0]=0
                maskd = data[:, :, i].nonzero()

                if len(maskd[0]) > 0 and len(maskd[1]) > 0 and np.count_nonzero(maskd) >  (percent / 100 * 16 * 16):
                    #print('found start/end on slice ' + str(i))
                    xmin[i] = maskd[0].min()
                    xmax[i] = maskd[0].max()
                    xsiz[i] = xmax[i] - xmin[i]
                    ymin[i] = maskd[1].min()
                    ymax[i] = maskd[1].max()
                    ysiz[i] = ymax[i] - ymin[i]
                    slices.insert(0, i)
                    numslices = numslices + 1
                else:
                    #print('no start/end on slice ' + str(i))
                    #plt.imshow(maskd)
                    xmin[i] = 0
                    xmax[i] = 0
                    xsiz[i] = 0
                    ymin[i] = 0
                    ymax[i] = 0
                    ysiz[i] = 0
                    #print('slice ' + str(i) + ' sucks')
            #print(mask + ' has ' + str(numslices) + ' slices found')

            try:
                xsmallest = xmin[xmin>0].min()
                xlargest = xmax[xmax <= data.shape[0]].max()
                longx = xlargest - xsmallest
                ysmallest = ymin[ymin>0].min()
                ylargest = ymax[ymax <= data.shape[1]].max()
                longy = ylargest - ysmallest

                # see if we should use 32x32, 64x64, 96x96 etc
                dim = 32.0 * ceil((max(longx, longy)) / 32.0)
                if dim ==32:
                    dim=64
                # Where should we crop in the volume
                xcenter = xsmallest + 0.5 * longx
                ycenter = ysmallest + 0.5 * longy
                xmincrop = int(floor(xcenter - 0.5 * dim))
                if xmincrop < 0:
                    xmincrop = 0
                xmaxcrop = int(xmincrop + dim)
                ymincrop = int(floor(ycenter - 0.5 * dim))
                if ymincrop < 0:
                    ymincrop = 0
                ymaxcrop = int(ymincrop + dim)

                for slice in slices:
                    # save name P_???_S_???_G_??_D_??
                    SLICE = str(int(slice)).zfill(3)
                    filenameout = 'P_' + PAT + '_S_' + SLICE + '_NEO_' + SCORE + '_D_' + DYN + '.npy'
                    output = os.path.join(npypath, filenameout)
                    img = data[xmincrop:xmaxcrop, ymincrop:ymaxcrop, slice]
                    # resize all to same 64x64
                    if img.shape[0] == 64:
                        scaled = img
                    else:
                        scaled = imresize(img, (64, 64), interp='bicubic')
                    # output to npy format array
                    numpy.save(output, scaled)
                os.rename(mask, os.path.join(npypath, filenamein))

            except:
                print('dyn ' + str(DYN) + ' did not work for ' + mask + ' calculating small/large has ' + str(np.count_nonzero(maskd[0])) + ' voxels')
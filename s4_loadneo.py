import numpy as np
import glob
import nibabel as nib
import os


def normslice(slice):
    ## CLIP top and bottom 5% values and scale rest of slice accordingly
    b, t = np.percentile(slice, (0.5, 99.5))
    slice = np.clip(slice, b, t)
    if np.std(slice) == 0:
        return slice
    else:
        return (slice - np.mean(slice)) / np.std(slice)

def s4(npypath ,savepath,D,npydata,percent):
    #Fileformat P_???_S_???_NEO_??_D_?.npy
    PAT = 1
    SLI = 3
    NEO = 5
    DYN = 7

    D01, D02, D03 = [], [], []

    ## Find all pats/slices that have dyn1
    if D==1 or D==2 or D==3:
        d1=glob.glob(npypath + 'P_*D_1.npy')
        for name in d1:
            patnum = str(os.path.basename(name).split('_')[PAT])
            slinum = str(os.path.basename(name).split('_')[SLI])
            D01.insert(0, patnum + '_' + slinum)
        alldyn = D01
        cha = 1
        n = alldyn.__len__()

    ## Find all pats/slices that have dyn1+dyn2
    if D==2 or D==3:
        d2=glob.glob(npypath + 'P_*D_2.npy')
        for name in d2:
            patnum = str(os.path.basename(name).split('_')[PAT])
            slinum = str(os.path.basename(name).split('_')[SLI])
            D02.insert(0, patnum + '_' + slinum)
        alldyn = np.intersect1d(D01, D02)
        cha = 2
        n = alldyn.__len__() * 2

    ## Find all pats/slices that have dyn1+dyn2+dyn3
    if D==3:
        d3=glob.glob(npypath + 'P_*D_3.npy')
        for name in d3:
            patnum = str(os.path.basename(name).split('_')[PAT])
            slinum = str(os.path.basename(name).split('_')[SLI])
            D01.insert(0, patnum + '_' + slinum)
        alldyn = np.intersect1d(alldyn,D03)
        cha = 3
        n = alldyn.__len__() * 3

    #Initialize empty variable data
    row = np.load(d1[0]).shape[0]
    col = np.load(d1[0]).shape[1]
    data = np.empty(shape=(n, row,col,cha+1)) #cha+1 for label

    ###Load numpy array as [n,x,y,c] c[0]-DX, c[1]-d1 c[2]-d2, c[3]=d3
    n=0
    slicestats = np.zeros((alldyn.__len__(),2))
    for p in alldyn:
        patnum = p.split('_')[0]
        slinum = p.split('_')[1]
        slicestats[n, 0] = patnum
        slicestats[n, 1] = slinum
        fnd1 = glob.glob(npypath + 'P_' + patnum + '_S_' + slinum + '*D_1.npy')
        fnd2 = glob.glob(npypath + 'P_' + patnum + '_S_' + slinum + '*D_2.npy')
        fnd3 = glob.glob(npypath + 'P_' + patnum + '_S_' + slinum + '*D_3.npy')
        try:
            #Load neoscore
            neoscore = int(os.path.basename(fnd1[0]).split('_')[NEO])
            data[n, 1, 1, 0] = neoscore

        except ValueError:
            print(p + ' did not work')

        try:
            # Load 1st dynamic
            if (D==1 or D==2 or D==3):
                xtemp = np.load(fnd1[0])
                data[n, :, :, 1] = normslice(xtemp)
        except:
            try:
                if (D == 1 or D == 2 or D == 3):
                    xtemp = np.load(fnd1[0])
                    data[n, :, :, 1] = np.squeeze(normslice(xtemp))
            except:
                print(p + ' loaddyn1 did not work')
        try:
            # Load 2nd dynamic
            if (D==2 or D==3):
                xtemp = np.load(fnd2[0])
                data[n, :, :, 2] = normslice(xtemp)
        except:
            try:
                if (D==2 or D==3):
                    xtemp = np.load(fnd2[0])
                    data[n, :, :, 2] = np.squeeze(normslice(xtemp))
            except:
                print(p + ' loaddyn2 did not work')

            # Load 3rd dynamic
        try:
            if D==3:
                xtemp = np.load(fnd3[0])
                data[n, :, :, 3] = normslice(xtemp)


        except:
            try:
                if D == 3:
                    xtemp = np.load(fnd3[0])
                    data[n, :, :, 3] = np.squeeze(normslice(xtemp))
            except:
                print(p + ' loaddyn3 did not work')
        n=n+1

    np.save(npydata,data)

    unique, unique_counts= np.unique(slicestats[:,0],return_index=False,return_inverse=False,return_counts=True,axis=0)
    slicestats = np.zeros([unique.__len__(),2])
    slicestats[:, 0] = unique
    slicestats[:, 1] = unique_counts


    np.savetxt(os.path.join(savepath,'slicestats' + str(percent) + '.csv'),slicestats,delimiter=',')


savepath  = '/data/breast/neo/testresult/'
npypath   = '/data/breast/neo/sliced/sliced30/'
npydata = '/data/breast/neo/testresult/NEO30.npy'
mask      = '/data/breast/neo/masked/*1.nii.gz'
percent      = 30
numdyn       = 1
batch_size   = 400
epochs       = 1000
kfold_splits = 5
neofile    = '/data/breast/neo/neotest.csv'
s4(npypath ,savepath,numdyn,npydata,percent)
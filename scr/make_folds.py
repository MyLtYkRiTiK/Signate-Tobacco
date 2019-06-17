import random
import pandas as pd
import os

def folds():
    if not os.path.exists(os.path.dirname('output/preprocessing/folds/')):
        os.makedirs(os.path.dirname('output/preprocessing/folds/'))
    classes = os.listdir('output/preprocessing/croped_images/')

    random.seed(777)
    n = 10
    fold_dict = {}
    val_dict = {}
    for fold in range(n):
        fold_dict['fold_{}'.format(fold)] = []
        val_dict['fold_{}'.format(fold)] = []
    for clas in classes:
        l = os.listdir('output/preprocessing/croped_images/'+clas+'/')
        random.shuffle(l)
        l = [l[x::n] for x in range(n)]
        for fold in range(n):
            val_dict['fold_{}'.format(fold)].extend([(clas+'/'+x, str(clas)) for x in l[fold]])
            fold_dict['fold_{}'.format(fold)].extend([(clas+'/'+x, str(clas)) for y in range(n) 
                                                      for x in l[y]  if y!=fold])
            random.shuffle(val_dict['fold_{}'.format(fold)])
            random.shuffle(fold_dict['fold_{}'.format(fold)])

    for fold in range(n):
        pd.DataFrame(fold_dict['fold_{}'.format(fold)], 
                     columns = ['name', 'label']).to_csv('output/preprocessing/folds/fold_{}.csv'.format(fold), index=False)
        pd.DataFrame(val_dict['fold_{}'.format(fold)], 
                     columns = ['name', 'label']).to_csv('output/preprocessing/folds/val_{}.csv'.format(fold), index=False)
    print('folds done')
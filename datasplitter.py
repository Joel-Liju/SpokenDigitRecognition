import glob
import os
from tqdm import tqdm
import shutil
dataFolder = 'dataset'
fNames = glob.glob(dataFolder+'/*.png')
for name in tqdm(fNames):
    # print(name)
    directory,file = name.split('\\')
    if not os.path.isdir(directory+'/'+file[0]):
        os.mkdir(directory+'/'+file[0])
    shutil.move(name,directory+'/'+file[0]+'/'+file)
print("done first part")
fNames = glob.glob(dataFolder+'/*.wav')
for name in fNames:
    os.remove(name)
print("done second portion")

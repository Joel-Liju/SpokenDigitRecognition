import os
import random
import shutil
directoryPath = "recordings"
trainsize = .8
files = os.listdir(directoryPath)
random.shuffle(files)

traindata = files[:int(trainsize*len(files))]
testdata = files[int(trainsize*len(files)):]

def restartDirectory(directorypath):
    if os.path.exists(directorypath):
        for f in os.listdir(directorypath):
            os.remove(os.path.join(directorypath,f))
    else:
        os.mkdir(directorypath)

restartDirectory("traindata")
for data in traindata:
    shutil.copy(directoryPath+'/'+data,"traindata/"+data)
print("done copying traindata")
restartDirectory("testdata")
for data in traindata:
    shutil.copy(directoryPath+'/'+data,"testdata/"+data)
print("done copying testdata")
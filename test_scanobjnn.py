'''# 指定文件夹路径
root = '../dataset/scanobjectnn/projection/test/'

# 列出文件夹中所有文件
files = os.listdir(root)
classnames = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]

for prefix in classnames:
    Kshot = [file for file in files if file.startswith(prefix)][:16]
    print(Kshot)
    for i in Kshot:
        match = re.search(r'\d+', i)
        if match:
            # 打印匹配到的数字
            num = match.group()
            print(num)
        else:
            print("No numbers found in filename.")
'''

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import h5py
import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader

class triDataset(Dataset): 
    def __init__(self,datapath,imgpath, transform=None,target_transform=None,small_datasets = None):
        super(triDataset,self).__init__()
        classnames = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]

        f = h5py.File(datapath, "r")
        data = f['/data'][:]
        #print(data[0].shape)
        label = f['/label'][:]
        self.label = label
        self.data = data

        img = []
        files = os.listdir(imgpath)
        if(small_datasets!=None):
            for prefix in classnames:
                Kshot = [file for file in files if file.startswith(prefix)][:small_datasets]
            #print(Kshot)
                img.extend(Kshot)
        else:
            img = files
        
        self.imgpath = imgpath
        self.img = img
        self.transform = transform
        self.target_tra·wnsform = target_transform
             
    def __getitem__(self, index):
        img_path = self.img[index]
        match = re.search(r'\d+', img_path)
        num = int(match.group())
        label = self.label[num]

        points = self.data[num]
        color = np.full((points.shape[0], 3), 0.5)

        points = np.hstack((points, color))

        img = Image.open(self.imgpath+img_path).convert('RGB')

        transf = transforms.ToTensor()
        img_tensor = transf(img)
        return (img_tensor, torch.as_tensor(points)),label

    def __len__(self):  
	    return len(self.img)


#train_set=triDataset(datapath = '../dataset/scanobjectnn/main_split_nobg/training_objectdataset_augmentedrot_scale75.h5',imgpath = '../dataset/scanobjectnn/projection/train/',small_datasets = 16)#,small_datasets = 10 #,small_datasets = 10
#test_set = triDataset(datapath = '../dataset/scanobjectnn/main_split_nobg/test_objectdataset_augmentedrot_scale75.h5',imgpath = '../dataset/scanobjectnn/projection/test/')

#print(train_set.__getitem__(0))

train_set=triDataset(datapath = '../dataset/scanobjectnn/main_split_nobg/training_objectdataset_augmentedrot_scale75.h5',imgpath = '../dataset/scanobjectnn/projection/train/',small_datasets = 16)#,small_datasets = 10 #,small_datasets = 10
test_set = triDataset(datapath = '../dataset/scanobjectnn/main_split_nobg/test_objectdataset_augmentedrot_scale75.h5',imgpath = '../dataset/scanobjectnn/projection/test/')
trainData = DataLoader(
    train_set,
    batch_size=20,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
)
testData = DataLoader(
    test_set,
    batch_size=40 ,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
)
classnames = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]

i = 0
for data_iter, inputs in enumerate(trainData):
    img = inputs[0][0]
    pc = inputs[0][1]
    label = inputs[1]
    print(img,pc,label)

    i+=1
    if(i ==3):
        break
'''import numpy as np

points = np.zeros((20, 3))
print(points)
extra_columns = np.full((points.shape[0], 3), 255)

# 在 points 数组的列方向上拓宽三列，值设为 255
points = np.hstack((points, extra_columns))
print(points)'''
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
def default_loader(fn):
    points = []
    with open(fn[1], 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y, z, nx, ny, nz = map(float, line.strip().split(','))
            points.append([x, y, z,0.5,0.5,0.5]) #,0.5,0.5,0.5clour but modelnet40 don't have it so they are grey
        '''pointCloud = np.array(points)
        pointCloud = pointCloud.reshape(1,10000,3)

        pointCloud = torch.tensor(pointCloud).float()
        pointCloud_1000 = utils.farthest_point_sample(pointCloud,1000).unsqueeze(0)
        points = torch.concat((pointCloud_1000[0],torch.full((1, 1000, 3), 0.5)),axis = 2)[0].tolist()'''
    return (Image.open(fn[0]).convert('RGB'),points)


class triDataset(Dataset): 
    def __init__(self,txt, transform=None,target_transform=None, loader=default_loader, small_datasets = None):
        super(triDataset,self).__init__()
        fh = open(txt, 'r')
        pairs = []
        classes = []
        i=0
        label = {}
        for line in (open('../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_shape_names.txt', 'r')):
            line = line.strip('\n')
            line = line.rstrip('\n')
            classes.append(line)
            label[line] = i
            i = i + 1
        for line in fh: 
            line = line.strip('\n')
            line = line.rstrip('\n')
            if(small_datasets!=None):
                if(int(line[-2:])>small_datasets or line[-4:-2] != '00'):
                    continue  #10
            #print(line)
            for i in range(1): #10
                pairs.append((('../dataset/modelnet40/modelnet40_projection/'+line[:-5]+'/'+line+'_'+str(i)+'.png', '../dataset/modelnet40/modelnet40_normal_resampled/'+line[:-5]+'/'+line+'.txt'), label[line[:-5]]))
        self.pairs = pairs
        self.transform = transform
        self.target_tra·wnsform = target_transform
        self.loader = loader  
        self.classes = classes     
    def __getitem__(self, index):
        fn, label = self.pairs[index]
        pair = self.loader(fn) 
        transf = transforms.ToTensor()
        img_tensor = transf(pair[0])
        return (img_tensor, torch.as_tensor(pair[1])),label
        #return (pair[0], torch.as_tensor(pair[1])),label
    def __len__(self):  
	    return len(self.pairs)
    
train_set=triDataset(txt='../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_train.txt',small_datasets = 16)#,small_datasets = 10 #,small_datasets = 10
#print(train_set.__getitem__(1))
trainData = DataLoader(
        train_set,
        batch_size=20,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )
i = 0
for data_iter, inputs in enumerate(trainData):
    print(inputs[1])
    i += 1

print(i)
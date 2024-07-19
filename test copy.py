'''from utils import open_clip
import torch
from utils.tokenizer import SimpleTokenizer
from memory_profiler import profile
clip_model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained="./checkpoints/clip_model.bin")
print("load finished")
text_encoder = clip_model.text
visual_encoder = clip_model.visual

tokenizer = SimpleTokenizer()


current_gpu_index = torch.cuda.current_device()

# 获取当前GPU的名称
current_gpu_name = torch.cuda.get_device_name(current_gpu_index)

# 获取GPU显存的总量和已使用量
total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
free_memory = total_memory - used_memory  # 剩余显存(GB)
print(f"GPU显存总量：{total_memory:.2f} GB")
print(f"已使用的GPU显存：{used_memory:.2f} GB")
print(f"剩余GPU显存：{free_memory:.2f} GB")

print("load to gpu")
device_point_encoder = torch.device("cuda:0")
text_encoder.to(device_point_encoder)
print("so far ok")
classes = ['cup', 'not_cup']

total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
free_memory = total_memory - used_memory  # 剩余显存(GB)
print(f"GPU显存总量：{total_memory:.2f} GB")
print(f"已使用的GPU显存：{used_memory:.2f} GB")
print(f"剩余GPU显存：{free_memory:.2f} GB")

print("compute outputs")
text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in classes]).to(device_point_encoder) 
print("text_inputs",text_inputs)
print(text_encoder(text_inputs))

total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
free_memory = total_memory - used_memory  # 剩余显存(GB)
print(f"GPU显存总量：{total_memory:.2f} GB")
print(f"已使用的GPU显存：{used_memory:.2f} GB")
print(f"剩余GPU显存：{free_memory:.2f} GB")'''


'''# 打开文件
with open("checkpoints.txt", "r") as file:
    # 读取每一行
    lines = file.readlines()

# 解析每一行
checkpoint_data = {}
for line in lines:
    # 分割每一行的键值对
    key, value = line.strip().split(": ")
    # 存储到字典中
    checkpoint_data[key] = float(value)

# 打印解析后的数据
print("Current Epoch:", int(checkpoint_data["current epoch"]))
print("Accuracy:", checkpoint_data["acc"])
print("Best Accuracy:", checkpoint_data["best acc"])'''
'''import subprocess

# 执行系统命令squeue
result = subprocess.run(["squeue"], capture_output=True, text=True)

# 检查返回结果中是否包含multi_prompt字段
if "multi_prompt" in result.stdout:
    print("返回结果中包含multi_prompt字段")
else:
    print("返回结果中不包含multi_prompt字段")'''

'''import pickle
path='../dataset/modelnet40/ModelNetFewshot/5way_10shot/0.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径
	   
f=open(path,'rb')
data=pickle.load(f)
 
print(data['test'][1][0])
print(len(data['test'][1][0]))
'''

'''import os
import numpy as np
import warnings
import pickle

from torch.utils.data import Dataset , DataLoader
import torch


warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetFewShot(Dataset):
    def __init__(self):
        self.root = '../dataset/modelnet40/ModelNetFewshot/'
        self.npoints = 8192
        self.use_normals = False
        self.num_category = 40
        self.process_data = True
        self.uniform = True
        #split = config.subset
        self.subset = 'train'
        self.way = 5
        self.shot = 10
        self.fold = 8
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')


        print('Load processed data from %s...' % self.pickle_path)

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print('The size of data is ' + str(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]

        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'test':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)
test = ModelNetFewShot()
trainData = DataLoader(
        test,
        batch_size=50,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )
i = 0
for data_iter, inputs in enumerate(trainData):
    print(inputs[2])
    print("gg")
    break'''
'''import random
from collections import defaultdict
import os
import h5py
import numpy as np
from collections import OrderedDict
def split_dataset_by_label(data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

def generate_fewshot_dataset(
         *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                      sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=0, classname=''):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            classname = line.strip()
            classnames[i] = classname
    return classnames

def read_data(classnames, datas, labels):
    items = []

    for i, data in enumerate(datas):
        label = int(labels[i])
        classname = classnames[label]
#from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
        item = Datum(
            impath=data,
            label=label,
            classname=classname
        )
        items.append(item)
    
    return items
def load_data(data_path):
    
    all_data = []
    all_label = []
    with open(data_path, "r") as f:
        for h5_name in f.readlines():
            f = h5py.File("../dataset/modelnet40/modelnet40_normal_resampled/"+h5_name.strip()[:-5]+'/'+h5_name.strip()+'.txt', 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label
dataset_dir = "../dataset/modelnet40/modelnet40_normal_resampled/"
train_data, train_label = load_data(os.path.join(dataset_dir, 'modelnet40_train.txt'))
test_data, test_label = load_data(os.path.join(dataset_dir, 'modelnet40_test.txt'))
text_file = os.path.join(dataset_dir, 'shape_names.txt')
classnames = read_classnames(text_file)
train = read_data(classnames, train_data, train_label)
test = read_data(classnames, test_data, test_label )

num_shots = 10
train = generate_fewshot_dataset(train, num_shots=num_shots)
print(train)'''

import copy
from thop import profile
import torch.nn as nn
from utils.tokenizer import SimpleTokenizer
import torch
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class multiPromptLearner(nn.Module):
    def __init__(self, classnames ):
        tokenizer = SimpleTokenizer()
        super().__init__()
        # text ptompt init
        n_ctx = 3  
        ctx_dim = 1280 
        # and here is 1280
        img_ctx =1  
        img_dim = 1024 #1792
        self.compound_prompts_depth = 0 #9 

        ##shallow prompt
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors) 
        self.text_pc_proj = nn.Linear(ctx_dim, 1408) 

        #### deep prompt
        '''self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])'''
        ''' for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        single_layer = nn.Linear(ctx_dim, 1408)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)'''
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.stack([tokenizer(p) for p in prompts]) #cat
        self.tokenized_prompts = tokenized_prompts

        #visual_prompt = torch.empty(img_ctx,img_dim)
        #nn.init.normal_(visual_prompt, std=0.02)
        #self.visual_prompt = nn.Parameter(visual_prompt)
        self.img_pc_proj = nn.Linear(img_dim, 1408) 

         
    def forward(self,image_embeddings=None):
        batch_size = 1
        ctx = self.ctx
        tokenized_prompts = self.tokenized_prompts
        #visual_prompt = self.visual_prompt
        #point_prompt = torch.cat((self.text_pc_proj(ctx),self.img_pc_proj(visual_prompt)),axis = 0)
        visual_prompt = image_embeddings
        text_to_pc = self.text_pc_proj(ctx).repeat(image_embeddings.shape[0],1,1)
        img_to_pc = self.img_pc_proj(visual_prompt).unsqueeze(1)
        point_prompt =  torch.cat((text_to_pc,img_to_pc),axis = 1) 
        print("point_prompt",point_prompt.shape)
        text_to_pc = self.text_pc_proj(ctx).repeat(batch_size,1,1)
        point_prompt = text_to_pc
        
        ###point_prompt = self.text_pc_proj(ctx)
                                  
        '''point_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            point_deep_prompts.append(layer(self.compound_prompts_text[index]))'''

        ###return ctx,tokenized_prompts,point_prompt,self.compound_prompts_text,point_deep_prompts
        #return ctx,tokenized_prompts,visual_prompt,point_prompt,self.compound_prompts_text,point_deep_prompts
        return ctx,tokenized_prompts,point_prompt

classnames = []
i = 0
for line in (open('../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_shape_names.txt', 'r')):
    line = line.strip('\n')
    line = line.rstrip('\n')
    classnames.append(line)
    i += 1
    if(i == 5):
        break
print(len(classnames))
model = multiPromptLearner(classnames)
input = torch.randn(1,1,1024)
print(input)
_,params = profile(model,inputs=(input))
print(params)
#print(list(model.parameters()))
params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))
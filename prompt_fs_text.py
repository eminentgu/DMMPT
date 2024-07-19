from utils import open_clip
import torch

import models.uni3d as models

from utils import utils
from models.multiPrompt import multiPrompt
from utils.params import parse_args
from PIL import Image
import sys

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
from torch.cuda.amp import autocast as autocast

from memory_profiler import profile
import numpy as np
import os
import pickle

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetFewShot(Dataset):
    def __init__(self,locate):
        self.root = '../dataset/modelnet40/ModelNetFewshot/'
        self.npoints = 8192
        self.use_normals = False
        self.num_category = 40
        self.process_data = True
        self.uniform = True
        #split = config.subset
        self.subset = locate['subset']
        self.way = locate['way']
        self.shot = locate['shot']
        self.fold = locate['fold']
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
        points, label, real_label = self.dataset[index]
        #print(_)
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]
        extra_columns = np.full((points.shape[0], 3), 0.5)
        points = np.hstack((points, extra_columns))

        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'test':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label, real_label

#@profile
def main(args):
    evaluate = True##False
    evaluate_acc = []
    #-----------------------------------------------------------------
    print("load teacher model and pretrain 3D encoder")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained="./checkpoints/clip_model.bin")
    #clip_model.to('cpu') #device
    clip_model.eval()

    device_point_encoder = torch.device("cuda:0")
    args, ds_init = parse_args(args)
    model = getattr(models, 'create_uni3d')(args = args)
    model.to(device_point_encoder)
    ckpt_path = "./checkpoints/model.pt"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    sd = checkpoint['module']
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print("load teacher model and pretrain 3D encoder successfully")
    #------------------------------------------------------------------

    print("load dataset")
    classnames = []
    for line in (open('../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_shape_names.txt', 'r')):
        line = line.strip('\n')
        line = line.rstrip('\n')
        classnames.append(line)
    way = 5
    shot = 10
    for fold in range(0,10):
        torch.cuda.empty_cache()
        locate = {"subset":'train',
            "way":way,
            "shot":shot,
            "fold":fold
            }
        train_set=ModelNetFewShot(locate)
        locate = {"subset":'test',
            "way":way,
            "shot":shot,
            "fold":fold
            }
        test_set =ModelNetFewShot(locate)
        classes = {}
        for _,label,real_label in train_set:
            if not (label in classes):
                classes[label] = real_label
            if(len(classes) == way):
                break
        #print("at fold",fold," ",classes)
        
        current_class = []
        for i in range(way):
            current_class.append(classnames[classes[i]])
        print("at fold",fold," current few shot classes are",current_class)
        trainData = DataLoader(
            train_set,
            batch_size=20,
            shuffle=True,
            pin_memory=False,
            num_workers=1,
        )
        testData = DataLoader(
            test_set,
            batch_size=20,
            shuffle=True,
            pin_memory=False,
            num_workers=1,
        )

        print("ptraining fold",fold)
        acc = 0
        best_acc = 0
        multi_prompt = multiPrompt(clip_model,model, classnames) #args
        #multi_prompt.to(device_prompt)
        #multi_prompt.to('cpu')#device
        #print("so far all good")
        #text_prompt =  multi_prompt.create_text_prompt()
        #visual_prompt = multi_prompt.create_visual_prompt() 


        if evaluate:
            print("begin evaluate")
            multi_prompt = multiPrompt(clip_model,model, classnames)

            savepath = str(way)+'way_'+str(shot)+'shot_'+str(fold)+'mn40_text_pc_latest.pth'
            multi_prompt.load_state_dict(torch.load(savepath), strict=False)

            #multi_prompt = torch.load('img_text_pc_prompt_mn40.pth')
            #multi_prompt.to(device_prompt)
            test_stats = evaluation(testData,multi_prompt)
            print("test on ",fold," acc: ",test_stats['acc'])
            evaluate_acc.append(test_stats['acc'])
            continue
        
        
        
        multi_prompt.prompt.cuda() 
        epoch = 50#50
        lr = 0.01
        name_to_update = "prompt"
        for name, param in multi_prompt.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        # Double check and init optism 
        enabled = set()
        p_wd, p_non_wd = [], []
        for name, param in multi_prompt.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                if param.ndim < 2 or 'bias' in name or 'ln' in name or 'bn' in name:
                    p_non_wd.append(param)
                else:
                    p_wd.append(param)
        print(f"Parameters to be updated: {enabled}")
        optim_params = [{"params": p_wd, "weight_decay": 0.01}, #args.wd =0.1
                        {"params": p_non_wd, "weight_decay": 0}]
        optimizer = torch.optim.AdamW(optim_params, lr=lr, betas=(0.9, 0.999), #args.betas
                                        eps=1e-8, weight_decay=0.01)
        scaler = amp.GradScaler(enabled=False)
        lr_schedule = utils.cosine_scheduler(lr,1e-5 , 50,len(train_set) // 1, warmup_epochs=1, start_warmup_value=1e-6)

        save_state = {}
        #print("Model's state_dict:")
        for param_tensor in multi_prompt.state_dict():
            if 'prompt' in param_tensor:
                save_state.update({param_tensor:multi_prompt.state_dict()[param_tensor]})
                print(param_tensor, "\t",multi_prompt.state_dict()[param_tensor].size())


        #for e in range(epoch):
        for e in range(epoch):
            with open('logs.txt', 'a') as f:
                f.write("\nbegin epoch: "+str(e))
            train_stats = train(trainData, multi_prompt, optimizer, scaler, epoch, lr_schedule, args)
            acc = train_stats['acc']
            print("\nat epoch",e,":",train_stats," best_acc", best_acc)
            if (e%1 == 0):
                print("saving current epoch,acc:",acc,"epoch", e)
                savepath = str(way)+'way_'+str(shot)+'shot_'+str(fold)+'mn40_text_pc_latest.pth'
                torch.save(save_state, savepath)
            if (acc > best_acc):
                best_acc = acc
                print("saving best checkpoint,best acc:",best_acc,"epoch", e)
                savepath = str(way)+'way_'+str(shot)+'shot_'+str(fold)+'mn40_text_pc_best.pth'
                torch.save(save_state, savepath)
            #torch.cuda.empty_cache()
    if evaluate:
        print("after ten fold evaluation,the accuracy is :",np.mean(evaluate_acc)," each is ",evaluate_acc)
#@profile
def train(trainData,model , optimizer, scaler, epoch, lr_schedule, args):
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    #print(f"Parameters to be updated: {enabled}")
    iters_per_epoch = len(trainData) // args.update_freq
    # switch to train mode
    model.train()
    acc = 0
    total = 0
    current = 0
    for data_iter, inputs in enumerate(trainData):
        optim_iter = data_iter // 1 #args.update_freq
        it = iters_per_epoch * epoch + optim_iter
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        with autocast():
            '''img = inputs[0][0].float()
            pc = inputs[0][1].float().cuda(0)
            label = inputs[1].cuda()'''
            pc = inputs[0].cuda(0)
            label = inputs[2].type(torch.int64).cuda(0)
            outputs = model(pc,label = label)
        loss = outputs['loss']
        loss /= args.update_freq
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #model.zero_grad(set_to_none=True)
        acc += outputs['acc']
        total += outputs['total']
        current +=1
        loss = loss.item()
        with open('logs.txt', 'a') as f:
            f.write("\ncurrent progress:"+str(current)+"/"+str(iters_per_epoch)+"loss: "+str(loss)+" acc:"+str(acc/total))
    return {'acc':acc/total}

def evaluation(testData,model):
    acc = 0
    total = 0
    current = 0
    iters_per_epoch = len(testData)
    with torch.no_grad():
        for data_iter, inputs in enumerate(testData):
            pc = inputs[0].cuda(0)
            label = inputs[2].type(torch.int64).cuda(0)
            with autocast():
                outputs = model(pc,label = label)
            acc += outputs['acc']
            total += outputs['total']
            current +=1
            with open('logs.txt', 'a') as f:
                f.write("\ncurrent progress:"+str(current)+"/"+str(iters_per_epoch)+" acc:"+str(acc/total))
    return {'acc':acc/total}

if __name__ == '__main__':
    main(sys.argv[1:])
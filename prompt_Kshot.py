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

#@profile
def main(args):
    evaluate =  True##False
    print("load teacher model and pretrain 3D encoder")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained="./checkpoints/clip_model.bin",device = torch.device("cuda:0"))
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
    for shot in [4,8,16]:
        print("load dataset,shot:",shot)
        train_set=triDataset(txt='../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_train.txt',small_datasets = shot)#,small_datasets = 10 #,small_datasets = 10
        test_set = triDataset(txt='../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_test.txt')
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
        classnames = []
        for line in (open('../dataset/modelnet40/modelnet40_normal_resampled/modelnet40_shape_names.txt', 'r')):
            line = line.strip('\n')
            line = line.rstrip('\n')
            classnames.append(line)
    
        torch.cuda.empty_cache()
        text_prompt_size = 3
        best_acc = 0
        multi_prompt = multiPrompt(clip_model,model, classnames, text_prompt_size) #args
        print("start at 0 epoch")
        #multi_prompt.to(device_prompt)
        #multi_prompt.to('cpu')#device
        #print("so far all good")
        #text_prompt =  multi_prompt.create_text_prompt()
        #visual_prompt = multi_prompt.create_visual_prompt()  
        if evaluate:
            print("begin evaluate")
            multi_prompt = multiPrompt(clip_model,model, classnames, text_prompt_size)
            multi_prompt.load_state_dict(torch.load('text_pc_prompt_mn40_best_acc_'+str(shot)+'shot.pth'), strict=False)

            #multi_prompt = torch.load('img_text_pc_prompt_mn40.pth')
            #multi_prompt.to(device_prompt)
            test_stats = evaluation(testData,multi_prompt)
            print("test on modelnet40","text_prompt_size:",shot,"acc: ",test_stats['acc'])
            continue
        multi_prompt.prompt.cuda()
        print("prepare for training")
        epoch = 50#50
        lr = 0.001 
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
        print("ALL GOOD,LFG")


        save_state = {}
        print("Model's state_dict:")
        for param_tensor in multi_prompt.state_dict():
            if 'prompt' in param_tensor:
                save_state.update({param_tensor:multi_prompt.state_dict()[param_tensor]})
                print(param_tensor, "\t",multi_prompt.state_dict()[param_tensor].size())


        for e in range(0,epoch):
            with open('logs.txt', 'a') as f:
                f.write("\nbegin epoch: "+str(e))
            train_stats = train(trainData, multi_prompt, optimizer, scaler, epoch, lr_schedule, args)
            acc = train_stats['acc']
            print("\nat epoch",e,":",train_stats," best_acc", best_acc)
            if (e%1 == 0):
                print("saving current epoch,acc:",acc,"epoch", e)
                torch.save(save_state, 'text_pc_prompt_mn40_latest_'+str(shot)+'shot.pth')
            if (acc > best_acc):
                best_acc = acc
                print("saving best checkpoint,best acc:",best_acc,"epoch", e)
                torch.save(save_state, 'text_pc_prompt_mn40_best_acc_'+str(shot)+'shot.pth')
                #torch.save(multi_prompt,'img_text_pc_prompt_mn40.pth')
            torch.cuda.empty_cache()

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
            img = inputs[0][0].to(torch.float16).cuda(0)
            pc = inputs[0][1].float().cuda(0)
            label = inputs[1].cuda()
            outputs = model(pc,img=img,label=label)
        loss = outputs['loss']
        loss /= args.update_freq
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)
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
            img = inputs[0][0].float().cuda(0)
            pc = inputs[0][1].float().cuda(0)
            label = inputs[1].cuda(0)
            with autocast():
                outputs = model(pc,img,label)
            acc += outputs['acc']
            total += outputs['total']
            current +=1
            with open('logs.txt', 'a') as f:
                f.write("\ncurrent progress:"+str(current)+"/"+str(iters_per_epoch)+" acc:"+str(acc/total))
    return {'acc':acc/total}

if __name__ == '__main__':
    main(sys.argv[1:])
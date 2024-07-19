import torch
import torch.nn as nn
from utils.tokenizer import SimpleTokenizer
from torch.nn import functional as F
import numpy as np
import copy
from torch.cuda.amp import autocast as autocast
'''clip_model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained="./clip_model/clip_model.bin")
clip_model.to('cpu') #de 
vice'''


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class multiPromptLearner(nn.Module):
    def __init__(self, classnames,text_prompt_size = 3):
        tokenizer = SimpleTokenizer()
        super().__init__()
        # text ptompt init
        n_ctx = text_prompt_size#text_prompt_size 
        ctx_dim = 1280 
        # and here is 1280
        img_ctx =1  
        img_dim = 1024 #1792

        ##shallow prompt
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.05) #0.02
        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors) 
        self.text_pc_proj = nn.Linear(ctx_dim, 1408) 
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        #print(prompts)
        tokenized_prompts = torch.stack([tokenizer(p) for p in prompts]) #cat
        '''print("-------------------------------------------------------------------------------")
        print(tokenized_prompts)
        print("-------------------------------------------------------------------------------")'''
        self.tokenized_prompts = tokenized_prompts 
        self.img_pc_proj = nn.Linear(img_dim, 1408) 

         
    def forward(self,image_embeddings=None):
        ctx = self.ctx
        tokenized_prompts = self.tokenized_prompts
        visual_prompt = image_embeddings
        text_to_pc = self.text_pc_proj(ctx).repeat(image_embeddings.shape[0],1,1)
        img_to_pc = self.img_pc_proj(visual_prompt).unsqueeze(1) #.cuda()
        point_prompt =  torch.cat((text_to_pc,img_to_pc),axis = 1) 
        return ctx,tokenized_prompts,point_prompt

class multiPrompt(nn.Module):
    def __init__(self,clip_model,pc_model, classnames,text_prompt_size = 3) :
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prompt =  multiPromptLearner(classnames,text_prompt_size).cuda()#,clip_model
        self.pc_model = pc_model
        self.classnames = classnames
        self.clip_model = clip_model
        tokenizer = SimpleTokenizer()
        self.tokenizer = tokenizer
    def encode_pc(self, pc,point_prompt):
        pc_feat = self.pc_model.encode_pc(pc, point_prompt)
        #pc_feat = self.pc_model.encode_pc(pc)
        return pc_feat
    def encoder_img(self,img):
        '''
        img_feat = self.clip_model.encode_image(img[0].to(torch.float16).cuda(0))
        for i in range(1,10):
            img_feat = (img_feat + self.clip_model.encode_image(img[i].to(torch.float16).cuda(0)))
        img_feat = img_feat/10'''
        img_feat = self.clip_model.encode_image(img)  
        return img_feat
    def encoder_text(self,ctx,tokenized_prompts):
        text = self.tokenizer(self.classnames)
        '''print("----------text = self.tokenizer(self.classnames)---------------------------")
        print(text)
        print("---------------------------------------------------------------------------")'''
        return self.clip_model.encode_text(text.cuda(),ctx.cuda(),tokenized_prompts.cuda())#.to("cpu").to("cpu").to("cpu")

    def forward(self,pc,img = None,label = None): #prompt_learner
        with torch.no_grad():
            image_features = self.encoder_img(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        ctx,tokenized_prompts,point_prompt = self.prompt(image_features)
        text_features = self.encoder_text(ctx,tokenized_prompts)
        point_features = self.encode_pc(pc,point_prompt)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        point_features = point_features / point_features.norm(dim=-1, keepdim=True)

        #print(text_features.shape,text_features)
        '''print(point_features)
        print(text_features.shape,text_features)
        numpy_array = text_features.cpu().detach().numpy()
        # 将numpy数组保存到txt文件
        with open("tensor_data.txt", "a") as f:  
            np.savetxt(f, numpy_array, fmt="%f")
            f.write("---------------------------------------------------------------------------------------------------------------------------\n")'''
        #np.savetxt("text_features.txt", numpy_array, fmt="%f", mode='a')

        logits = self.logit_scale * point_features @ text_features.t()
        #print(logits)
        if label != None:
            loss = F.cross_entropy(logits, label)
        pred = torch.max(logits,dim = 1)[1]
        acc = 0
        #print(pred)
        total = len(pred)
        for i in range(len(pred)):
            if(pred[i] == label[i]):
                acc +=1
        return {'loss':loss,
                'acc':acc,
                'total':total,
                #'text':text_features,
                #'point':point_features
                }
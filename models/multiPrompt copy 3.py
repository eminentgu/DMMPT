import open_clip
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
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        single_layer = nn.Linear(ctx_dim, 1408)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.stack([tokenizer(p) for p in prompts]) #cat
        self.tokenized_prompts = tokenized_prompts

        #visual_prompt = torch.empty(img_ctx,img_dim)
        #nn.init.normal_(visual_prompt, std=0.02)
        #self.visual_prompt = nn.Parameter(visual_prompt)
        #####self.img_pc_proj = nn.Linear(img_dim, 1408) 

         
    def forward(self,batch_size,image_embeddings=None):
        ctx = self.ctx
        tokenized_prompts = self.tokenized_prompts
        #visual_prompt = self.visual_prompt
        #point_prompt = torch.cat((self.text_pc_proj(ctx),self.img_pc_proj(visual_prompt)),axis = 0)
        #####visual_prompt = image_embeddings
        #####text_to_pc = self.text_pc_proj(ctx).repeat(image_embeddings.shape[0],1,1)
        #####img_to_pc = self.img_pc_proj(visual_prompt).unsqueeze(1)
        #####point_prompt =  torch.cat((text_to_pc,img_to_pc),axis = 1) 
        text_to_pc = self.text_pc_proj(ctx).repeat(batch_size
        ,1,1)
        point_prompt = text_to_pc
        
        ###point_prompt = self.text_pc_proj(ctx)
                                  
        point_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            point_deep_prompts.append(layer(self.compound_prompts_text[index]))

        ###return ctx,tokenized_prompts,point_prompt,self.compound_prompts_text,point_deep_prompts
        #return ctx,tokenized_prompts,visual_prompt,point_prompt,self.compound_prompts_text,point_deep_prompts
        return ctx,tokenized_prompts,point_prompt,self.compound_prompts_text,point_deep_prompts

class multiPrompt(nn.Module):
    def __init__(self,clip_model,pc_model, classnames) :
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prompt =  multiPromptLearner(classnames).cuda()#,clip_model
        self.pc_model = pc_model
        #self.pc_model.cuda()
        self.classnames = classnames
        self.clip_model = clip_model
        tokenizer = SimpleTokenizer()
        self.tokenizer = tokenizer
    def encode_pc(self, pc,point_prompt, point_deep_prompts):
        #point_prompt = point_prompt
        #print(pc,point_prompt)
        pc_feat = self.pc_model.encode_pc(pc, point_prompt, point_deep_prompts)
        return pc_feat
    def encoder_img(self,img,visual_prompt=None):
        '''with open('logs.txt', 'a') as f:
            f.write("\nmultiprompt line 98 img shape ,prompt shape"+str(img.shape)+str(visual_prompt.shape))
            print('multiprompt line 98 img shape ,prompt shape',img.shape,visual_prompt.shape)'''
        ###visual_prompt = visual_prompt.unsqueeze(0).repeat(img.shape[0], 1, 1)
        ###img_feat = self.clip_model.encode_image(img,visual_prompt)
        img_feat = self.clip_model.encode_image(img)  
        return img_feat
    def encoder_text(self,ctx,tokenized_prompts,compound_prompts_text):
        text = self.tokenizer(self.classnames)
        return self.clip_model.encode_text(text,ctx.to("cpu"),tokenized_prompts.to("cpu"),compound_prompts_text.to("cpu"))

    def forward(self,pc,img,label = None): #prompt_learner
        #####with torch.no_grad():
            #####image_features = self.encoder_img(img)
            #####image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        #####ctx,tokenized_prompts,point_prompt,compound_prompts_text, point_deep_prompts = self.prompt(image_features)
        ctx,tokenized_prompts,point_prompt,compound_prompts_text, point_deep_prompts = self.prompt(pc.shape[0])

        text_features = self.encoder_text(ctx,tokenized_prompts,compound_prompts_text)
        #image_features = self.encoder_img(img,visual_prompt)

        for i in range(len(point_deep_prompts)):
            point_deep_prompts[i] = point_deep_prompts[i].cuda(0)
        point_features = self.encode_pc(pc,point_prompt, point_deep_prompts)

        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cuda()
        point_features = point_features / point_features.norm(dim=-1, keepdim=True)
        #point_features = point_features.detach().cpu()


        logits = self.logit_scale * point_features @ text_features.t()
        if label != None:
            loss = F.cross_entropy(logits, label)
        pred = torch.max(logits,dim = 1)[1]
        acc = 0
        total = len(pred)
        for i in range(len(pred)):
            if(pred[i] == label[i]):
                acc +=1
        return {'loss':loss,
                'acc':acc,
                'total':total
                }
        #self.labels = torch.arange(40).long() #batch size
        #print("multiprompt line 118")
        #print(point_features.shape)
        #print(text_features.shape)
        #print(image_features.shape)
        '''loss = -1
        logits_per_pc_text = self.logit_scale * point_features @ text_features.t()

        logits_per_image_text = self.logit_scale * image_features @ text_features.t()
        if label != None:
            loss_text = F.cross_entropy(logits_per_pc_text, label)
            loss_image = F.cross_entropy(logits_per_image_text, label)
            loss = (loss_text + loss_image)/2
        pred = torch.max(logits_per_pc_text,dim = 1)[1]
        acc = 0
        total = len(pred)
        for i in range(len(pred)):
            if(pred[i] == label[i]):
                acc +=1'''
        #print(pred,label)
        '''return {'text_feat': text_features,
                'pc_feat': point_features,
                #'img_feat': image_features,
                'logit_scale': self.logit_scale.exp(),
                }'''
    '''
    'loss':loss,
                'acc':acc,
                'total':total
    '''

    '''image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        point_features = point_features / point_features.norm(dim=-1, keepdim=True)
        point_features = point_features.detach().cpu()
        logits = self.logit_scale * point_features @ text_features.t()
        if label != None:
            loss = F.cross_entropy(logits, label)
            
        pred = torch.max(logits,dim = 1)[1]
        acc = 0
        total = len(pred)
        for i in range(len(pred)):
            if(pred[i] == label[i]):
                acc +=1
        #print(pred,label)
        return {'text_feat': text_features,
                'pc_feat': point_features,
                #'image_feat': image_features,
                'logit_scale': self.logit_scale.exp(),
                'loss':loss,
                'acc':acc,
                'total':total}'''

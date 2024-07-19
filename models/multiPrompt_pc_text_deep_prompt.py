import open_clip
import torch
import torch.nn as nn
from utils.tokenizer import SimpleTokenizer
from torch.nn import functional as F
import numpy as np
import copy
from torch.cuda.amp import autocast as autocast
'''clip_model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained="./clip_model/clip_model.bin")
clip_model.to('cpu') #device'''


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class multiPromptLearner(nn.Module):
    def __init__(self, classnames ):
        tokenizer = SimpleTokenizer()
        super().__init__()
        # text ptompt init
        n_ctx = 2  #prompt size change transformer.py line 222,725 as well
        ctx_dim = 1280 # this should be modified to be the same as clip_model.ln_final.weight.shape[0] 
        # and here is 1280
        img_size =224
        self.compound_prompts_depth = 0 #9 

        ##shallow prompt
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        #print(f'Initial context: "{prompt_prefix}"')
        #print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
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
        #text = tokenizer(["fish","cat"])
        #print("text:",text)
        #print("prompts",prompts)
        #print("prompt shape",prompts.shape)
        #print("token prompts", tokenized_prompts.shape)
        #print(clip_model.encode_text(text,ctx_vectors,tokenized_prompts))

        visual_prompt = torch.empty(img_size,img_size)
        nn.init.normal_(visual_prompt, std=0.02)
        self.visual_prompt = nn.Parameter(visual_prompt)

        

        self.img_pc_proj = nn.Linear(img_size, 1408) 
        #self.text_pc_proj.half()

         
    def forward(self):
        ctx = self.ctx
        tokenized_prompts = self.tokenized_prompts
        visual_prompt = self.visual_prompt
        #print(ctx.shape)
        point_prompt = self.text_pc_proj(ctx) #+self.img_pc_proj(visual_prompt)

        point_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            point_deep_prompts.append(layer(self.compound_prompts_text[index]))

        return ctx,tokenized_prompts,visual_prompt,point_prompt,self.compound_prompts_text,point_deep_prompts


class multiPrompt(nn.Module):
    def __init__(self,clip_model,pc_model, classnames) :
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prompt =  multiPromptLearner(classnames)#,clip_model
        self.pc_model = pc_model
        self.pc_model.cuda()
        self.classnames = classnames
        self.clip_model = clip_model
        tokenizer = SimpleTokenizer()
        self.tokenizer = tokenizer
    def encode_pc(self, pc,point_prompt, point_deep_prompts):
        point_prompt = point_prompt.cuda()
        point_prompt = point_prompt.repeat(pc.shape[0],1,1)
        #print(pc,point_prompt)
        pc_feat = self.pc_model.encode_pc(pc, point_prompt, point_deep_prompts)
        return pc_feat
    def encoder_img(self,img,visual_prompt):
        visual_prompt = visual_prompt.unsqueeze(0)
        visual_prompt = visual_prompt.repeat(img.shape[0],1,1)
        img_feat = self.clip_model.encode_image(img+visual_prompt)
        return img_feat
    def encoder_text(self,ctx,tokenized_prompts,compound_prompts_text):
        text = self.tokenizer(self.classnames)
        return self.clip_model.encode_text(text,ctx,tokenized_prompts,compound_prompts_text)
    
    '''def get_text_prompt(self):
        return self.ctx'''
    def forward(self,pc,img,label = None): #prompt_learner
        ctx,tokenized_prompts,visual_prompt,point_prompt,compound_prompts_text, point_deep_prompts = self.prompt()
        #print("mulprompt.py line 82",pc.shape,img.shape,label.shape)
        text_features = self.encoder_text(ctx,tokenized_prompts,compound_prompts_text)
        image_features = self.encoder_img(img,visual_prompt)
        
        
        for i in range(len(point_deep_prompts)):
            point_deep_prompts[i] = point_deep_prompts[i].cuda(0)
        point_features = self.encode_pc(pc,point_prompt, point_deep_prompts)

        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        point_features = point_features / point_features.norm(dim=-1, keepdim=True)
        point_features = point_features.detach().cpu()
        #print("done! line94 text,pc feat",text_features,point_features)
        logits = self.logit_scale * point_features @ text_features.t()
        loss = None
        if label != None:
            loss = F.cross_entropy(logits, label)
            '''if(loss == float('nan')):
                print(logits,label)'''
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
                'total':total}




'''classnames = ["fish","cat","dog","pig"]
prompt = multiPromptLearner(classnames,clip_model)
ctx,tokenized_prompts,visual_prompt,point_prompt = prompt()
print(ctx.shape,tokenized_prompts.shape,visual_prompt.shape,point_prompt.shape)

image = preprocess(Image.open("1.png")).unsqueeze(0)
visual_prompt = visual_prompt.unsqueeze(0)
visual_prompt = visual_prompt.repeat(img.shape[0],1,1)
img_feat = clip_model.encode_image(image+visual_prompt)
print(img_feat.shape)'''
'''text = tokenizer(classnames)
print(prompt.text_encoder(text))
print(prompt.text_encoder(text).shape)'''
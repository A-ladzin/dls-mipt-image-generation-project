import numpy as np
import torch
import clip
from tqdm import tqdm
from IPython.display import clear_output
from torch import nn
import torch
from encoder4editing.models.encoders.model_irse import Backbone
import gc
from torch.cuda.amp import  GradScaler
import torch.nn.functional as F
from encoder4editing.models.stylegan2.op import fused_leaky_relu
from encoder4editing.configs.paths_config import model_paths
import math
from functools import wraps




import matplotlib.pyplot as plt


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1e-2, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim)/lr_mul)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )
        

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    

class LatentMapperBlock(nn.Module):
    def __init__(self,n_mlp,latent_dim):
        super(LatentMapperBlock,self).__init__()
        self.device = 'cuda'
        self.maps = nn.ModuleList()
        
        self.map = [PixelNorm()]
        
        for i in range(n_mlp):
            self.map.append(EqualLinear(latent_dim,latent_dim,lr_mul=2**-i,activation='fused_lrelu'))
            self.map.append(
                nn.BatchNorm1d(512,momentum=0.1)
            )
            self.map.append(
                nn.Dropout(0.1)
            )

        self.map = nn.Sequential(*self.map)

        
    def forward(self,x):
        out = self.map(x.mean(1))
        return  out.unsqueeze(1).expand(x.shape)




class LatentMapper(nn.Module):
    def __init__(self,num_latents,n_mlp,n_groups,latent_dim,levels = None, skip_levels = None):
        super(LatentMapper,self).__init__()
        self.device = 'cuda'
        self.maps = nn.ModuleList()

        self.n_groups = n_groups
        
        if levels == None:
            self.levels = []
            for i in range(1,n_groups):
                self.levels.append((num_latents//n_groups)*i)
        else:
            self.levels = levels
            
        for i in range(n_groups):
                if i in skip_levels:
                    self.maps.append(nn.Dropout(0.1))
                else:
                    self.maps.append(LatentMapperBlock(n_mlp,latent_dim))

        
    def forward(self,latents):
        for i in range(self.n_groups-1):
            if i == 0:
                out = self.maps[0](latents[:,:self.levels[0],:])
            else:
                out = torch.cat((out,self.maps[i](latents[:,self.levels[i-1]:self.levels[i],:])),dim=1)
        out = torch.cat((out,self.maps[-1](latents[:,self.levels[-1]:,:])),dim=1)

        return out



        



class LatentMapperOptimization(nn.Module):
    def __init__(self,prompt,generator,n_groups = 3, n_mlp = 4,inf_latents = None, levels = None,skip_levels = []):
        """
        Parameters
        ----------
        prompt : str, list[str]
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """
        super(LatentMapperOptimization,self).__init__()

        self.inf_latents = inf_latents
        

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = generator

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.prompt = prompt
        self.load_prompt(prompt)
        


        self.backbone = Backbone(112,50,'ir_se',drop_ratio=0.2).to(self.device)
        self.backbone.load_state_dict(torch.load(model_paths['ir_se50']))

        self.latent_mapper = LatentMapper(18,n_mlp,n_groups,512,levels=levels, skip_levels = skip_levels).to(self.device)


        self.requires_grad_(False)
        self.latent_mapper.requires_grad_(True)

        self.scaler = GradScaler()


        self.lambda_id = None
        self.lambda_l2 = None
        self.running_average_i = None
        self.running_average_l = None
        self.running_average_c = None
        self.total_i = None
        self.total_l = None
        self.total_c = None
        self.current_epoch = None
        self.epochs = None
        self.optimizer = None
        self.clip_threshold = 0.88


    def load_prompt(self,prompt):
        self.prompt_tokens = clip.tokenize(prompt).to(self.device)



    

    
    def clip_loss(self,sample,text):
        sample = nn.functional.adaptive_avg_pool2d(sample,(224,224))
        sample = sample-sample.min()
        sample = sample/sample.max()

        total = 1 - self.clip_model(sample,text)[0]/100

        return total.mean()
    

    def id_loss(self,a,b):
        return torch.mean(1 - torch.bmm(a.unsqueeze(1),b.unsqueeze(1).permute(0,2,1)))



    def id_encode(self,img):
        img = img-img.min()
        img = img/img.max()
        img = img[:, :, 35*4:223*4, 32*4:220*4]
        img = nn.functional.adaptive_avg_pool2d(img,(112,112))
        encoded = self.backbone(img)
        encoded = encoded/torch.norm(encoded,2,1,keepdim=True)
        return encoded



    def forward(self,latents,beta=1):
        m_latents = self.latent_mapper(latents)
        latents = beta*m_latents+latents
        sample, _ = self.gen(
        latents, truncation=0.8, return_latents = True,input_is_latent = True,randomize_noise = False
        )
        


        embedding = self.id_encode(sample)

        return sample,embedding,latents 
    


        
    def fit(self,epochs,
           lambda_id = 2, lambda_l2 = 2,
                id_running_mean_smooth = 0.99, l2_running_mean_smooth = 0.95, clip_running_mean_smooth = 0.9,
                batch_size = 3,lr = 0.07,lr_decay = 2,inference_interval = 1,mode = 'simple',**kwargs):
            """
            Parameters
            ----------
            epochs : int
                number of iterations
            lambda_id : float
                id_loss multiplier
            lambda_l2 : float
                l2_loss multiplier
            id_running_mean_smooth, l2_running_mean_smooth, clip_running_mean_smooth : float
                coefficients of calculated running means
            batch_size :
                number of generations each iteration
            lr :
                learning rate
            lr_decay :
                exponent of lr decreasing rate calculation
            inference_interval :
                shows intermediate statistics each
            **kwargs :
                individual parameters of training mode
            """
            if mode == 'balanced':
                func = self.fit_balanced_
            elif mode == 'experimental':
                func = self.fit_experimental_
            else: 
                func = self.fit_simple_


            self.lambda_id = lambda_id
            self.lambda_l2 = lambda_l2
            
            self.eval()
            self.latent_mapper.train()

            if self.optimizer == None:
                self.optimizer = torch.optim.AdamW(self.latent_mapper.parameters(),lr= lr,weight_decay=1e-3 , betas = (0.9,0.999)
                    )
            self.total_c = [0]
            self.total_i = [0]
            self.total_l = [0]
            self.running_average_i = [0]
            self.running_average_l = [0]
            self.running_average_c = [0]
            self.epochs = epochs

            

            for i in tqdm(range(epochs)):
                self.current_epoch = i
                self.eval()
                self.latent_mapper.train()
                self.optimizer.param_groups[0]['lr'] = lr - (lr*(i/epochs)**lr_decay)
                
                #Batch Generation
                if i>-1:
                    for j in range(batch_size):
                        styles = torch.randn(2,1, 512, device='cuda')
                        with torch.no_grad():
                            real_sample, latents_init = self.gen(
                                styles, truncation=0.7, return_latents = True,randomize_noise = True
                            )
                            
                            if j == 0:
                                latents_batch = latents_init
                                samples_batch = real_sample
                            else:
                                latents_batch = torch.cat((latents_init,latents_batch))
                                samples_batch =torch.cat((real_sample,samples_batch))
                        del latents_init
                        del real_sample
                        torch.cuda.empty_cache()
                        gc.collect()
                    real_id = self.id_encode(samples_batch)
                



                gc.collect()
                torch.cuda.empty_cache()

                
                
                #Forward pass
                sample,id_embed,latents_m = self.forward(latents_batch)
                c_loss = self.clip_loss(sample,self.prompt_tokens)
                id_loss = self.id_loss(id_embed,real_id)
                l2_loss = nn.functional.mse_loss(latents_batch,latents_m)
                self.total_c[-1] += c_loss.item()
                self.total_i[-1] += id_loss.item()
                self.total_l[-1] += l2_loss.item()

                if i > 0:
                    self.running_average_i[-1] = (self.running_average_i[-2]*(len(self.total_i)-1) + self.total_i[-1])/(len(self.total_i))
                    self.running_average_i[-1] = self.running_average_i[-1]*id_running_mean_smooth+(1-id_running_mean_smooth)*self.total_i[-1]
                    self.running_average_l[-1] = (self.running_average_l[-2]*(len(self.total_l)-1) + self.total_l[-1])/(len(self.total_l))
                    self.running_average_l[-1] = self.running_average_l[-1]*l2_running_mean_smooth+(1-l2_running_mean_smooth)*self.total_l[-1]
                    self.running_average_c[-1] = (self.running_average_c[-2]*(len(self.total_c)-1) + self.total_c[-1])/(len(self.total_c))
                    self.running_average_c[-1] = self.running_average_c[-1]*clip_running_mean_smooth+(1-clip_running_mean_smooth)*self.total_c[-1]

                else:
                    self.running_average_i[-1] = self.total_i[-1]
                    self.running_average_l[-1] = self.total_l[-1]
                    self.running_average_c[-1] = self.total_c[-1]


                #####
                loss = func(c_loss,id_loss,l2_loss,**kwargs)
                #####
               

                #Backward pass         
                self.scaler.scale(loss).backward(retain_graph=True)
        
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()


                
                #Graph visualization
                if i % inference_interval == 0:
                    clear_output(True)
                    fig = plt.figure(figsize = (16,9))
                    ax = plt.axes(facecolor = "#EAE6CE")
                    plt.title(f"Epoch: {i}/{epochs}, Current LR: {self.optimizer.param_groups[0]['lr']:.4f}")
                    plt.grid(True)
                    plt.plot(self.total_c)
                    plt.plot(self.total_i)
                    plt.plot(self.total_l)

                    plt.plot(self.running_average_c,linestyle = 'dotted', c = '#111166')
                    plt.plot(self.running_average_i,linestyle = 'dotted', c = 'magenta')
                    plt.plot(self.running_average_l, linestyle = 'dotted', c = '#116611')
                    plt.legend([f"Clip Threshold: {self.clip_threshold:.3f}",f"λ Id: {self.lambda_id:.3f}", f"λ L2: {self.lambda_l2:.3f}",f"Average Clip: {self.running_average_c[-1]:.3f}",f'Average ID: {self.running_average_i[-1]:.3f}', f'Average L2: {self.running_average_l[-1]:.3f}'])
                    
                    if not isinstance(self.inf_latents, type(None)):
                        self.eval()
                        plt.figure(figsize = (20,4))
                        with torch.no_grad():
                            for x in range(1,8):
                                plt.subplot(1,7,x)
                                result,_,_ = self.forward(self.inf_latents,beta = 0.2*x)
                                result = result.detach().cpu().squeeze().permute(1,2,0)
                                result = torch.clip_((result+1)/2,0,1)
                                plt.xticks([])
                                plt.yticks([])
                                plt.imshow(result)

                    plt.show()

                self.total_c.append(0)
                self.total_i.append(0)
                self.total_l.append(0)
                self.running_average_l.append(0)
                self.running_average_i.append(0)
                self.running_average_c.append(0)
                
                # if i % 500 == 0:
                #     torch.save(self.latent_mapper.state_dict(),f"{self.prompt[0]}_{i}.pt")


    def fit_simple_(self,c_loss,id_loss,l2_loss,**kwargs):
        return c_loss+id_loss*self.lambda_id+l2_loss*self.lambda_l2
   

    def fit_balanced_(self,c_loss,id_loss,l2_loss,**kwargs):
            """
            Parameters
            ----------
                clip_correction_factor : float
                    decreasing lambdas force
                id_correction_factor, l2_correction_factor : float, float
                    increasing lambdas force
                id_threshold, l2_threshold : float
                    lambdas will not be dropped below these points
                explode_ : float
                    critical point for id_lambda
            """
            
            kwargs.setdefault("clip_correction_factor",0.05)
            kwargs.setdefault("id_correction_factor",0.05)
            kwargs.setdefault("l2_correction_factor",0.05)
            kwargs.setdefault("id_threshold",0)
            kwargs.setdefault("l2_threshold",0)
            kwargs.setdefault("explode_",5)

            
            id = np.mean(self.total_i[-5:]) > self.running_average_i[-1] 
            l2 = np.mean(self.total_l[-5:]) > self.running_average_l[-1]
            cc = np.mean(self.total_c[-5:]) > self.running_average_c[-1]
            if cc:
                if kwargs["id_threshold"]<self.lambda_id:
                    self.lambda_id-=(self.lambda_id*kwargs["clip_correction_factor"])
                if kwargs["l2_threshold"] < self.lambda_l2:
                    self.lambda_l2-=(self.lambda_l2*kwargs["clip_correction_factor"])
            else:
                if id:
                    self.lambda_id+=(self.lambda_id*kwargs["id_correction_factor"])
                if l2:
                    self.lambda_l2+=(self.lambda_l2*kwargs["l2_correction_factor"])
            
            if self.lambda_id > kwargs["explode_"]:
                self.lambda_id /=2

            loss = self.lambda_id*id_loss + c_loss + self.lambda_l2*l2_loss

            return loss
    


    def fit_experimental_(self,c_loss,id_loss,l2_loss,**kwargs):
        """
        Parameters
        ----------
            progressive_start : int
                pregressive tuning starts
            progressive_end : int
                progressive_tuning_ends
            progressive_interval : int
                step length of progressive tuning
            clip_target : float
                c_loss be pushed to that target
            l2_decay:
                l2 decay rate
            id_decay:
                id decay rate

        """
        kwargs.setdefault("progressive_start",self.epochs//10)
        kwargs.setdefault("progressive_end",int(self.epochs*0.8))
        kwargs.setdefault("progressive_interval",(kwargs['progressive_end']-kwargs['progressive_start'])//20)
        kwargs.setdefault("l2_decay",0.95)
        kwargs.setdefault("id_decay",1.05)
        kwargs.setdefault("clip_target",0.666)


        loss = c_loss+id_loss*self.lambda_id+l2_loss*self.lambda_l2


        if self.current_epoch > kwargs["progressive_start"]:
            if self.total_c[-1] > self.clip_threshold:
                loss = c_loss
            elif self.total_i[-1] > self.running_average_i[-1]:
                loss = self.lambda_id*id_loss
                if self.total_l[-1] > self.running_average_l[-1]:
                    loss += self.lambda_l2*l2_loss
            
            if self.current_epoch%kwargs["progressive_interval"] == 0:
                self.lambda_l2 = self.lambda_l2*kwargs["l2_decay"]
                self.lambda_id = self.lambda_id*kwargs["id_decay"]
                self.clip_threshold = max(0.8 - (0.8-kwargs["clip_target"])*(self.current_epoch/(kwargs["progressive_end"])),kwargs["clip_target"])

        return loss
                



            




            
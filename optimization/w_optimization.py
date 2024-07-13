import numpy as np
import torch
import clip
from tqdm import tqdm
from IPython.display import clear_output
from torch import nn
import torch
from encoder4editing.models.encoders.model_irse import Backbone
import matplotlib.pyplot as plt


class Optimizer(nn.Module):
    def __init__(self,latents_init,truncation,prompt,gen,device = "cuda"):
        super(Optimizer,self).__init__()

        self.gen=gen

        self.device = device
        self.truncation = torch.Tensor([truncation]).to(self.device)
        self.latents = latents_init.to(self.device)

        self.latents_init = torch.clone(latents_init).to(self.device)


        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.prompt_tokens = clip.tokenize(prompt).to(self.device)


        self.backbone = Backbone(112,50,'ir_se',drop_ratio=0.5).to(self.device)
        self.backbone.load_state_dict(torch.load('encoder4editing/pretrained_models/model_ir_se50.pth'))
        self.requires_grad_(False)
        _,self.id = self.forward(self.latents_init)

        
        self.latents.requires_grad_(True)







    
    def clip_loss(self,sample,text):
        sample = nn.functional.adaptive_avg_pool2d(sample,(224,224))
        sample = sample-sample.min()
        sample = sample/sample.max()
        total = 1- self.clip_model(sample,text)[0]/100
        return total
    

    def id_loss(self,a,b):
        return 1 - torch.dot(a.view(-1),b.view(-1))



    def id_encode(self,img):
        img = img-img.min()
        img = img/img.max()
        img = nn.functional.adaptive_avg_pool2d(img,(256,256))
        img = img[:, :, 35:223, 32:220]
        img = nn.functional.adaptive_avg_pool2d(img,(112,112))
        encoded = self.backbone(img)
        encoded = encoded/torch.norm(encoded,2,1,keepdim=True)
        return encoded



    def forward(self,latents):
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.gen.eval()
        sample, _ = self.gen(
        latents, truncation=self.truncation, return_latents = True,input_is_latent = True,randomize_noise = True
        )

        embedding = self.id_encode(sample)

        return sample,embedding
    

    def fit(self,epochs,
           lambda_id = 2, lambda_l2 = 2,
           correction_factor = 1e-3,
            running_mean_smooth = 0.99,
            lr_decay = 1.5,
            clip_threshold = 0,
            lr = 0.07,
            optimizer = None, 
            progressive_delay = 0,
            validation_interval = 100,make_video = False
            ):
        
        self.train()
        self.latents.requires_grad_(True)

        if make_video:
                import imageio
                frames = []
                from PIL import Image


        if optimizer == None:
            optimizer = torch.optim.AdamW([self.latents],lr = lr)

        total_c = [0]
        total_i = [0]
        total_l = [0]
        running_average_i = [0]
        running_average_l = [0]
        running_average_c = [0]

        for i in tqdm(range(epochs)):
            optimizer.param_groups[0]['lr'] = lr-lr*(i/epochs)**np.exp(lr_decay)
            self.backbone.eval()
            self.backbone.requires_grad_(False)
            optimizer.zero_grad()
            sample,id_embed = self.forward(self.latents)


            c_loss = self.clip_loss(sample,self.prompt_tokens)
            id_loss = self.id_loss(id_embed,self.id)
            l2_loss = nn.functional.mse_loss(self.latents_init,self.latents)
            
            total_c[-1] += c_loss.item()
            total_i[-1] += id_loss.item()
            total_l[-1] += l2_loss.item()
            if c_loss < clip_threshold:
                c_loss*=0

            if i > 0:
                running_average_i[-1] = (running_average_i[-2]*(len(total_i)-1) + total_i[-1])/(len(total_i))
                running_average_i[-1] = running_average_i[-1]*running_mean_smooth+(1-running_mean_smooth)*total_i[-1]
                running_average_l[-1] = (running_average_l[-2]*(len(total_l)-1) + total_l[-1])/(len(total_l))
                running_average_l[-1] = running_average_l[-1]*running_mean_smooth+(1-running_mean_smooth)*total_l[-1]
                running_average_c[-1] = (running_average_c[-2]*(len(total_c)-1) + total_c[-1])/(len(total_c))
                running_average_c[-1] = running_average_c[-1]*running_mean_smooth+(1-running_mean_smooth)*total_c[-1]

            else:
                running_average_i[-1] = total_i[-1]
                running_average_l[-1] = total_l[-1]
                running_average_c[-1] = total_c[-1]


            if i > progressive_delay:
                id = np.mean(total_i[-10:]) > running_average_i[-1] 
                l2 = np.mean(total_l[-10:]) > running_average_l[-1]
                cc = c_loss > running_average_c[-1]

                if cc:
                    lambda_id-=(lambda_id*correction_factor)
                    lambda_l2-=(lambda_l2*correction_factor)
                else:
                    if id:
                        lambda_id+=(lambda_id*correction_factor)
                    if l2:
                        lambda_l2+=(lambda_l2*correction_factor)
                
            loss = lambda_id*id_loss + c_loss + lambda_l2*l2_loss

            #Backward pass         
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            

            if i % validation_interval == 0:
                #Graph visualization
                with torch.no_grad():
                    clear_output(True)
                    
                    fig = plt.figure(figsize = (16,9))
                    ax = plt.axes(facecolor = "#EAE6CE")
                    plt.title(f"Epoch: {i}/{epochs}, Current LR: {optimizer.param_groups[0]['lr']:.4f}")
                    plt.grid(True)
                    plt.plot(total_c)
                    plt.plot(total_i)
                    plt.plot(total_l)
                    plt.plot(running_average_c,linestyle = 'dotted', c = '#111166')
                    plt.plot(running_average_i,linestyle = 'dotted', c = 'magenta')
                    plt.plot(running_average_l, linestyle = 'dotted', c = '#116611')
                    plt.legend([f"Clip loss",f"λ Id: {lambda_id:.3f}", f"λ L2: {lambda_l2:.3f}",f"Average Clip: {running_average_c[-1]:.3f}",f'Average ID: {running_average_i[-1]:.3f}', f'Average L2: {running_average_l[-1]:.3f}'])
                    plt.figure(figsize = (8,4))
                    plt.subplot(1,2,1)
                    plt.xticks([])
                    plt.yticks([])
                    sample = torch.clip_((sample[0]+1)/2,0,1)
                    plt.imshow(sample.detach().cpu().permute(1,2,0))
                    inversion,_ = self.gen(self.latents_init,truncation = 1,return_latents = False,input_is_latent = True,randomize_noise = False)
                    inversion = torch.clip_((inversion[0]+1)/2,0,1)
                    plt.imshow(inversion.detach().cpu().permute(1,2,0))
                    plt.subplot(1,2,2)
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(sample.detach().cpu().permute(1,2,0))
                    plt.show()
            elif make_video:
                sample = torch.clip_((sample[0]+1)/2,0,1)

            if make_video:
                im = Image.fromarray(np.uint8(nn.functional.adaptive_avg_pool2d(sample.detach().cpu(),(512,512)).permute(1,2,0)*255))
                frames.append(im)
            


            total_c.append(0)
            total_i.append(0)
            total_l.append(0)
            running_average_l.append(0)
            running_average_i.append(0)
            running_average_c.append(0)
        if make_video:
            imageio.mimsave('results/movie.gif', frames,loop=0)









        
        
        
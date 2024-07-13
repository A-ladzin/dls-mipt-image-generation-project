from argparse import Namespace
from encoder4editing.models.psp import pSp
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from encoder4editing.utils.common import tensor2im
from encoder4editing.utils.alignment import align_face
from encoder4editing.configs.paths_config import model_paths
import dlib
from functools import wraps

def load_model(model_path,stylegan_size = 1024):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['stylegan_size'] = stylegan_size
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    return net





def run_alignment(image_path,verbose = 2):

  predictor = dlib.shape_predictor(model_paths['shape_predictor'])
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  if verbose:
    print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True,resize = False)
    return images, latents

def prepare_img(image_path,net,resize_dims = (256,256),verbose = 2,return_original = False):
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    
    input_image = run_alignment(image_path,verbose = verbose)
    input_image.resize(resize_dims)

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]

        if verbose > 1:
            f = plt.figure()
            f.set_facecolor("black")
            plt.xticks([])
            plt.yticks([])
            plt.title("Inversion")
            plt.imshow(display_alongside_source_image(tensor2im(result_image),input_image))
            plt.show()
    if return_original:
        return transformed_image,result_image,latent.unsqueeze(0)
    return result_image,latent.unsqueeze(0)


def display_alongside_source_image(result_image, source_image,resize_dims = (256,256)):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)






def series(image_path,optimizer,net,beta_step = 0.35, n_series = 4,mode = 'show'):
    if mode == 'show':
        func = show_series
    else:
        func = get_series
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    resize_dims = (256,256)

    input_image = run_alignment(image_path,verbose=0)

    input_image.resize(resize_dims)

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
    result_image = result_image-result_image.min()
    result_image = result_image/result_image.max()
    optimizer.eval()
    series = None
    for i in range(n_series+1):
        if i == 0:
            result = func(i,n_series,input_image,beta_step)
            if not result is None:
                series = result
        else:
            samp = optimizer(latents,beta=beta_step*(i-1))[0].detach().cpu().squeeze().permute(1,2,0)
            samp = (samp+1)/2
            samp = torch.clip_(samp,0,1)
            result = func(i,n_series,samp,beta_step)
            if not result is None:
                series = torch.cat((series,result),dim=1)
    if not result is None:
        return series




def show_series(i,n_series,img,beta_step):
    title = f"β : {(i-1)*beta_step}"
    if i == 0:
        fig = plt.figure(figsize = ((n_series+1)*8,8))
        fig.set_facecolor("black")
        title = 'real_image'
    if i == 1:
        title = 'inversion'
    plt.subplot(1,n_series+1,i+1)
    plt.title(title,c="white")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


def get_series(i,n_series,img,beta_step):
    if i == 0:
        img = transforms.Resize((1024,1024))(img)
        img = transforms.ToTensor()(img)
        img = img.permute(1,2,0)
    return img


def create_personal_grid(image_path,mappers,optimizer,net,beta_step = 0.35, n_series=4):
    grid = []
    fig = plt.figure(figsize = ((4*n_series,4*len(mappers))))
    fig.set_facecolor("black")
    plt.xticks([])
    plt.yticks([])
    for prompt in mappers:
        optimizer.load_state_dict(torch.load(prompt))
        grid.append(series(image_path,optimizer,net,beta_step = beta_step,n_series = n_series,mode='get'))
    grid = torch.cat(grid,dim=0)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)
    




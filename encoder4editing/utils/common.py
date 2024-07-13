from PIL import Image
import matplotlib.pyplot as plt
import torch


# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')


def generate_grid(generator,optimizer,n_rows = 4, n_columns = 5, beta_step = 0.3):
    """
    Print intensity scale grid of randomly generated images
    """
    gen = generator
    opt = optimizer

    gen.eval()
    opt.eval()
    for j in range(n_rows):
        styles = torch.randn(2,1, 512, device='cuda')
        with torch.no_grad():
            real_sample, latents_init = gen(
                styles, truncation=0.2, return_latents = True,randomize_noise = True
            )
            real_sample =real_sample-real_sample.min()
            real_sample = real_sample / real_sample.max()
            if j == 0:
                latents_batch = latents_init
                samples_batch = real_sample
            else:
                latents_batch = torch.cat((latents_init,latents_batch))
                samples_batch =torch.cat((real_sample,samples_batch))
        del latents_init
        del real_sample
    fig = plt.figure(figsize = (40,4*latents_batch.shape[0]))
    fig.set_facecolor('black')
    plt.xticks([])
    plt.yticks([])


    for i in range(latents_batch.shape[0]):
        row = samples_batch[i].detach().cpu().permute(1,2,0)
        if i == 0:
            reals = row
        else:
            reals = torch.cat((reals,row))

    for x in range(n_columns-1):
        sample,id_embed,latents_m = opt.forward(latents_batch,beta =beta_step*(x+1))
        sample = torch.clip_((sample+1)/2,0,1)
        for i in range(latents_m.shape[0]):
            img = sample[i].detach().cpu().permute(1,2,0)
            if i == 0:
                column = img
            else:
                column = torch.cat((column,img),dim=0)
        if x == 0:
            mods = column
        else:
            mods = torch.cat((mods,column),dim=1)
    result = torch.cat((reals,mods),dim=1)
    plt.imshow(result)
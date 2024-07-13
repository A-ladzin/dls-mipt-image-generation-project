
## Text-Based StyleGAN Latent Space Optimization for Editing Face Images
###### MIPT Deep Learning School Project




### Description:
Implementation of two methods of text-driven image manipulation:
-   Latent vector optimization
-   Latent mapper

from the paper [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://arxiv.org/abs/2103.17249)


### Prerequisites
- NVIDIA GPU + CUDA CuDNN
- Python 3.10.7


### Installation

1. Clone the repository

```
git clone https://github.com/A-ladzin/dls-mipt-image-generation-project.git
cd dls-mipt-image-generation-project
```

2. Create and activate a new environment
```
python -m venv venv
source venv/Scripts/activate
```

3. Install all dependencies from requirements.txt
```
pip install -r requirements.txt
```

4. Download pretrained models:

    -   Download one of pretrained encoders:
    -   
        |[FFHQ Inversion](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing) | Pretrained FFHQ e4e encoder taken from [omertov](https://github.com/omertov/encoder4editing/).
        
        |[FFHQ Inversion](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view) | Pretrained FFHQ pSp encoder taken from [eladrich](https://github.com/eladrich/pixel2style2pixel).

    -   Download pretrained IR-SE50 model:
    
        |[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch).
    -   Download Face Landmarks predictor:
        [dlib](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    
    Save all models to encoder4editing/pretrained_models


5. 
    -   You can find some pretrained latent mappers at the following link [G-Drive](https://drive.google.com/drive/folders/1Ib2X7izns9D-oWEtVasGF3jg4Rnzh-0h?usp=sharing)
        
        Save them to `mappers/`


6. You can get aquainted with the training process and see some examples with result evaluations in the `optimization_playground.ipynb` for the latent vector optimization or `mapper_playground.ipynb` notebook for the latent mapper optimization


![alt text](results/movie.gif)


## Related Works

[e4e](https://arxiv.org/abs/2102.02766) (Tov et al.).

    

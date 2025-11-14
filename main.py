import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random
from zipfile import ZipFile
import shutil
import subprocess
import requests
from io import BytesIO
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from types import SimpleNamespace
from typing import Any
import re
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

#########################################
SHOW_PLTS = False
#########################################


#########################################
#########################################
### DCGAN CODE BASED ON TORCH DOCS ###
#########################################
#########################################

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m:Any)->None:
    """
    Initialize weights for the model specifiec in parameter m.
    No return value.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 1) Self‐Attention block (SAGAN style)
class SelfAttention(nn.Module):
    '''
    Self Attention block based upon SAGAN.
    Takes input dimension as a parameter, creates an instance of itself.
    '''
    def __init__(self, in_dim:int)->None:
        '''
        Initialize the Self Attention class instance, takes input dimension parameter.
        No return value
        '''
        super().__init__()
        self.ch_in    = in_dim
        self.query    = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key      = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value    = nn.Conv2d(in_dim, in_dim,    1)
        self.gamma    = nn.Parameter(torch.zeros(1))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        Forwrd pass of Self Attention module.
        Takes a tensore as input, and returns a modifired tensor.
        '''
        B, C, H, W = x.shape
        # project
        proj_q = self.query(x).view(B, -1, H*W).permute(0,2,1)  # (B, N, C//8)
        proj_k = self.key(x).view(B, -1, H*W)                   # (B, C//8, N)
        energy = torch.bmm(proj_q, proj_k)                     # (B, N, N)
        attn   = torch.nn.functional.softmax(energy, dim=-1)                     # (B, N, N)
        proj_v = self.value(x).view(B, -1, H*W)                # (B, C, N)

        out = torch.bmm(proj_v, attn.permute(0,2,1))           # (B, C, N)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# Generator Code

class Generator(nn.Module):
    '''
    Generator class of the GAN system. Takes hyperparameters as input for init.
    Creates an instance of itself.
    '''
    def __init__(self, params:SimpleNamespace)->None:
        '''
        Initialization function for the Generator.
        Take a SimpleNamespace dict of hyperparameters as input.
        No return value
        '''
        super(Generator, self).__init__()
        Fmap = params.gen_featuremap_sz
        self.num_gpu = params.num_gpu

        self.block4x4 = nn.Sequential(
            nn.ConvTranspose2d(params.z_size, Fmap*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Fmap*8), nn.ReLU(True),
        )
        self.block8x8 = nn.Sequential(
            nn.ConvTranspose2d(Fmap*8, Fmap*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Fmap*4), nn.ReLU(True),
        )
        self.block16x16 = nn.Sequential(
            nn.ConvTranspose2d(Fmap*4, Fmap*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Fmap*2), nn.ReLU(True),
        )
        self.block32x32 = nn.Sequential(
            nn.ConvTranspose2d(Fmap*2, Fmap, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Fmap), nn.ReLU(True),
        )

        # ← INSERT: self-attention on 32×32 feature maps
        self.attn = SelfAttention(Fmap)

        self.block64x64 = nn.Sequential(
            nn.ConvTranspose2d(Fmap, params.num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input:torch.Tensor)->torch.Tensor:
        """
        Forward pass on the network.
        Takes a batch of input image data, returns generated images.
        """
        x = self.block4x4(input)    # 4×4
        x = self.block8x8(x)        # 8×8
        x = self.block16x16(x)      # 16×16
        x = self.block32x32(x)      # 32×32
        x = self.attn(x)            # apply self-attention
        x = self.block64x64(x)      # 64×64 output
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN.
    Inputs:
        (batch_size, num_channels, 256, 256) images.
    Outputs:
        Real/Fake probability for each image (batch_size,).
    """
    def __init__(self, params:SimpleNamespace)->None:
        """
        Intializes the Discriminator network.
        Takes a hyperparamter dictionary as input.
        No return value, but will create an instance of the class.
        """
        super(Discriminator, self).__init__()
        self.num_gpu = params.num_gpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(params.num_channels, params.dis_featuremap_sz, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(params.dis_featuremap_sz) x 32 x 32``
            nn.Conv2d(params.dis_featuremap_sz, params.dis_featuremap_sz * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.dis_featuremap_sz * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(params.dis_featuremap_sz*2) x 16 x 16``
            nn.Conv2d(params.dis_featuremap_sz * 2, params.dis_featuremap_sz * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.dis_featuremap_sz * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(params.dis_featuremap_sz*4) x 8 x 8``
            nn.Conv2d(params.dis_featuremap_sz * 4, params.dis_featuremap_sz * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.dis_featuremap_sz * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(params.dis_featuremap_sz*8) x 4 x 4``
            nn.Conv2d(params.dis_featuremap_sz * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input:torch.Tensor)->torch.Tensor:
        """
        Forward pass on the network.
        Takes a batch of input image data, returns class probability of the images.
        """
        return self.main(input)


#########################################
#########################################
#########################################
#########################################

def generate_dataset(min_f_score: int, redownload: bool)->tuple:
    '''
    Used to generate a fresh copy of the dataset.
    Downloads the reference CSV from the fitzpatrick17k project Github,
    then downloads the relevant images, and cleans the data. Takes integer representing
    lower bound (inclusive) to filter rows by Fitzpatrick score by. Also takes a boolean indicating 
    whether to re-download files alread present locally.
    Returns the paths to a CSV and the image directory as a tuple, or (-1,-1) if the process failed.

    '''

    # Mimic a real browser and establish requests session
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0 Safari/537.36",
    }
    session = requests.Session()
    session.headers.update(browser_headers)

    # Create save directories
    save_dir = 'data/fresh_data/'
    img_dir = save_dir + 'src_images/'
    os.makedirs(img_dir, exist_ok=True)

    if redownload or not os.path.exists(save_dir+'fitzpatrick17k.csv'):
        try:
            response = session.get('https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv')
            response.raise_for_status()

            with open(save_dir+'fitzpatrick17k.csv', "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f'Failed to retrieve source CSV: {e}')
            session.close()
            raise RuntimeError('Failed to generate dataset')
    

    # Read in source CSV to a dataframe
    df = pd.read_csv(save_dir+'fitzpatrick17k.csv')

    # Filter to desired Fitzpatrick score
    df = df[df['fitzpatrick_scale'] >= min_f_score]

    # Filter by type
    type_filter = [
    'melanoma','lentigo maligna','malignant melanoma','superficial spreading melanoma'
    ]

    pattern = r"(?i)\b(?:" + "|".join(re.escape(t) for t in type_filter) + r")\b"
    df = df[df["label"].str.contains(pattern, na=False)]

    # Drop rows with missing URLs
    df.dropna(subset=['url'], inplace=True)

    # Download all images using URLs in dataframe. Record failed and invalid downloads
    remove_idxs = []
    remove_reasons = []
    img_pths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        try:
            img_pth = os.path.join(img_dir + str(row.fitzpatrick_scale) + '/' + row.md5hash + '.jpg')
            if not redownload and os.path.exists(img_pth):
                img_pths.append(str(row.fitzpatrick_scale) + '/' + row.md5hash + '.jpg')
                continue

            response = session.get(row.url)
            response.raise_for_status()

            # 2) open with PIL from the bytes buffer
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                print(f"Skipping non-RGB ({img.mode}):", row.url)
                remove_idxs.append(idx)
                remove_reasons.append('NON_RGB')
                continue

            os.makedirs(img_dir + str(row.fitzpatrick_scale) + '/', exist_ok=True)
            with open(img_pth, "wb") as f:
                f.write(response.content)

            img_pths.append(str(row.fitzpatrick_scale) + '/' + row.md5hash + '.jpg')
        except Exception as e:
            print(f"\nERROR for md5hash: {row.md5hash}\n\t{e}")
            remove_idxs.append(idx)
            remove_reasons.append('FAILED DOWNLOAD')

    # Save a record of the rows with unsaved images
    unsaved_df = df.loc[remove_idxs].copy()
    unsaved_df['Reason'] = remove_reasons
    unsaved_df.reset_index(drop=True, inplace=True)
    unsaved_df.to_csv(save_dir+'unsaved_rows.csv')

    # Removing failed entries from dataframe
    df.drop(index=remove_idxs, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['local_path'] = img_pths
    df.to_csv(save_dir+'downloaded_fitzpatrick17k.csv')

    session.close()

    return (save_dir+'downloaded_fitzpatrick17k.csv', img_dir)

def download_dataset(redownload: bool)->tuple:
    '''
    Used to download the version of the prepared data used in this project.
    Downloads the reference CSV and the image files.
    Takes a boolean indicating whether to re-download the files.
    Returns the paths to a CSV and the image directory as a tuple, or (-1,-1) if the process failed.
    '''

    target_dir = 'data/prepared_data/'
    csv_pth = ''
    img_dir = ''
    
    # If not redownload, confirm data is intact, otherwise need to redownload
    if os.path.exists(target_dir) and not redownload:
        total = 0
        count = 0
        for dirpath, dirnames, filenames in os.walk(target_dir):
            total += len(filenames)
            if dirpath == target_dir:
                img_dir = dirpath + dirnames[0] + '/'
            for file in filenames:
                if '.pth' in file:
                    total -= 1
                if '.json' in file:
                    total -= 1
                if '.csv' in file:
                    total -= 1
                    csv_pth = dirpath+file
                    with open(csv_pth, 'r', encoding='utf-8', errors='ignore') as f:
                        count = sum(1 for _ in f) - 1
        if count > 0 and count == total:
            return (csv_pth, img_dir)

    # Mimic a real browser and establish requests session
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0 Safari/537.36",
    }
    session = requests.Session()
    session.headers.update(browser_headers)

    zip_url = 'https://github.com/jkelly8823/fitzpatrick_testing/archive/refs/heads/main.zip'

    with tqdm(total=1, desc="Processing", bar_format="{desc} {bar} | elapsed: {elapsed}", leave=True) as pbar:
        response = session.get(zip_url)
        response.raise_for_status()


        with ZipFile(BytesIO(response.content)) as z:
            names = z.namelist()
            first_entry = names[0].split('/', 1)[0]
            extract = [nm for nm in names if '.pth' not in nm]
            extracted_dir = first_entry
            z.extractall(members=extract)
        
        for nm in names:
            splits = nm.split('/')
            if 'csv' in nm:
                csv_pth = target_dir + splits[1]
            if len(splits) > 2:
                img_dir = target_dir + splits[1] + '/'
            if csv_pth != '' and img_dir != '':
                break
        shutil.move(extracted_dir, target_dir)

        pbar.update(1)

    return (csv_pth, img_dir)

class ImageTransformer:
    '''
    A class to wrap the image transformer function for the transform pipeline.
    Takes a target image size as an int on initialization. Takes a PIL Image.Image when called.
    Returns a rescaled and padded PIL Image.Image when called.
    This class exists to avoid pipeline pickling errors with multiple workers.
    '''
    def __init__(self, target_size:int)->None:
        '''
        Initialize image transformer. Takes target size int input, no direct return value.
        '''
        self.target_sz = target_size
        pass
    def __call__(self, image:Image.Image)->Image.Image:
        new_img = self.image_scaler(img=image, target_size=self.target_sz)
        return new_img
        
    def image_scaler(self, img: Image.Image, target_size:int)->Image.Image:
        '''
        Function to rescale and pad images to a given dimension.
        Takes a PIL Image.Image and int target image size as inputs.
        Returns a rescaled and padded image.
        '''

        # Size to resize and pad to target size

        # Resize image
        w, h = img.size
        scale_factor = min(target_size/w, target_size/h)
        new_w, new_h = int(w*scale_factor), int(h*scale_factor)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Pad image
        pad_w, pad_h = (target_size - new_w), (target_size - new_h)
        padding = (
            pad_w//2,
            pad_h//2,
            pad_w - pad_w//2,
            pad_h - pad_h//2
        )
        img = ImageOps.expand(img, padding)

        return img

def build_dataloaders(img_dir: str, batch_sz: int, worker_num: int, img_sz:int)->set:
    '''
    Function to build the training and testing dataloaders for use with torch models.
    Takes the top level directory containing images, and ints for batch size, number of workers,
    and target image size as parameters. Returns a set like (training_dataloader, test_dataloader).
    '''
    # Transform the images
    tfms = transforms.Compose([
        ImageTransformer(img_sz),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std= [0.5, 0.5, 0.5]
        ),
    ])

    # Create the dataset
    dataset = datasets.ImageFolder(
        root=img_dir,
        transform=tfms
    )

    # Create split counts for data
    train_ratio = 0.8
    data_len = len(dataset.samples)
    train_count  = int(data_len * train_ratio)
    test_count = data_len - train_count

    # Randomly split into train and test sets
    train_data, test_data = random_split(
        dataset,
        [train_count, test_count],
        generator=torch.Generator().manual_seed(42)
    )

    # Create the dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=worker_num,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=worker_num,
        pin_memory=True
    )

    return (train_loader, test_loader)

def get_model_checkpt(redownload:bool)->set:
    '''
    Function to retrieve the checkpoint file for the trained model from github.
    Takes a boolean redownload parameter, indicating whether to redownload checkpoints
    that already exist locally.
    Returns a set of paths to checkpoint model checkpoints like (g_path, d_path).
    '''

    base_pth = 'data/prepared_data/'
    os.makedirs(base_pth, exist_ok=True)

    names = [
        'generator_state', 
        'discriminator_state', 
        'clf_baseline_state',
        'clf_history_baseline',
        'clf_state',
        'clf_history'
    ]

    print("Retrieving model checkpoints...")

    file_pth = ''
    try:
        # Mimic a real browser and establish requests session
        browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0 Safari/537.36",
            "Authorization": "", #HIDDEN
        }
        session = requests.Session()
        session.headers.update(browser_headers)
        
        ret_pths = []
        for nm in names:
            file_pth = f'{base_pth}{nm}.pth'
            ret_pths.append(file_pth)
            if os.path.exists(file_pth) and not redownload:
                continue
            
            response = session.get(f'https://raw.githubusercontent.com/jkelly8823/fitzpatrick_testing/refs/heads/main/{nm}.pth')
            response.raise_for_status()

            with open(file_pth, "wb") as f:
                f.write(response.content)
        session.close()
    except Exception as e:
        print(f'Failed to retrieve model checkpoint: {e}')
        session.close()
        raise RuntimeError('Failed to get model checkpoint.')

    print("Model checkpoint download complete!")
    return (ret_pths[0], ret_pths[1], ret_pths[2], ret_pths[3], ret_pths[4], ret_pths[5])



#########################################
#########################################
### DCGAN CODE FROM TORCH DOCS KINDA ####
#########################################
#########################################
def plot_images(data_batch:list, device:torch.device, figname:str)->None:
    '''
    Function produce a plot of images.
    Takes a batch from a Dataloader, the torch device, and the figure name as arguments.
    No return value. 
    '''
    plt_dir = 'results/'
    os.makedirs(plt_dir, exist_ok=True)
    # Plot some training images
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(figname)
    plt.imshow(np.transpose(vutils.make_grid(data_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(plt_dir+''.join([char for char in figname.lower() if char.isalnum()]))
    if SHOW_PLTS:
        plt.show()

def plot_values(val1:list, val2:list, stage:str)->None:
    '''
    Function produce a plot for losses and scores.
    Takes a generator and discriminator values as lists, and a string denoting training/testing stage.
    No return value. 
    '''
    plt_dir = 'results/'
    os.makedirs(plt_dir, exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator {stage}")
    if 'loss' in stage.lower():
        plt.plot(val1,label="Generator")
        plt.plot(val2,label="Discriminator")
        plt.ylabel("Loss")
        plt.xlabel("iterations")
    # Previously used to try to score GAN testing
    # elif 'score' in stage.lower():
    #     plt.plot(val1,label="Real")
    #     plt.plot(val2,label="Fake")
    #     plt.ylabel("Score")
    #     plt.xlabel("Minibatch")

    plt.legend()
    
    filenm = plt_dir + '_'.join(stage.lower().split()) + '.jpg'
    plt.savefig(filenm)
    if SHOW_PLTS:
        plt.show()

def animated_progression(img_list:list, stage:str)->None:
    '''
    Function to visualize the training progression of generated images.
    Takes a list containing items of one or more images per step, and a string denoting training/testing stage.
    No return value.
    '''
    plt_dir = 'results/'
    os.makedirs(plt_dir, exist_ok=True)
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    try:
        ani.save(plt_dir+f'{stage.lower()}_progression.mp4')
    except (ValueError, FileNotFoundError):
        ani.save(plt_dir + f'{stage.lower()}_progression.gif', writer=animation.PillowWriter(fps=1))


def test_models(tp:SimpleNamespace, dl: DataLoader, g_checkpoint:str, d_checkpoint:str)->set:
    """
    Function to test the models against the test dataloader from given model checkpoints.
    Takes tp (dict of tunable params), dl (Dataloader for test data), and path strings to the 
    generator and discriminator checkpoints.
    Returns a set of (real_scores, fake_scores, img_list) where
    real_scores is the avg discriminator classification of real images 
    fake_scores is the avg discriminator classification of generated images
    and img_lists is a list of grids of generated images
    """
    # Create the generator
    netG = Generator(tp).to(tp.device)

    # Create the Discriminator
    netD = Discriminator(tp).to(tp.device)

    # Handle multi-GPU if desired
    if (tp.device.type == 'cuda') and (tp.num_gpu > 1):
        netG = nn.DataParallel(netG, list(range(tp.num_gpu)))
        netD = nn.DataParallel(netD, list(range(tp.num_gpu)))

    # Load saved model states
    netG.load_state_dict(torch.load(g_checkpoint, map_location=tp.device))
    netD.load_state_dict(torch.load(d_checkpoint, map_location=tp.device))

    # Set models to eval
    netG.eval()
    netD.eval()

    # Consistent fixed noise
    fixed_noise = torch.randn(64, tp.z_size, 1, 1, device=tp.device)

    # Test on data
    print('Starting testing loop...')
    real_scores = []
    fake_scores = []
    img_list = []
    with torch.no_grad():  # No gradients needed
        for i, (real_images, _) in enumerate(dl):
            real_images = real_images.to(tp.device)

            # 5. Generate fake images
            fake_images = netG(fixed_noise)


            # Uncertainty on scoring calculations, focus on visual inspection instead
            # 6. (Optional) Get Discriminator outputs
            real_score = netD(real_images).mean().item()
            fake_score = netD(fake_images).mean().item()

            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # print(f"Batch: {i}/{len(dl)-1} Real Scores: {real_score:.4f} Fake Scores: {fake_score:.4f}")

            # 7. (Optional) Save batch of fake images
            # if (i % max(1,len(dl)//5)) == 0:
            img_list.append(vutils.make_grid(fake_images.detach().cpu(), padding=2, normalize=True))

            print(f"Batch: {i}/{len(dl)-1} complete...")

    return (real_scores, fake_scores, img_list)


def train_models(tp:SimpleNamespace, dl: DataLoader)->set:
    """
    Function to test the models against the test dataloader from given model checkpoints.
    Takes tp (dict of tunable params), dl (Dataloader for training data)
    Returns a set of (G_losses, D_losses, img_list, out_pth+'generator_state.pth', out_pth+'discriminator_state.pth') where
    G_losses and D_losses are the respective generator and discriminator losses,
    img_lists is a list of grids of generated images,
    then a path to the saved generator state,
    and a path to the saved discriminator state
    """

    # Create the generator
    netG = Generator(tp).to(tp.device)

    # Create the Discriminator
    netD = Discriminator(tp).to(tp.device)

    # Handle multi-GPU if desired
    if (tp.device.type == 'cuda') and (tp.num_gpu > 1):
        netG = nn.DataParallel(netG, list(range(tp.num_gpu)))
        netD = nn.DataParallel(netD, list(range(tp.num_gpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, tp.z_size, 1, 1, device=tp.device)

    # Establish convention for real and fake labels during training
    real_label = 0.9
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=tp.learning_rate_gen, betas=(tp.beta1_gen, tp.beta2_gen))
    optimizerD = optim.Adam(netD.parameters(), lr=tp.learning_rate_dis, betas=(tp.beta1_dis, tp.beta2_dis))
    
    # Define a perceptual feature extractor
    # Preload VGG once
    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.to(tp.device).eval()
    for p in vgg.parameters(): p.requires_grad = False

    # Two perceptual extractors:
    #   - low-level: up through conv2_2  (layer 8)
    #   - mid-level: up through conv4_3  (layer 23)
    perc_low  = nn.Sequential(*list(vgg[:8])).to(tp.device).eval()
    perc_mid  = nn.Sequential(*list(vgg[:23])).to(tp.device).eval()

    lambda_low = tp.lambda_low   # weight for low-level features
    lambda_mid = tp.lambda_mid  # weight for mid-level features

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(tp.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dl, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(tp.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=tp.device)
            # Forward pass real batch through D
            output = netD(real_cpu + torch.randn_like(real_cpu)*tp.noise_std).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, tp.z_size, 1, 1, device=tp.device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output_real = netD(fake).view(-1)
            adv_loss = criterion(output_real, label)

            # Multi-scale perceptual losses
            # resize to 64×64 for VGG
            real_rs = nn.functional.interpolate(real_cpu, size=64, mode='bilinear', align_corners=False)
            fake_rs = nn.functional.interpolate(fake,     size=64, mode='bilinear', align_corners=False)

            # low-level features (edges, textures)
            real_low  = perc_low(real_rs).detach()
            fake_low  = perc_low(fake_rs)
            loss_low  = nn.functional.l1_loss(fake_low,  real_low)

            # mid-level features (shapes, structure)
            real_mid  = perc_mid(real_rs).detach()
            fake_mid  = perc_mid(fake_rs)
            loss_mid  = nn.functional.l1_loss(fake_mid,  real_mid)

            perc_loss = lambda_low * loss_low + lambda_mid * loss_mid

            # Total generator loss
            errG = adv_loss + tp.lambda_perc * perc_loss

            errG.backward()
            D_G_z2 = output_real.mean().item()
            optimizerG.step()

            # Output training stats
            if (i % max(1,len(dl)//2) == 0) or i == len(dl)-1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch+1, tp.num_epochs, i+1, len(dl),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % max(1,(len(dl)*len(data[0]))//25) == 0) or ((epoch == tp.num_epochs-1) and (i == len(dl)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    out_pth = 'results/'
    os.makedirs(out_pth, exist_ok=True)
    torch.save(netG.state_dict(), out_pth+'generator_state.pth')
    torch.save(netD.state_dict(), out_pth+'discriminator_state.pth')
    return (G_losses, D_losses, img_list, out_pth+'generator_state.pth', out_pth+'discriminator_state.pth')

def generate_images(tp:SimpleNamespace, g_checkpoint:str, num_images=64, output_dir='results/generated')->None:
    """
    Generate and save individual samples from a trained Generator.

    Args:
        tp (SimpleNamespace): Params namespace with attributes:
            - z_size (int): latent vector dimension
            - device (torch.device)
        g_checkpoint (str): model checkpoint direction
        num_images (int): Number of images to generate and save.
        output_dir (str): Folder to save the generated images into.

    Usage:
        netG.load_state_dict(torch.load('G.pth'))
        netG.eval()
        generate_images(netG, tp, num_images=100)

    No return value
    """
    # Make sure empty output directory exists
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print("Removed previously generate images!") 
    os.makedirs(output_dir, exist_ok=True)

    # Create the Generator
    netG = Generator(tp).to(tp.device)

    # Handle multi-GPU if desired
    if (tp.device.type == 'cuda') and (tp.num_gpu > 1):
        netG = nn.DataParallel(netG, list(range(tp.num_gpu)))

    # Load saved model states
    netG.load_state_dict(torch.load(g_checkpoint, map_location=tp.device))
    
    # Generate latent vectors and images
    netG.eval()
    with torch.no_grad():
        z = torch.randn(num_images, tp.z_size, 1, 1, device=tp.device)
        fake = netG(z).cpu()

        # Rescale to [0,1]
        fake = (fake + 1.0) / 2.0

        # Save each image individually
        for idx, img in enumerate(fake):
            vutils.save_image(img, os.path.join(output_dir, f"gen_{idx:04d}.png"))

    print(f"[INFO] Saved {num_images} images to '{output_dir}/'")
#########################################
#########################################
#########################################
#########################################

# ==========================================
# 2. HAM10000 - Classifier Pipeline
# ==========================================
            
# Download and preprocess HAM10000 dataset
def download_ham10000(destination="data/HAM10000")->None:
    """
    Downloads and extracts the HAM10000 dataset from Kaggle.

    Args:
        destination (str): Directory where the dataset will be extracted.

    No return value
    """
    os.makedirs(destination, exist_ok=True)

    kaggle_key = {"username":"","key":""}  #HIDDEN
    with open(f"{os.getcwd()}/kaggle.json", "w+") as f:
        f.write(json.dumps(kaggle_key))

    # Set Kaggle API token environment variable if needed
    if os.path.exists("kaggle.json"):
        os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
    # Otherwise ensure the user has one in ~/.kaggle/
    elif not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise FileNotFoundError(
            "Missing kaggle.json: please place it in this directory "
            "or in ~/.kaggle/kaggle.json for Kaggle API authentication."
        )

    meta_pth = os.path.join(destination, "HAM10000_metadata.csv")
    if not os.path.exists(meta_pth):
        print("Downloading HAM10000 dataset from Kaggle...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000",
            "-p", destination, "--unzip"
        ], check=True)
        print("Download and extraction complete.")
    else:
        print("HAM10000 dataset already exists.")

            
def preprocess_ham10000(image_dir1:str, image_dir2:str, metadata_path:str)->pd.DataFrame:
    """
    Preprocess the HAM10000 metadata to include image paths and numeric labels.
    
    Args:
        image_dir1 (str): Path to part_1 images.
        image_dir2 (str): Path to part_2 images.
        metadata_path (str): Path to metadata CSV file.

    Returns:
        pd.DataFrame: DataFrame with image paths and encoded labels.
    """
    df = pd.read_csv(metadata_path)
    df['path'] = df['image_id'].apply(
        lambda x: os.path.join(image_dir1, x + '.jpg')
        if os.path.exists(os.path.join(image_dir1, x + '.jpg'))
        else os.path.join(image_dir2, x + '.jpg')
    )
    return df


class SkinLesionDataset(Dataset):
    """
    Custom dataset for HAM10000 skin lesion images.

    Args:
        dataframe (pd.DataFrame): DataFrame containing metadata and image paths.
        transform_dict (dict): Mapping of artifact group to transform.
        label_map (dict): Mapping of class labels to integer indices.
    """
    def __init__(self, dataframe:pd.DataFrame, transform:transforms.Compose, label_map:dict)->None:
        '''
        Init function for SkinLesionDataset class.
        Takes a dataframe, and composed transform object, and a dict.
        No return value
        '''
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self)->int:
        '''
        Wrapper to get the len of the df
        '''
        return len(self.df)

    def __getitem__(self, idx:int)->tuple:
        '''
        wrapper to get item at idx from the df
        Returns an Image and a str in a list
        '''
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = self.label_map[row['dx']]
        image = self.transform(image)
        return image, label

def create_dataloaders(df:pd.DataFrame, batch_size=32, use_synthetic=False)->tuple:
    """
    Creates DataLoaders for training and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing metadata for real images.
        batch_size (int): Batch size for DataLoaders.
        use_synthetic (bool): Whether to include synthetic images in training set.
    
    Returns:
        tuple: (train_loader, test_loader, label_map)
    """
    if use_synthetic:
        synthetic_path = 'results/generated/'
        if os.path.exists(synthetic_path):
            synthetic_files = [f for f in os.listdir(synthetic_path) if f.endswith('.jpg') or f.endswith('.png')]
            if synthetic_files:
                synthetic_df = pd.DataFrame({
                    'image_id': [f.split('.')[0] for f in synthetic_files],
                    'dx': ['mel'] * len(synthetic_files),
                    'path': [os.path.join(synthetic_path, f) for f in synthetic_files]
                })
                df = pd.concat([df, synthetic_df], ignore_index=True)
                print(f"[INFO] Added {len(synthetic_df)} synthetic samples.")
            else:
                print("[WARN] Synthetic folder found, but no valid image files were loaded.")
        else:
            print("[WARN] Synthetic path not found. Skipping synthetic images.")

    label_map = {label: idx for idx, label in enumerate(sorted(df['dx'].unique()))}

    # Split dataset into train and test set
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['dx'], random_state=42)

    # Create validation split from train set
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['dx'], random_state=42)

    #Augmentated transformation
    augment_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    standard_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    train_dataset = SkinLesionDataset(train_df, augment_transform, label_map)
    val_dataset = SkinLesionDataset(val_df, standard_transform, label_map)
    test_dataset = SkinLesionDataset(test_df, standard_transform, label_map)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        label_map
    )

#########################################
#########################################
#########################################
#########################################

def build_resnet18_model(num_classes=7)->torch.nn.Module:
    """
    Builds a modified ResNet18 model for HAM10000 classification in PyTorch.

    Args:
        num_classes (int): Number of output classes (default is 7 for HAM10000).
    
    Returns:
        model (torch.nn.Module): Modified ResNet18 model.
    """
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
    )
    return resnet
            
            
def train_classifier(tp:SimpleNamespace, model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, criterion:nn.Module, optimizer:torch.optim.Optimizer)->dict:
    """
    Train a classifier with early stopping.

    Args:
        model (nn.Module): The neural network to train.
        tp: (SimpleNamespace): Namespace containing tunable parameters.
        train_loader (DataLoader): Dataloader for the training set.
        val_loader (DataLoader): Dataloader for the validation set.
        criterion (nn.Module): Loss function, e.g. nn.CrossEntropyLoss().
        optimizer (Optimizer): Optimizer for updating model parameters.

    Returns:
        dict: A history dictionary containing:
            - 'train_losses': list of training losses per epoch
            - 'train_accuracies': list of training accuracies per epoch
            - 'val_losses': list of validation losses per epoch
            - 'val_accuracies': list of validation accuracies per epoch
            - 'best_model_state': state_dict of the best‐performing model
    """
            
    model = model.to(tp.device)
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(tp.clf_epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(tp.device), labels.to(tp.device)
            optimizer.zero_grad()
            #print(">> images.shape:", images.shape)
            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        train_acc  = correct / total
        train_accuracies.append(train_acc)

        val_loss, val_acc = eval_classifier(tp, model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{tp.clf_epochs} - Train Loss: {train_loss:.4f}| Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= tp.clf_patience:
                print("[INFO] Early stopping triggered.")
                break

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_model_state': best_state
    }

def eval_classifier(tp:SimpleNamespace, model:nn.Module, dataloader:DataLoader, criterion:nn.Module)->tuple:
    """
    Evaluate a trained classifier on a given dataset.

    Args:
        tp: (SimpleNamespace): Namespace containing tunable parameters.
        model (nn.Module): The trained model.
        dataloader (DataLoader): Dataloader for evaluation data.
        criterion (nn.Module): Loss function to compute evaluation loss.

    Returns:
        tuple:
            - avg_loss (float): Average loss over the dataset.
            - accuracy (float): Classification accuracy over the dataset.
    """
    model.eval()
    running_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(tp.device), labels.to(tp.device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), correct / total


def plot_classifier(history:dict, use_synth:bool)->None:
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history (dict): Dictionary returned by train_classifier containing:
            - 'train_losses'
            - 'val_losses'
            - 'train_accuracies'
            - 'val_accuracies'
        use_synth (bool): T/F value specifying whether synthetic images were included
    Side Effects:
        - Saves the figure as 'classifier_learning_curves.png'.
        - Shows the plot.
    """
    plt_dir = 'results/'
    os.makedirs(plt_dir, exist_ok=True)
    

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Accuracy')
    plt.plot(history['val_accuracies'], label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    
    plt.tight_layout()
    name_addition = ""
    if use_synth:
        name_addition = "_synthetic"
    plt.savefig(plt_dir + f'classifier_learning_curves{name_addition}.png')
    if SHOW_PLTS:
      plt.show()


def final_classification_metrics(model:nn.Module, dataloader:DataLoader, label_map:dict, use_synth:bool)->None:
    """
    Compute and display final classification metrics on a test set.

    Args:
        model (nn.Module): The trained classifier model.
        dataloader (DataLoader): Dataloader for test data.
        label_map (dict): Mapping from class index to class name.
        use_synth (bool): T/F value specifying whether synthetic images were included

    Side Effects:
        - Prints classification report, confusion matrix, and ROC AUC score.
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs.logits
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            y_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    os.makedirs('results/', exist_ok=True)

    name_addition = ""
    if use_synth:
        name_addition = "_synthetic"

    clf_report = pd.DataFrame(classification_report(y_true, y_pred, target_names=label_map.keys(), output_dict=True)).transpose()
    clf_report.to_csv(f'results/classification_report{name_addition}.csv')
    print("Classification Report:\n",clf_report)

    conf_matrix = confusion_matrix(y_true, y_pred)
    np.savetxt(f"results/confusion_matrix{name_addition}.txt", conf_matrix, delimiter='\t', fmt='%d')
    print("Confusion Matrix:\n", conf_matrix)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_map.keys())
    disp.plot(cmap=plt.cm.Blues) # You can change the colormap
    plt.savefig(f'results/clf_confusion_matrix{name_addition}.png')
    if SHOW_PLTS:
        plt.show()

    roc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    with open(f"results/roc_auc_score{name_addition}.txt", "w+") as f:
        f.write(f"ROC AUC OvR Score: {roc}")
    print(f"ROC AUC OvR Score: {roc}")

#########################################
#########################################
#########################################
#########################################



def main(args=None)->None:
    '''
    Main function of the script, takes the command line args as a parameter.
    Directs dataflow and code execution.
    No return value.
    '''
    if not any([args.train, args.test, args.testlocal, args.debug]):
        raise ValueError('Missing required command line arguments')
    if args.train and (args.test or args.testlocal):
        raise ValueError('Cannot set both train and test flags. Testing is done automatically after training.')
    if (args.test and args.testlocal):
        raise ValueError('Cannot set both test and testlocal flags. Checkpoints must either be github or local, not both.')
    
    # Set show_plts flag value
    global SHOW_PLTS
    SHOW_PLTS = args.show
    
    # For reproducibility
    manualSeed = 888
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    # Set device for use in torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    ##########################################
    # Master dict to hold tunable params
    ##########################################
    param_dict = {
        'workers': 2,
        'batch_size': 32,
        'image_size': 64,
        'num_channels': 3,
        'z_size': 128, #  Size of z latent vector (i.e. size of generator input)????
        'gen_featuremap_sz': 150, #128
        'dis_featuremap_sz': 64, #64
        'num_epochs': 1000,
        'learning_rate_gen': 9e-4, 
        'learning_rate_dis': 1e-4,
        'beta1_gen': 0.5,
        'beta2_gen': 0.999,
        'beta1_dis': 0.5,
        'beta2_dis': 0.999,
        'num_gpu': 1,
        'noise_std': 0.01,
        'lambda_perc': 1.0,
        'lambda_low': 0.5,
        'lambda_mid': 0.5,
        'clf_epochs': 100, 
        'clf_patience': 10,
        'clf_lr': 1e-5,
        'clf_weight_decay': 7e-6,
        'clf_batch_size': 32,
        'device': device
    }
    tp = SimpleNamespace(**param_dict) # TP = Tunable Params
    ##########################################
    # Note: Can also adjust image transforms in build_dataloader()
    ##########################################
    
    # Automatically download the Fitzpatrick17k dataset
    csv_pth = ''
    img_pth = ''
    redownload = args.redownload
    if args.fresh:
        threshold = args.fresh
        csv_pth, img_pth = generate_dataset(min_f_score=threshold, redownload=redownload)
    else:
        csv_pth, img_pth = download_dataset(redownload=redownload)

    if csv_pth == -1 or img_pth == -1:
        raise RuntimeError('The script was unable to download the dataset. Review the terminal outputs for more information.')

    # Create dataloaders
    training_dataloader, testing_dataloader = build_dataloaders(img_dir=img_pth, batch_sz=tp.batch_size, worker_num=tp.workers,img_sz=tp.image_size)
    # Train the model if train flag
    if args.train or args.traingan:
        # Plot the first batch of real training images
        data_iter = next(iter(training_dataloader))
        plot_images(data_batch=data_iter, device=tp.device, figname='Sample Real Training Images')

        # Train the models
        g_losses, d_losses, gen_imgs, g_checkpoint, d_checkpoint = train_models(tp=tp, dl=training_dataloader)
        plot_values(val1=g_losses, val2=d_losses, stage='Training Loss')
        animated_progression(img_list=gen_imgs, stage='Training')
        plot_images(data_batch=gen_imgs, device=tp.device, figname='Final Synthetic Images Training')

    # Download model checkpoint if test flag
    if args.test:
        if args.use_synth:
            g_checkpoint, d_checkpoint, _, _, clf_checkpoint, clf_history = get_model_checkpt(redownload=redownload)
        else:
            g_checkpoint, d_checkpoint, clf_checkpoint, clf_history, _, _ = get_model_checkpt(redownload=redownload)

    elif args.testlocal:
        g_checkpoint = 'results/generator_state.pth'
        d_checkpoint = 'results/discriminator_state.pth'

    real_scores, fake_scores, gen_imgs = test_models(tp=tp, dl=testing_dataloader, g_checkpoint=g_checkpoint, d_checkpoint=d_checkpoint)
    # Commented out due to uncertainty in score calculations
    # Basing testing capability on vision inspection insteaf
    # plot_values(val1=real_scores, val2=fake_scores, stage='Testing Scores')
    animated_progression(img_list=gen_imgs, stage='Testing')
    plot_images(data_batch=gen_imgs, device=tp.device, figname='Final Synthetic Images Testing')

    # Automatically download the HAM10000 dataset
    download_ham10000()
    df_ham = preprocess_ham10000(
        image_dir1="data/HAM10000/HAM10000_images_part_1",
        image_dir2="data/HAM10000/HAM10000_images_part_2",
        metadata_path="data/HAM10000/HAM10000_metadata.csv"
    )

    if args.use_synth:
        generate_images(tp, g_checkpoint=g_checkpoint, num_images=1000, output_dir='results/generated')   

    # Classifier loaders (with synthetic if requested)
    tr_dl, val_dl, te_dl, label_map = create_dataloaders(
        df=df_ham,
        batch_size=tp.clf_batch_size,
        use_synthetic=args.use_synth
    )

    clf_dir = 'results/'

    # ResNet18: train or load
    clf = build_resnet18_model(num_classes=len(label_map)).to(tp.device)
    
    # Train the model if train flag
    if args.train or args.trainclf:
        clf_history = train_classifier(
            tp=tp,
            model=clf,
            train_loader=tr_dl,
            val_loader=val_dl,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(clf.parameters(), 
                                 lr=tp.clf_lr, 
                                 weight_decay=tp.clf_weight_decay, 
                                 betas=(0.9, 0.999)),
        )
        if args.use_synth:
            torch.save(clf_history['best_model_state'], clf_dir+"clf_state.pth")
            torch.save(clf_history, clf_dir+"clf_history.pth")
        else:
            torch.save(clf_history['best_model_state'], clf_dir+"clf_baseline_state.pth")
            torch.save(clf_history, clf_dir+"clf_history_baseline.pth")
    elif args.test:
        clf.load_state_dict(torch.load(clf_checkpoint, map_location=device))
        clf_history = torch.load(clf_history)
    elif args.testlocal:
        try:
            if args.use_synth:
                clf.load_state_dict(torch.load(clf_dir+"clf_state.pth", map_location=device))
                clf_history = torch.load(clf_dir+"clf_history.pth")
            else:
                clf.load_state_dict(torch.load(clf_dir+"clf_baseline_state.pth", map_location=device))
                clf_history = torch.load(clf_dir+"clf_history_baseline.pth")

        except Exception as e:
            print('Failed to load CLF state with set flags:\n',e)
            return
        
            
    plot_classifier(clf_history, args.use_synth)

    # Final evaluation on test set
    final_classification_metrics(clf, te_dl, label_map, args.use_synth)    

if __name__ == '__main__':
    # Collect command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--fresh',
                        metavar='Threshold',
                        type=int,
                        help='Create fresh data subset from scratch for Fitzpatrick scale >= x')
    parser.add_argument('--redownload',
                        action='store_true',
                        help='Delete local files and redownload target data source')
    parser.add_argument('--train',
                        action='store_true',
                        help='Train the model')
    parser.add_argument('--traingan',
                        action='store_true',
                        help='Train only the GAN system')
    parser.add_argument('--trainclf',
                        action='store_true',
                        help='Train only the classifier system')
    parser.add_argument('--test',
                        action='store_true',
                        help='Test saved model checkpoint from github')
    parser.add_argument('--testlocal',
                        action='store_true',
                        help='Test saved model checkpoint from local training runs')
    parser.add_argument('--use_synth',
                        action='store_true',
                        help='Use synthetic images for classifier training')
    parser.add_argument('--use_baseline',
                        action='store_true',
                        help='Do not use synthetic images for classifier training')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show plots and graphs')
    
    parser.add_argument('--debug',
                        action='store_true',
                        help='Used to trigger whatever I need')
    
    args = parser.parse_args()

    if args.use_baseline and args.use_synth:
        raise ValueError("Cannot set both baseline and synth flags.")
    elif not args.use_baseline: # use_synth default true
        args.use_synth = True

    main(args)
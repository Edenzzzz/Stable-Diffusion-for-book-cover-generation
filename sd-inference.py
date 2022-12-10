# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 70267, "status": "ok", "timestamp": 1670648335832, "user": {"displayName": "Eden Tan", "userId": "11491147198112718283"}, "user_tz": 360} id="Qcr_2f8gQs0G" outputId="c025ffe1-105e-4ed8-b367-b4ff85ff9d42"
#@title Install the required libs
# %pip install -qq diffusers["training"]==0.7.2
# %pip install -qq transformers==4.24.0 ftfy
# %pip install -qq "ipywidgets>=7,<8"
# %pip install wandb
# %pip install kornia
# %pip install bitsandbytes
#0.10 version doesn't contain login()
# %pip install huggingface_hub==0.11.1

#deepspeed
# # !pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade
# # !pip install deepspeed==0.7.4 --upgrade
# # !pip install diffusers==0.6.0 triton==2.0.0.dev20221005 --upgrade
# # !pip install transformers[sentencepiece]==4.24.0 accelerate --upgrade

# + id="rpElaJI0Zhsd" executionInfo={"status": "ok", "timestamp": 1670648339964, "user_tz": 360, "elapsed": 1120, "user": {"displayName": "Eden Tan", "userId": "11491147198112718283"}}
# %pip freeze>requirements.txt

# + colab={"base_uri": "https://localhost:8080/"} id="ZtZzUJVdZzLi" executionInfo={"status": "ok", "timestamp": 1670648414303, "user_tz": 360, "elapsed": 249, "user": {"displayName": "Eden Tan", "userId": "11491147198112718283"}} outputId="4873cdc5-cf75-4e51-a6d6-d4049b06e15b"
# !python --version

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 16590, "status": "ok", "timestamp": 1670617239598, "user": {"displayName": "Eden Tan", "userId": "11491147198112718283"}, "user_tz": 360} id="zwvU-RUBQvcN" outputId="c4d1d93c-adaf-45a6-f7d6-4d250846bdad"

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1984, "status": "ok", "timestamp": 1670617241578, "user": {"displayName": "Eden Tan", "userId": "11491147198112718283"}, "user_tz": 360} id="kVhLWX98Qs0J" outputId="2c3538a6-a7af-4b58-d6c1-ccce7a1a9eb0"
#@title Login to the Hugging Face Hub
#@markdown Add a token with the "Write Access" role to be able to add your trained concept to the [Library of Concepts](https://huggingface.co/sd-concepts-library)
from huggingface_hub import login
login("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")
# notebook_login()
# from google.colab import drive
# drive.mount("/content/drive",force_remount=True)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11618, "status": "ok", "timestamp": 1670617253194, "user": {"displayName": "Eden Tan", "userId": "11491147198112718283"}, "user_tz": 360} id="ZNincXUlQs0K" outputId="d7c71e46-3207-49f1-9074-5d481b7d7eb6"
#@title Import required libraries
# # %pip install protobuf==3.20.* #For deepspe

import argparse
import itertools
import math
import os
import random
import numpy as np
import torch,torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torchvision
import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler,DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer,TrainingArguments
import kornia.augmentation as K#augmentaiton
import pandas as pd
import wandb
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("--version",type=str,help="wandb model version, e.g. v1",required=True)
parser.add_argument("--run_id",type=str,help="wandb run id of model",required=True)
parser.add_argument("--data_root",default="./book dataset",help="path to read csv files")
parser.add_argument("--batch_size",default=3,type=int,help="Generation batch size. For a GPU with 16gb memory, 4 is maximum.")
parser.add_argument("--save_for__fid",default=False,help="whether to generate and save more images for FID score evaluation")
parser.add_argument("--num_imgs",type=int,default=4000,help="number of images to generate for computing FID score. Only to be specified if save_for_fid is True")
parser.add_argument('--save_dir',type=str,default="./Output_images",help="Output dir for generated images.")
parser.add_argument("--delete_model",type=bool,default=True,help="whether to delete downloaded model artifact to save storage")
args = parser.parse_args()
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
#For reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
global_seed=42
set_seed(global_seed)

# + id="BjKr2arpQs0K"
#@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
#pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" #@param {type:"string"}

# + id="34721mAdQs0L"
#@title Setup the prompt templates for training 
book_cover_templates=[#the first entry is for "highly legible text"
    "A {} book cover with author: {}, book title: {} ",
    #repeat some prompts to give model prior knowledge about book cover styles
    "A {} book cover written by author: {} with book title: {} ",
#     "A {} simple book cover with author {}, book title {} ",
#     "A plain {} book cover with author {}. The book title is{} ",
#     "A {} vivid book cover with author {}, book title {} ",
    "A  {} book cover with author name:{}, book title: {}",
# #     "We are going to create a clear, {}, highly detailed book cover with author named {}, and book title is '{}'",
#     "An intricate {}, book cover including book author:{}, book title: '{}'",
#     "A detailed, {}, book cover with {} ,written by author {}",
#     "A creative, colorful {}, book cover written by {}. The book title is {}, ",
#     "A {} old-fashioned, plain book cover written by {}. The book title is {}",
#     "A simple, {}, old-fashioned book cover with author name {}, book title {} ",
#     "A simple, {}, plain book cover with author name {}, book title {} ",
    "A detailed {} book cover with author: {} and book title: {} "
    
]
#TODO: add more to match the number of templates
summary_placeholders=[
    ', and abstract {}',
    ",summary {}",
    ", the book describes that {}",
    ", book discription {}",
    ", main story {}",
    ", the book is mainly about {}",
    ", and main story {}",
    "and book abstract {}",
    ", and book description {}"
]
test_templates=[#the first entry is for "highly legible text"
    "A {} book cover with author: {}, book title: {} ",
    #repeat some prompts to give model prior knowledge about book cover styles
    "A {} book cover written by author: {} with book title: {} ",
    "A {} simple book cover with author: {}, book title: {} ",
    "A {} vivid, fantastic book cover with author: {}, book title: {} ",
#     "We are going to create a clear, {}, highly detailed book cover with author named {}, and book title is '{}'",
    "An intricate {}, book cover including book author:{}, book title: '{}'",
    "A detailed, {}, book cover written by author: {}, with title:{}",
    "A creative, colorful {}, book cover written by: {}. The book title is: {}, ",
    "A {} old-fashioned, plain book cover written by: {}. The book title is: {}",
    "A simple, {}, old-fashioned book cover with author name: {}, book title: {} ",
    "A cartoon-styled, entertaining book cover with author name: {}, book title: {}"
]


#pad to the same length 
for i in range(len(summary_placeholders),len(test_templates)):
  summary_placeholders+=[random.choice(summary_placeholders)]
summary_placeholders=summary_placeholders[:len(test_templates)]

#fix random seed by fixing latents
latents=None
def visualize_prompts(
    pipeline: StableDiffusionPipeline,
    summerize=False,
    include_desc=False,
    max_length=15,
    legible_prompt=True,
    samples_per_prompt=4,
    img_size=512,
    inference_steps=75,
    save_to_drive=False,
    save_dir=None,
    batch_generate=True
    ):
    
    """
    Visualizes the output of the given StableDiffusionPipeline for the given test prompts.
    Args:
    - pipeline: an instance of StableDiffusionPipeline
    - summerize (bool, optional): Whether to summerize the book description. Default is False.
    - include_desc (bool, optional): Whether to include the book description in the prompt. Default is False.
    - max_length (int, optional): The maximum length of the summerized description. Only used when summerize=True.
    - legible_prompt (bool, optional): Whether to add "legible text" to the prompt. Default is True.
    - samples_per_prompt (int, optional): The number of samples to generate for each prompt. Default is 4.
    - img_size (int, optional): The output image size. Default is 512.
    - inference_steps (int, optional): The number of denoising steps. The bigger the less noisy. Default is 75.
    - save_to_drive (bool, optional): Whether to save the generated images to Google Drive. Default is False.
    - save_dir (str, optional): The path to the directory where the generated images should be saved. Only used if save_to_drive=True.
    - batch_generate (bool, optional): Whether to speed up generation by generating in batches. Default is True.

    Returns:
    - None

    Note:
    - If include_desc is True, batch_generate will be set to False because passing stacked descriptions of different length to the model will cause an error.

    Examples:
    ```
    pipeline = StableDiffusionPipeline()
    visualize_prompts(pipeline)
    ```
    This will generate and display 3 samples for each of the test prompts using the default settings.
    ```
    visualize_prompts(pipeline, summerize=True, include_desc=True, max_length=20)
    ```
    This will generate and display 3 samples for each of the test prompts, where each prompt includes a summerized book description with a maximum length of 20.
    ```
    visualize_prompts(pipeline, save_to_drive=True, save_dir="generated_images")
    ```
    This will generate and save 3 samples for each of the test prompts to the Google Drive directory "generated_images".
    ```
    visualize_prompts(pipeline, summerize=True, include_desc=True, max_length=20, batch_generate=False)
    ```
    This will generate and display 3 samples for each of the test prompts, where each prompt includes a summerized book description with a maximum length of 20. The images will be generated one by one instead of in batches.
    """

    if summerize==True:
      assert include_desc==True, "include_desc is False, \
      no summerization can be done without book description!" 

    assert save_dir and save_to_drive and os.path.isdir(save_dir), "Must specify save_to_drive=True and save_dir with a valid dir"
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated()
    
    #fix random seed by fixing latents.
    #generate fixed latents if no latents exist
    global latents
    if latents==None or latents.shape[0]!=samples_per_prompt:
      generator = torch.Generator(device='cuda')
      generator = generator.manual_seed(global_seed)
      latents=torch.zeros(samples_per_prompt,
                           pipeline.unet.in_channels,img_size // 8, img_size // 8)
      for j in range(samples_per_prompt):
        latents[j,:,:,:] = torch.randn(
            (pipeline.unet.in_channels, img_size // 8, img_size // 8),
            generator = generator,
            device = 'cuda'
        )


    import matplotlib.pyplot as plt,random
    #generate from test prompts only
    df=pd.read_csv(args.data_root+"/df_test.csv")

    #set up figures
    dpi=plt.figure().dpi
    fig,axes=plt.subplots(len(test_templates),
                          samples_per_prompt,
                          figsize=(img_size/dpi*samples_per_prompt,
                                    img_size/dpi*len(test_templates))
                          )
    fig.subplots_adjust(wspace=0, hspace=0)#combind with axes[i][j].set_aspect('auto'); remove spacing
    # plt.suptitle(,y=0.89)

    ###fix random seed by fixing latents
    if include_desc:
      from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
      tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
      if summerize:
          model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
      


    ### Get model output 
    for i in range(len(test_templates)):
      text=[]
      for j in range(samples_per_prompt):
        row=df.iloc[j]
        legible_text,author,title,description = ("",row['book_authors'], row['book_title'], row['book_desc'])

        if legible_prompt:
            legible_text="legible text"
        if summerize:
            torch.cuda.empty_cache()
            inputs = tokenizer(description, max_length=1024, 
                                return_tensors="pt",truncation=True,padding="max_length")
            summary_ids = model.generate(inputs['input_ids'], num_beams=3,\
                                          min_length=2, max_length=max_length)
            description = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, 
                                          clean_up_tokenization_spaces=False)[0]#batch_decode returns a list of strings; here len(list)=1, only one input string
            torch.cuda.empty_cache()
        ###get prompt
        template=test_templates[i]
        if include_desc:
          template+=summary_placeholders[i]#append new prompt to list
          template=template.format(legible_text,author,title,description)
          text += [template]
        else:
          text += [template.format(legible_text,author,title)]


      ###inference 
      from torch import autocast
      images=[]
      print(f"Inference iteration {i}")

      with autocast("cuda"):
        if batch_generate:#batch generation
          index = 0
          while index < len(text):
            images+=pipeline(text[index:index+args.batch_size],height=img_size,width=img_size,
                            num_inference_steps=50, guidance_scale=7.5,
                            latents=latents[index:index+args.batch_size]).images
            index = index+args.batch_size
        else:#To avoid out of memory, generate one at a time
          for j in range(samples_per_prompt):
            images += pipeline(text[j],height=img_size,
                              width=img_size,num_inference_steps=inference_steps, 
                              guidance_scale=7.5,latents=latents[None,j]).images
                              
      try:
        axes[i][0].set_title(f"Prompt {i}, legible={legible_prompt},summerize={summerize},include_desc={include_desc}")
        for j in range(samples_per_prompt):
            axes[i][j].imshow(images[j])
            axes[i][j].set_aspect('auto')#remove spacing
      ###single plot case
      except:
        axes[i].set_title(f"Prompt {i}, legible={legible_prompt},summerize={summerize},include_desc={include_desc}")
        #debug
        print(images[0])
        print("images:",images)
        axes[i].imshow(images[0])
        axes[i].set_aspect('auto')

    if save_to_drive:  
      ###save fig with paramters
      img_name=f"summerize={summerize},\
                include_desc={include_desc}.png"
      path=os.path.join(save_dir,img_name)
      plt.savefig(path)
      fig.show()
    
    ###save checkpoint generation results in wandb
    img_path="checkpoint_image_sample.jpg"
    plt.savefig(img_path)
    fig.show()
    image=Image.open(img_path)
    wandb.log({"examples":wandb.Image(image)})
    subprocess.run(["rm", "checkpoint_image_sample.jpg"])


### Fine tune result evaluation
wandb_model = "stable_diffusion_model:"+args.version
noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
run_id = args.run_id

#load from wandb checkpoint
os.environ["WANDB_API_KEY"]="16d21dc747a6f33247f1e9c96895d4ffa5ea0b27"
#can't use artifact in offline mode
#     os.environ['WANDB_MODE'] = 'online'
#     wandb.init(id=run_id,resume="must") 
wandb.init(project="book_cover_generation",id=run_id,name="stable_diffusion "+wandb_model.split(":")[-1]+"+inference",resume='must')
my_model_artifact = wandb.run.use_artifact(wandb_model)
  # Download model weights to a folder and return the path
model_dir = my_model_artifact.download()

  # Load your Hugging Face model from that folder
  #  using the same model class
tokenizer = CLIPTokenizer.from_pretrained(
    model_dir,
    subfolder="tokenizer",
    use_auth_token=True,
    Padding="max_length",
    Truncation=True,
  )
pipeline = StableDiffusionPipeline.from_pretrained(
      model_dir,
      torch_dtype=torch.float16,
      safety_checker=None,
      scheduler = noise_scheduler,
      tokenizer = tokenizer#enable padding
      ).to('cuda')
#delete downloaded model to save storage
if args.delete_model:
  subprocess.run(["rm", "-r","artifacts"])

print(f'Load {wandb_model} from wandb cloud checkpoint')
if os.path.isdir(args.data_root+"/"+wandb_model.split(":")[-1]+" inference"):
  print("Save dir already exists.")
save_dir = args.save_dir+"/"+wandb_model.split(":")[-1]+" inference"
os.makedirs(save_dir,exist_ok=True)
print(f"Visualization results will be saved in {save_dir}")

# # + colab={"background_save": true} id="u3FupDFgQs0M"
# visualize_prompts(pipeline,summerize=False,samples_per_prompt=4,
#                   include_desc=False,legible_prompt=False,
#                   batch_generate=True,save_to_drive=True,
#                   save_dir=save_dir)

# # + id="Temfe6g7Qs0N"
# visualize_prompts(pipeline,summerize=True,include_desc=True,
#                   samples_per_prompt=4,
#                   legible_prompt=False,save_to_drive=True,
#                   save_dir=save_dir)

# + id="SkOKrdIeQs0N" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["99cd0bee284948678623bea419d45f0e", "c2fb567bfa2c42d28f18ed6246234652", "78459a35cd424407a8b592eeebe114eb", "f8e2f17f944840fba71ce60aca09db93", "bfa2bb592cd74a86902e9efc0be0cde9", "6758e09338964bad8592275c060f42e0", "de76a604132a4143b7bdac70027bd943", "8ce453f18aba401fb8d0db74ff0bd9c4", "12a15899ddff4a10af575d51982a884a", "cbf7fc89354d44ddada3a9d6e250d27b", "254cd6cfdc7e45ed8f35dda199cd4441", "f7cf3ee8080b494cb4b4674dd879d941", "8ccb335cae24419f90c34d3485ec21c6", "95f1183a6a13423780443ffef57d982e", "5fd1d8c9a8d54002b52f14cb00d65fbe", "c1647a61db024cc3885a13dff7d5fff4", "f9df4afd7607434f985800aeb12b5d8d", "f2187452d2e644cba3e40794a7c388a4", "991708cc8e574383a65492e349b79e16", "b7553856128c4cad9a393e909a46ea47", "e6aa4917dbe0496ca61616289a39f0d4", "7ff9757724994d09af4384b6c5bfe78e", "3c0c7b09f8d541d8a36361d9ffff258c", "7f68fbe3256f4fdf82516c9c87249a39", "f8f99fe40a744eed83f53a43acc37299", "d7261586a4794fda81bc032bcb4eee3e", "2d6fabefcb4244e28262830625e4872d", "0ffc5b138dae43c6a47dc1d0d7e44df9", "34c491c4fd4d474f937986e634543f03", "77314ef4c13342a0b2122a8fa842f4e9", "0dfab41c77d04062a65a6bd15a93b846", "6e6896329d974827afd87bb384496353", "c1e9b20329894398b3e32cf52fae9515", "c657a6560f924a7e97a3640f8da889a4", "85b0331887f5414fa2af9f3d8185ad3f", "eaf94b08617b49d99cf3b728e1880e5f", "def60d7c078e4063ba0c44d4586399db", "23d4ac9d7aca40f98eaa8c5dd4064308", "f8ae71af06af4468aa02f5bfadd2c5d8", "d98bdfadf66e43808544d2d1d6c6c89a", "d3b88b207ca5479187101a56f94b9615", "c1b4d8f5021745dc84269c91ba38d1a5", "e5903818192242eebb1494ffb48299f9", "974e70a3e7234a85a801d809a2547d13", "d4f78488c4e447ae8b7783a548f065c5", "013798a1b9844bea9231c2f45b933ae9", "41004f45fae241bc99bc6bc667745826", "d185c4c02d5f43389954a900b5512e18", "1a51890beb74483396c3c3bc318f1ef8", "6212cd4390ef42a580328ee671abf577", "036c0db177064176bd9f5165cfcde019", "60c51471d0a6438aa0964853df76409e", "9e195c19344f48be9752fccfcd950e5a", "afd562ec05ff45c89fff2d50c0fbbd2f", "ce0451310d3e48de8a568b45d00eb358", "a952978ab3e144ddb6b62b3a2c9c94bd", "9ea63a656b214e6c95daf9ccb07c6f46", "d704f186b951424bb243d12315c6ea8d", "3af84ff9ea8f4b95b732fe633390c96d", "14f8bdeca804476490ada24d77ef06f9", "ba136bed79ba401b8a0b25d2325b67b7", "ad295321914541f7bedfb29c59dc3eee", "3ebe8641a92143549043558412de5ca0", "87c8d4c9d60741faa3b232db03bd5a7b", "fa028bc545314889bae970dbf5453714", "f515f7f75b4a4f6fa49c66a6561d74fd", "ce31fa6beb2149cbbecba37a4ca53b56", "f8a6f817db8749938d9556b19ddea8ae", "e9adada028dc4ce4bd6c5b361bc5b57c", "c0e987a708b64578b12a3609db7c8ff0", "fcab2e28f60c4306a2ffe8330f21f78a", "5c8eb8776bec42b385f8fe61d8f97d38", "ae5fe276e11a46c5bc5d1aad4f4ec0be", "23ac61b3212b4b2b8ba77b1092a757b7", "7fd316a5b49e485e9633a7c6f088905a", "ab3ffbb835984313a1d2c171c4e3b381", "f4321ce562a440ac8bfbf737925bdb32", "098858fc83e04eb1814dde5ca8f53610", "1470261394404f16962d9de93aa819fc", "abbe8558dcb34a489fddf92cebf085d3", "b417abda2c504b45bcc49cf1331563df", "8dcb7276340f4141b1097514e46b1267", "21a5afaf8b814b7fa68d4a5949e8ebd3", "77e7a507922b46f1a450e4023083901d", "f6b94ddab9224ec6bcff0dcdaaf82b3c", "0850d057e3834c0bb595adccd2df1bc0", "4de8b1981e6749f785e6d81678c17266", "709865b9bd404b72bfa923704fe99930", "dca88c38b37543ffba2f95d1394f61ea", "111d882b8bf4403ba6834956ffe097f2", "465fa830fe0547f1997eaf4900bdcfc8", "66b6f322ac9a4739bbf7b4c2fa4759ed", "f5e9041d47564f22a371e424ae4648ca", "b353517debb845d5aecaffa36b9a6c6a", "60883431cc86438580f0ee4c91e0f16f", "919a281b8d9240b0abf1fed11ef1fbac", "5c022c52a2034e0d8d18bfe6b0680532", "02e650786e064c0687eb589a9dacc6d3", "3bc823c0995d4248841e18a6756e0161", "de21c2b86dfa4f91a2b544fd3b865652", "8f171a3d072145c6831bf60b03830631", "eecb1c0075064e30815fd2a5ac62aa44", "a73f9a1cbfda49f5b06861cb51e466d8", "09bea104a4104d2d94c12260b8e57497", "37079c1fd75d4493b2b49f2ccc8e18cb", "97c59f45e49847bab8556384862162d5", "c541ff2cf59f43eba6758f144d2a80a4", "7a7b57734dba48f5a661dfaa50f34b58", "b57ab5d1efc94c13bb77362198bb9211", "b2df8378b29b4f1f8edbe637b91c91a0", "80f2221377ed40f0b03a2c6840b7881b", "365dbbcf10924470980887aa35d5d8e5", "82837cc7a8d74cc6953fd20307533a72", "4f304f86867b41ae9c7e15375628c2f3", "57cc3870c00546d08be4ee8616c4e470", "7af7971fa83e422d8be132a463c76053", "bf404665f05b4a9f94efd49fa7110bee", "481c625edbd3444a8d7ccbe84d8e02a1", "f0831ea11f2446e3a96f8484586e70d1", "c55c687928834b8a87002b9a22704fb3", "2475ae80809647a5aa14deb70f2e9ae4", "acd8bf04e0c64c4ebfb7b321cb526307", "5cf4828033754c2ea3b2dd7994ee2ff0", "8a5762de2a714ef9b02b4f9c82061c37", "460595d4e391423fbd3b1286268bfc80", "45711f8a9b144f1cb7ead4254339146d", "aaef950f0e6c4cb8b973d7ae00471af4", "e94be3d5cae5412c8a6875bc46e9be3b", "8318ee6fee8c49ec8c906d954aef98e9", "9b9c70fb55be4f01854175bc386c637f", "3331e538bf1f467bbb5c6e0d9db4d65a", "9fc79ad8e6cb4668818430446356fdfc", "48a3fcf753e449289ef32930a0aa3d26", "67747d7f5f464c6b89eaedb38a34e22d", "cd7d5544e0624d0babbc545dc1a6874f", "ac6a3f0262f04ffe80249f236c7721d9", "f911669e94f1404392398381d2f07db5", "19fba1d70af241fa9fc1cb355d339f2a", "8e2d9f83f5dc4d2dae38d86e95807e8b", "56a0396d8d114b67841ad0c10e900a5f", "6ac0c663b320470ab9dc67cac2b8e1c6", "07ee00c51e724b91af1c18da47409332", "e1fc14350ae64e44894188da7192b40d", "2da7b973917e45e7b45878af1a3b80e3", "fe3d9493940f4bdd97eb4f5dc5099d4b", "ee29598054304e059874f75c287fce6c", "7337134bdf5c43318f479ee7708bc908", "6379a279d39c4cb79dd33be710edcb5f", "6bd0fce3386545ca8c568bacd26d8f38", "109217886a50440394d6f0836ebfc349", "7e9d5ede13cd43d890491e2ee3af62bc", "1d13927210e743ac8b8e5581e83774fb", "d9361e5609e144dfb0a90d802997265a", "a688ed37c4664c1f812e3ce74eb45fde", "dcbbdc61133440d6b60d73c787a421ac", "4a7ea9b40f9646839f3e58e0600c6b69", "1ae7a39645774eb697e0b448aeb0ad39", "ea8d23d087b54f5eae09e82d01047e5f", "7c59854a314745f3b43b5aa1f894029e", "74ee201fb1cd4ede875a57aa44a99e70", "90070b84d1684f57a1b160e3b270c68e", "53cec92fb9914970b41adc1dd2c319d1", "4d37229b71464e0bb581b0f905ea18b5", "569ac775d2cf4c7cbd3c78d99c701905", "bc96064e886a4570894e44ffcaeb361e", "400ec9bd7b714b9199a9442ae92cba90", "7e107914ec6f443a8f54509d2c9754be", "46bd216220284c46867c9b8c7ace244c", "46068e0ab71544b6878e0c1e50b9ab26", "89c8a92622034739a6fcfa0d9cdafdc8", "b00b2e0ba443470a840a4ca0fc75fdee", "60c027e759ff40df8f102b878696073c", "c73abef07da9401e954955ffef45a6d5", "2a263692f01e4b9bbe35e29b9d238c5c", "1852d2bf28484fe9980bb5083055e2a3", "7f8430fe960543fba97003247bdc69ab", "94840ac84b6745379351f156e9c994d0", "f649cf5c87c440379406d67a6bc5d1ed", "489644d3ee5f443c88dcdd8716d41d80", "7c932deb99104001abbe35eedec5a2e3", "dfe41ef1f4d542db8b664e27015423ed", "cb4f7a836f6449f28438c52765d21182", "af8bcc6e67134ec1ba09b4448290992a", "dd0e1d4f1e3c4621a6ee6707a66e890d", "22eeaeec01ab408e91fc8b323a1b2c3d", "944b11957cce4260baded61c5f12e8f3", "0b211f2de43e4b1a805ca3718c960cd5", "bb7f34608f924f698b97228cc2453951", "7c7ba370c1574fcd8c28bcbd77dfab52", "0ba1a91223f7498596b0ec302c92db64", "a32fdf04629a4669a22e9d9acf96d35a", "b3c56fd5fad94b5888d3f8f0ce09785e", "b0ee4628083a47b4b84dbaccba911c72", "ca9c7666c2de407a8ad9be7b2d96f582", "6a21adc363da44c1828c3512d5608f64", "3b300e87bf884fc6b6b0582297c30d28", "fca563ea987a403ba7ba3a8909465034", "5a2c17cc87ee4d7098d065d7e21a4599", "3023d52dd2bb4f028e734f2278e99469", "e873b73446434092a55fa9e0273bcf5b", "af5c567136704fdebf2b73c1c357e838", "097046d132e344bda8de76023cf4bc96", "25df0b64d9d24738b2e113ded264d67a", "e59c3f99899049d9af835d6f110bd697", "36c196555cd9419a944ad7204fe2e79d", "02b29bba36af4311aaed25514b73929f", "2136791418e14c19b77777f81a7b53be", "c9c976d5f35e41398b67df7772fd038e", "5d489597315b4c6a9b21ced44104185c", "7a2bdb7000d34f04bba08c623b191c80"]} outputId="4d35c025-ae10-4062-c90f-637d26775706"
visualize_prompts(pipeline,summerize=False,samples_per_prompt=4,
                  include_desc=True,legible_prompt=False,
                  batch_generate=True,save_to_drive=True,
                  save_dir=save_dir)

# + [markdown] id="oa75Kjt0Qs0N"
# ### Just for fun

# + id="SFhSD7qZQs0O"
#@title Run the Stable Diffusion pipeline
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_allocated()

from torch import autocast
# prompt = "a grafitti in a wall with a <cat-toy> on it" #@param {type:"string"}
prompt="Clear, highly detailed book cover with title: Badger's love story and author: Wenxuan Tan"
# prompt="Clear, highly detailed book cover with description "+book_df.loc[7202]['book_desc']

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}
width=512
height=512
all_images = [] 

for _ in range(num_rows):
    with autocast("cuda"):
        index = 0
        prompt = [prompt]*4
        #batch generation
        while index < 4: 
          images = pipeline(prompt[index:index+args.batch_size],height=height,width=width,num_inference_steps=50, guidance_scale=7.5).images
          all_images.extend(images)
          index += args.batch_size
grid = image_grid(all_images, num_samples, num_rows)
wandb.log({"For_fun":wandb.Image(grid)})

####

#@title Login to the Hugging Face Hub
#@markdown Add a token with the "Write Access" role to be able to add your trained concept to the [Library of Concepts](https://huggingface.co/sd-concepts-library)
from huggingface_hub import login
login("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")
# notebook_login()
# from google.colab import drive
# drive.mount("/content/drive",force_remount=True)

# + id="1_h0kO-VnQog" outputId="530fc822-cdad-4642-9e8b-37f9651dbd9d"
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
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
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
parser.add_argument("--lr",help="learning rate",default=5e-6,type=int)
parser.add_argument("--epochs",default=12,type=int)
parser.add_argument("--train_unet",help="whether to train Unet or not",default=False,type=bool)
parser.add_argument("--decay",help="weight_decay",default=1e-2,type=int)
parser.add_argument("--train_text_encoder",default=True,type=bool)
parser.add_argument("--data_root",default="../book dataset",type=str)
parser.add_argument("--num_examples",default=6000,type=int,help="number of training examples")
parser.add_argument("--num_devices",default=3)
parser.add_argument("--gradient_acc_steps",default=8,type=int)
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


# + id="If5Jswe526QP"
#@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
#pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" #@param {type:"string"}
pretrained_model_name_or_path ="CompVis/stable-diffusion-v1-4"
# data_root="/kaggle/input/goodreads-best-books"
# label_root="/kaggle/input/goodreads-best-book-cleaned-version"
data_root=args.data_root
label_root=args.data_root

book_cover_templates=[#the first entry is for "highly legible text"
    "A {} book cover with author {}, book title {} ",
    #repeat some prompts to give model prior knowledge about book cover styles
    "A {} book cover written by author {} with book title {} ",
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
    "A detailed {} book cover with author {} and book title {} "
    
]
#TODO: add more to match the number of templates
summary_placeholders=[
    ", and summary: {}",
    ', and abstract: {}',
    ",summary: {}",
    ", the book describes that {}",
    ", book discription: {}",
    ", main story: {}",
    ", the book is mainly about {}",
    ", and main story: {}",
    "and book abstract: {}",
    ", and book description: {}"
]
test_templates=[#the first entry is for "highly legible text"
    "A {} book cover with author {}, book title {} ",
    #repeat some prompts to give model prior knowledge about book cover styles
    "A {} book cover written by author {} with book title {} ",
    "A {} simple book cover with author {}, book title {} ",
    "A plain {} book cover with author {}. The book title is{} ",
    "A {} vivid book cover with author {}, book title {} ",
    "A  {} book cover with author name:{}, book title: {}",
#     "We are going to create a clear, {}, highly detailed book cover with author named {}, and book title is '{}'",
    "An intricate {}, book cover including book author:{}, book title: '{}'",
    "A detailed, {}, book cover with {} ,written by author {}",
    "A creative, colorful {}, book cover written by {}. The book title is {}, ",
    "A {} old-fashioned, plain book cover written by {}. The book title is {}",
    "A simple, {}, old-fashioned book cover with author name {}, book title {} ",
    "A simple, {}, plain book cover with author name {}, book title {} ",
    "A detailed {} book cover with author {} and book title {} "
    
]
#pad to the same length 
for i in range(len(summary_placeholders),len(test_templates)):
  summary_placeholders+=[random.choice(summary_placeholders)]
summary_placeholders=summary_placeholders[:len(test_templates)]

# imagenet_templates_small = [
#     "a photo of a {}",
#     "a rendering of a {}",
#     "a cropped photo of the {}",
#     "the photo of a {}",
#     "a photo of a clean {}",
#     "a photo of a dirty {}",
#     "a dark photo of the {}",
#     "a photo of my {}",
#     "a photo of the cool {}",
#     "a close-up photo of a {}",
#     "a bright photo of the {}",
#     "a cropped photo of a {}",
#     "a photo of the {}",
#     "a good photo of the {}",
#     "a photo of one {}",
#     "a close-up photo of the {}",
#     "a rendition of the {}",
#     "a photo of the clean {}",
#     "a rendition of a {}",
#     "a photo of a nice {}",
#     "a good photo of a {}",
#     "a photo of the nice {}",
#     "a photo of the small {}",
#     "a photo of the weird {}",
#     "a photo of the large {}",
#     "a photo of a cool {}",
#     "a photo of a small {}",
# ]

# imagenet_style_templates_small = [
#     "a painting in the style of {}",
#     "a rendering in the style of {}",
#     "a cropped painting in the style of {}",
#     "the painting in the style of {}",
#     "a clean painting in the style of {}",
#     "a dirty painting in the style of {}",
#     "a dark painting in the style of {}",
#     "a picture in the style of {}",
#     "a cool painting in the style of {}",
#     "a close-up painting in the style of {}",
#     "a bright painting in the style of {}",
#     "a cropped painting in the style of {}",
#     "a good painting in the style of {}",
#     "a close-up painting in the style of {}",
#     "a rendition in the style of {}",
#     "a nice painting in the style of {}",
#     "a small painting in the style of {}",
#     "a weird painting in the style of {}",
#     "a large painting in the style of {}",
# ]

# + id="fcA-kMQblqUe"
#@title Training hyperparameters 
hyperparam = {
    "learning_rate": args.lr, #original: 5e-4
    "scale_lr": False,
    "epochs": args.epochs,
    "train_batch_size": 1,
    "gradient_accumulation_steps": args.gradient_acc_steps,
    "seed": global_seed,
    "weight_decay": args.decay,
    # "noise_scheduler": "DDIM",
    "pretrained_model_name_or_path": pretrained_model_name_or_path,
    "output_dir": "./model",
    "training_dataset_size":args.num_examples,
    "train_unet": args.train_unet,
    "train_text_encoder": args.train_text_encoder,
    "num_templates": len(book_cover_templates),
    "include_summary": False,#True to add book summary to prompts
    "templates" : book_cover_templates
}

# + id="xp2InXqXW8aY" outputId="32997303-8897-4243-9a83-e4b75a03272e"
#@title Load the Stable Diffusion model
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_auth_token=True,
)

# Load models and create wrapper for stable diffusion
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
    , use_auth_token=True
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
    , use_auth_token=True
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
    , use_auth_token=True
)
#test model with out of max_length token sequence 

# input=tokenizer(book_df.loc[7202]['book_desc'], return_tensors="pt").input_ids[:,:999]
# print(input.shape)
# print("Test encode above max_length(77) text",text_encoder(input))


tokenizer.tokenize(" . "),tokenizer.decode(tokenizer("谁")['input_ids']),tokenizer.convert_tokens_to_ids(tokenizer.tokenize("谁")),tokenizer.convert_tokens_to_ids('.'),tokenizer.encode(tokenizer.tokenize("."))==tokenizer.encode("."),tokenizer.decode(tokenizer.encode("."))

# + id="2ntpEpVfnd-0" outputId="d7437541-b405-4225-8897-86d383419b62"
#@title Setup the dataset and train loader

used_times=[]
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        # learnable_property="object",  # [object, style]
        size=512,
        training_size=1000,#use a subset of the training set to save time
        interpolation="bicubic",
        flip_prob=0,
        test_speed=False,
        include_desc=False,
        summerize_length:int="max length of summerized book description", #not implemented
        legible_text_prob=0,#add "legible text" to prompt
    ):
        
        self.data_root = data_root
        self.image_path=data_root+"/images/images"
        #changed path for kaggle 
        self.df=pd.read_csv(os.path.join(label_root,"df_train.csv")).iloc[:training_size]
        # self.df.set_index(self.df.columns[0],drop=True,inplace=True)
        self.tokenizer = tokenizer
        # self.learnable_property = learnable_property
        # self.size = size
        self.flip_prob = flip_prob
        self.test_speed=test_speed
        self.include_desc=include_desc
        self.summerize_length=summerize_length
        self.transform = nn.Sequential( 
            K.RandomHorizontalFlip(p=self.flip_prob)
        )
        self.legible_text_prob=legible_text_prob
        # self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.size=size
        self._length = len(self.df)
        
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        if self.include_desc:
          self.templates = [str1+str2 for str1, str2 in zip(book_cover_templates,summary_placeholders) ]
        else:
          self.templates=book_cover_templates
    
        
        print("dataset.self.tokenizer.model_max_length:",self.tokenizer.model_max_length)
        print("dataset.self.tokenizer.truncation_side",self.tokenizer.truncation_side)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if self.test_speed:
            import time
            start_time=time.time()

        example = {}
        # print(self.df.head())
        # print(self.df.iloc[i][self.df.columns[0]])
        # print(self.df.columns[0])
        #first column is the index of the data point
        image = Image.open(os.path.join(self.image_path,str(self.df[self.df.columns[0]].iloc[i])+".jpg"))

        if not image.mode == "RGB":
            image = image.convert("RGB")

        #randomly choose a prompt
        legible_text,author,title,description=("",self.df.iloc[i]['book_authors'], self.df.iloc[i]['book_title'], self.df.iloc[i]['book_desc'])
        if random.random()<=self.legible_text_prob:
            legible_text="legible text"
            
        #debug
        try:
          template=random.choice(self.templates)
          if self.include_desc:
            text = template.format(legible_text,author,title,description)
          else:
            text = template.format(legible_text,author,title)
        except Exception as e:
            print(e)
            print(template)
          
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]



        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        
        image = (image / 127.5 - 1.0).astype(np.float32)
    
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        #Apply data augmentation 
        #Kornia K.RandomHorizontalFlip() unsqueezes the tensor at dim 0, so apply squeeze() to get 3d tensor
        example["pixel_values"] = torch.squeeze(self.transform(example["pixel_values"]))
        # print("Afte transform",example["pixel_values"].shape)

        if self.test_speed:
            print("Used time=",time.time()-start_time)
            global used_times
            used_times.append(time.time()-start_time)
        return example


train_dataset = TextualInversionDataset(
      data_root=data_root,
      tokenizer=tokenizer,
      size=512,
      training_size=hyperparam["training_dataset_size"],
      include_desc=hyperparam["include_summary"]
)
def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,pin_memory=True,num_workers=2)
print("Number of training examples used:", len(train_dataset.df))
print("Number of templates:",len(train_dataset.templates))
print("Training image size:",train_dataset.size)
print("Train loader size:", len(create_dataloader(hyperparam["train_batch_size"])))
print()
print("Templates:\n",train_dataset.templates)

# + [markdown] id="GD5MU6EzFe27"
# We have added the `placeholder_token` in the `tokenizer` so we resize the token embeddings here, this will a new embedding vector in the token embeddings for our `placeholder_token`

# + id="24-9I6mIoORT" outputId="868320de-f46b-42e1-88b9-0a878ac78093"
text_encoder.resize_token_embeddings(len(tokenizer))

# + [markdown] id="IA3Xj7gBFv-6"
#  Initialise the newly added placeholder token with the embeddings of the initializer token

# + id="0mtxiZMNoQvE"
token_embeds = text_encoder.get_input_embeddings().weight.data


noise_scheduler=DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

#@title Visualize training result
#fix random seed by fixing latents
latents=None
def visualize_prompts(
    pipeline: StableDiffusionPipeline,
    summerize=False,
    include_desc=False,#include description
    max_length=15,#only when summerize=True
    legible_prompt=True,
    samples_per_prompt=3,
    img_size=512,
    inference_steps=75,
    save_to_drive=True,
    batch_generate=False
    ):
    if summerize==True:
      assert include_desc==True, "include_desc is False, \
      no summerization can be done without book description!" 
    if include_desc==True and batch_generate==True:
      #TODO: checkout the bug: passing tokenizer with padding=True to from_pretrained() does not solve this.
      print("Setting batch_generate to false since passing stacked descriptions of different length to model will cause error.")
      print("---------------------------------------------")
      batch_generate=False
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
    df=pd.read_csv(label_root+"/df_test.csv")

    #set up figures
    dpi=plt.figure().dpi
    fig,axes=plt.subplots(len(test_templates),
                          samples_per_prompt,
                          figsize=(img_size/dpi*samples_per_prompt,
                                   img_size/dpi*len(test_templates))
                          )
    fig.subplots_adjust(wspace=0, hspace=0)#combind with axes[i][j].set_aspect('auto'); remove spacing
    # plt.suptitle(,y=0.89)

    #fix random seed by fixing latents
    if include_desc:
      from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
      tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
      if summerize:
          model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")


    for i in range(len(test_templates)):
      text=[]
      for j in range(samples_per_prompt):
        row=df.iloc[j]
        legible_text,author,title,description = ("",row['book_authors'], row['book_title'], row['book_desc'])

        if legible_prompt:
            legible_text="legible text"
        if summerize:
            inputs = tokenizer(description, max_length=1024, 
                               return_tensors="pt",truncation=True,padding="max_length")
            summary_ids = model.generate(inputs['input_ids'], num_beams=3,\
                                         min_length=2, max_length=max_length)
            description = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, 
                                          clean_up_tokenization_spaces=False)[0]#batch_decode returns a list of strings; here len(list)=1, only one input string
        #get prompt
        template=test_templates[i]
        if include_desc:
          template+=summary_placeholders[i]#append new prompt to list
          template=template.format(legible_text,author,title,description)
          text += [template]
        else:
          text += [template.format(legible_text,author,title)]


      #inference 
      from torch import autocast
      images=[]
      print(f"Inference iteration {i}")

      with autocast("cuda"):
        if batch_generate:#batch generation
          images=pipeline(text,height=img_size,width=img_size,
                          num_inference_steps=50, guidance_scale=7.5,
                          latents=latents).images
        else:#To avoid out of memory, generate one at a time
          for j in range(samples_per_prompt):
            images+=pipeline(text[j],height=img_size,
                              width=img_size,num_inference_steps=inference_steps, 
                              guidance_scale=7.5,latents=latents[None,j]).images
                              
      try:
        axes[i][0].set_title(f"Prompt {i}, legible={legible_prompt},summerize={summerize},include_desc={include_desc}")
        for j in range(samples_per_prompt):
            axes[i][j].imshow(images[j])
            axes[i][j].set_aspect('auto')#remove spacing
      #single plot case
      except:
        axes[i].set_title(f"Prompt {i}, legible={legible_prompt},summerize={summerize},include_desc={include_desc}")
        #debug
        print(images[0])
        print("images:",images)
        axes[i].imshow(images[0])
        axes[i].set_aspect('auto')

    if save_to_drive:  
      #save fig with paramters
      img_name=f"Generated_covers:legible={legible_prompt},summerize={summerize},\
                include_desc={include_desc},max_length={max_length}.png"
      path="./"+img_name
      plt.savefig(path)
      fig.show()
    else:
      #save checkpoint generation results in wandb
      img_path="checkpoint_image_sample.jpg"
      plt.savefig(img_path)
      from PIL import Image
      image=Image.open(img_path)
      wandb.log({"examples":wandb.Image(image)})



def freeze_params(params):
    for param in params:
        param.requires_grad = False



# Freeze all parameters except for the token embeddings in text encoder
# params_to_freeze = itertools.chain(
#     text_encoder.text_model.encoder.parameters(),
#     text_encoder.text_model.final_layer_norm.parameters(),
#     text_encoder.text_model.embeddings.position_embedding.parameters(),
# )
# freeze_params(params_to_freeze)


def training_function(
                    text_encoder, vae, unet,
                    resume=False,train_unet=False,train_text_encoder=True,
                    gradient_checkpointing=False,use_8bit_adam=True):
    logger = get_logger(__name__)#TODO: switch to wandb
    wandb.login(key='16d21dc747a6f33247f1e9c96895d4ffa5ea0b27',relogin=True)
    wandb.init(
           project="book_cover_generation", 
           config=hyperparam, 
           name="stable_diffusion",
           tags=["reverted to Kaggle version 10","Simplified templates", "text_encoder_only"],
           )
    
    #extract hyperparams
    train_batch_size = hyperparam["train_batch_size"]
    gradient_accumulation_steps = hyperparam["gradient_accumulation_steps"]
    learning_rate = hyperparam["learning_rate"]
    num_train_epochs = hyperparam["epochs"]
    output_dir = hyperparam["output_dir"]
    weight_decay=hyperparam["weight_decay"]
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    train_dataloader = create_dataloader(train_batch_size)
    if hyperparam["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    print("lr after hyperparam[\"scale_lr\"]:",learning_rate)

    

    #prepare models for training
    
    
    if train_text_encoder: 
      text_encoder = accelerator.prepare(text_encoder)
      text_encoder.train()
      if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
    else:
      text_encoder.to(accelerator.device,dtype=torch.float16)
      freeze_params(text_encoder.parameters())
      text_encoder.eval()

    if train_unet:
      unet=accelerator.prepare(unet)
      unet.train()
      if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
      
    else:
      # Move models that don't need to be trained to device with fp16 
      unet.to(accelerator.device,dtype=torch.float16)
      freeze_params(unet.parameters())
      unet.eval()

    freeze_params(vae.parameters())
    vae.to(accelerator.device,dtype=torch.float16)
    vae.eval()

    # Initialize the optimizer
    print(f"Train unet:{unet.training} || Train text_encoder:{text_encoder.training}")
    param_list=[model.parameters() for model in [unet,text_encoder] if model.training]
    params_to_train=itertools.chain(*param_list)
     
    if use_8bit_adam:
      import bitsandbytes as bnb
      optimizer_class = bnb.optim.AdamW8bit
    else:
      optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_train,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-6,verbose=True)
    print("optimizer after wrapping using accelerator:",optimizer)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    max_train_steps = num_update_steps_per_epoch*num_train_epochs

    print("num_update_steps_per_epoch:",num_update_steps_per_epoch)
    print('num_train_epochs',num_train_epochs)
    print("accelerator.num_processes",accelerator.num_processes)


    ###########
    # Train!  #
    ###########
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    print("Train!")
    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Max gradient update steps")
    global_step = 0
    min_loss=1e9
    for epoch in range(num_train_epochs):
        text_encoder.train()
        epoch_loss=None
        for step, batch in enumerate(train_dataloader):
          from torch import autocast
          with autocast('cuda'):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                #aggregate epoch training loss
                if not epoch_loss:
                  epoch_loss=loss.detach().item()
                else:
                  epoch_loss+=loss.detach().item()
                accelerator.backward(loss)


                #save best model every 1/2 epoch
                saves_per_epoch=2
                if (step+1)%int(len(train_dataloader)/saves_per_epoch)==0 or step+1==len(train_dataloader):
                  epoch_loss=epoch_loss/len(train_dataloader)*saves_per_epoch
                  wandb.log({"mean_epoch_loss":epoch_loss,
                            "epoch":int((step+1)/(len(train_dataloader)/saves_per_epoch))
                           })
                  if epoch_loss<min_loss:
                    min_loss=epoch_loss
                    epoch_loss=0
                    
                    print(f"New min epoch loss {min_loss} at training step {step} of epoch {epoch}! Saving model...")
                    if accelerator.is_main_process:
                      #without this, float16 weights will be saved, but unet and vae sub modules in Stablediffusion pipeline don't support that!
                      #you have to save float32 weights and then switch to float16 in from_pretrained()
                      unet.to(accelerator.device,dtype=torch.float32)
                      vae.to(accelerator.device,dtype=torch.float32)
                      pipeline = StableDiffusionPipeline(
                          text_encoder=accelerator.unwrap_model(text_encoder),
                          vae=vae,
                          unet=unet,
                          tokenizer=tokenizer,
                          scheduler=noise_scheduler,
                          safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                          feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                      )


                      #save model
                      # pipeline.save_pretrained(output_dir)
                      #save model to wandb
                      pipeline.save_pretrained(output_dir)
                      del pipeline
                      artifact = wandb.Artifact("stable_diffusion_model", "model")
                      artifact.add_dir(output_dir)
#                       pipeline = StableDiffusionPipeline.from_pretrained(
#                         "./model",
#                         torch_dtype=torch.float16,
#                       ).to('cuda')
                      #save generated image to wandb
                      #causes OOE 
#                       visualize_prompts(
#                           pipeline,
#                           summerize=False,
#                           include_desc=False,
#                           legible_prompt=True,
#                           samples_per_prompt=1,
#                           save_to_drive=False,
#                           batch_generate=False,
#                           )
#                       del pipeline

                      #switch back to float16 for training
                      unet.to(accelerator.device,dtype=torch.float16)
                      vae.to(accelerator.device,dtype=torch.float16)
                      
                      #save optimizer
                      try:
                        torch.save(optimizer.state_dict(),os.path.join(output_dir,"optimizer_state_dict.pt"))
                      except:
                        optimizer.save_state(output_dir)


                optimizer.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            scheduler.step()
            logs = {"loss": loss.detach().item(),"epoch": epoch,"step": f"{step}/{len(train_dataloader)}"}
            wandb.log(logs)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
          #for distributed training 
          accelerator.wait_for_everyone()
    wandb.run.log_artifact(artifact)
    subprocess.run(["rm","-r",output_dir])

      
        
          # Also save the newly trained embeddings
          # learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
          # learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
          # torch.save(learned_embeds_dict, os.path.join(output_dir, "learned_embeds.bin"))

# + [markdown] id="pem_EG-C6OE9"
# # Train model!

# + id="jXi0NdsyBA4S" outputId="ac99079d-e0d2-45ea-ae00-050ac2b1ffb2"

import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_allocated()


import accelerate
from multiprocess import set_start_method
set_start_method("spawn")#avoid CUDA error: RuntimeError: Cannot re-initialize CUDA in forked subprocess
#args in the second line:
#resume,train_unet,train_text_encoder,gradient_checkpointing,use_8bit_adam
accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, 
                            False,hyperparam["train_unet"],hyperparam["train_text_encoder"],False,True),
                            num_processes=args.num_devices)



# ### Load from  checkpoint

# + id="2CMlPbOeEC09"
 #@title Fine tune result evaluation
output_dir=hyperparam["output_dir"]
if os.path.isdir(output_dir):
    #load from local checkpoint
    try:
      pipeline = StableDiffusionPipeline.from_pretrained(
                hyperparam["output_dir"],
                torch_dtype=torch.float16,
                safety_checker = None
              ).to('cuda')
      print(f"Built pipeline from {output_dir}")
    except:
      #manual pipeline
      accelerator = Accelerator()

      if not "text_encoder" in globals():
        text_encoder = CLIPTextModel.from_pretrained(
            hyperparam["output_dir"], subfolder="text_encoder"
            , use_auth_token=True
        )
      if not "vae" in globals():
        vae = AutoencoderKL.from_pretrained(
            hyperparam["output_dir"], subfolder="vae"
            , use_auth_token=True
        )
        vae.to(accelerator.device,dtype=torch.float32)
      if not "unet" in globals():
        unet = UNet2DConditionModel.from_pretrained(
            hyperparam["output_dir"], subfolder="unet"
            , use_auth_token=True
        )
        unet.to(accelerator.device,dtype=torch.float32)
      if not "tokenizer" in globals():
        tokenizer = CLIPTokenizer.from_pretrained(
        hyperparam["output_dir"],
        subfolder="tokenizer",
        use_auth_token=True,
      )

      pipeline = StableDiffusionPipeline(
                      text_encoder=text_encoder,
                      vae=vae,
                      unet=unet,
                      tokenizer=tokenizer,
                      scheduler=PNDMScheduler(
                          beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                      ),
                      safety_checker=None,
                      feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),

      )
      print(f"Built pipeline from components from {output_dir}")
else:
    #load from wandb checkpoint
    os.environ["WANDB_API_KEY"]="16d21dc747a6f33247f1e9c96895d4ffa5ea0b27"
    #can't use artifact in offline mode
#     os.environ['WANDB_MODE'] = 'online'

    with wandb.init(project="book_cover_generation") as run:
      my_model_artifact =run.use_artifact("stable_diffusion_model:v17")
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
          tokenizer=tokenizer#enable padding
          ).to('cuda')
    print('Load model from wandb cloud checkpoint')


# + [markdown] id="yY7vzTj4HIiK"
# ## Visualize different prompt strategies

# + id="NWJLAxXRbvPj" outputId="83fb187d-4316-43c6-f4c2-d93d7d029c75"
visualize_prompts(pipeline,summerize=False,include_desc=False,legible_prompt=False)

# + [markdown] id="nCJdBiUWLD4Q"
# ### Test effectiveness of summerization with other factors controlled for.

# + id="GLfsx-3HDMx2" outputId="ad3d29da-256c-4329-d91b-5572f80f2386"
visualize_prompts(pipeline,summerize=True,include_desc=True,legible_prompt=False)

# + id="QeoyImZMtFKH" outputId="4f63367e-6b0f-4d2b-8134-b1dd0df6a929"
visualize_prompts(pipeline,summerize=False,include_desc=True,legible_prompt=False)



# + [markdown] id="30H2mXBsKy4r"
# ## Model Evaluation

# + id="BXW02rvFKt3e"
# /kaggle/input/goodreads-best-book-cleaned-version/df_test.csv
# /kaggle/input/goodreads-best-books/df_test.csv
# #@title CLIP score
# import gradio as gr
# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# def calculate_score(image, text):
#     labels = text.split(";")
#     labels = [l.strip() for l in labels]
#     labels = list(filter(None, labels))
#     if len(labels) == 0:
#         return dict()
#     inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image.detach().numpy()

#     results_dict = {
#         label: score / 100.0 for label, score in zip(labels, logits_per_image[0])
#     }
#     return results_dict

# + id="vw2LVBFvLXrq"
#@title FID score


# + id="E3UREGd7EkLh" outputId="759772e3-978b-4f23-96e7-93bab5c71526"
#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_allocated()

from torch import autocast
# prompt = "a grafitti in a wall with a <cat-toy> on it" #@param {type:"string"}
prompt="Clear, highly detailed book cover with title とある魔術の禁書目録 2"
# prompt="Clear, highly detailed book cover with description "+book_df.loc[7202]['book_desc']

num_samples = 2 #@param {type:"number"}
num_rows = 2 #@param {type:"number"}
width=512
height=512
all_images = [] 

for _ in range(num_rows):
    with autocast("cuda"):
        images = pipeline([prompt] * num_samples,height=height,width=width,num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid
# -



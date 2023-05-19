from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler, AutoencoderKL
from accelerate import Accelerator
import wandb
import torch, os, subprocess
from torch.utils.data import Dataset
import pandas as pd 
import PIL
from PIL import Image
import random
import numpy as np


book_cover_templates = [  # the first entry is for "highly legible text"
    "A {} book cover with author {}, book title {} ",
    # repeat some prompts to give model prior knowledge about book cover styles
    "A {} book cover written by author {} with book title {} ",
    #     "A {} simple book cover with author {}, book title {} ",
    #     "A plain {} book cover with author {}. The book title is{} ",
    #     "A {} vivid book cover with author {}, book title {} ",
    "A  {} book cover with author name:{}, book title: {}",
    # #     "We are going to create a clear, {}, highly detailed book cover with author named {}, and book title is '{}'",
    "An intricate {} book cover including book author:{}, book title: '{}'",
    #     "A detailed, {} book cover with {} ,written by author {}",
    #     "A creative, colorful {}, book cover written by {}. The book title is {}, ",
    #     "A {} old-fashioned, plain book cover written by {}. The book title is {}",
    #     "A simple, {}, old-fashioned book cover with author name {}, book title {} ",
    #     "A {} old-fashioned, plain book cover written by {}. The book title is {}",
    #     "A simple, {}, old-fashioned book cover with author name {}, book title {} ",
    #     "A simple, {}, plain book cover with author name {}, book title {} ",
    "A detailed {} book cover with author {} and book title {} "

]
# TODO: add more to match the number of templates
summary_placeholders = [
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
test_templates = [  # the first entry is for "highly legible text"
    "A {} book cover with author {}, book title {} ",
    # repeat some prompts to give model prior knowledge about book cover styles
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


class CustomDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        # learnable_property="object",  # [object, style]
        size=512,
        training_size=1000,  # use a subset of the training set to save time
        interpolation="bicubic",
        test_speed=False,
        include_desc=False,
        summerize_length: int = "max length of summerized book description",  # not implemented
        legible_text_prob=0,  # add "legible text" to prompt
        label_root=None,
    ):

        self.data_root = data_root
        self.image_path = os.path.join(data_root, "images", "images")
        # changed path for kaggle
        label_root = data_root if label_root is None else label_root
        self.df = pd.read_csv(os.path.join(
            label_root, "df_train.csv")).iloc[:training_size]
        
        # self.df.set_index(self.df.columns[0],drop=True,inplace=True)
        self.tokenizer = tokenizer
        # self.learnable_property = learnable_property
        # self.size = size
        self.test_speed = test_speed
        self.include_desc = include_desc
        self.summerize_length = summerize_length
        self.legible_text_prob = legible_text_prob
        # self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.size = size
        self._length = len(self.df)

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        if self.include_desc:
            self.templates = [
                str1+str2 for str1, str2 in zip(book_cover_templates, summary_placeholders)]
        else:
            self.templates = book_cover_templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if self.test_speed:
            import time
            start_time = time.time()

        example = {}
        image = Image.open(os.path.join(self.image_path, str(
            self.df[self.df.columns[0]].iloc[i])+".jpg"))

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # randomly choose a prompt
        legible_text, author, title, description = (
            "", self.df.iloc[i]['book_authors'], self.df.iloc[i]['book_title'], self.df.iloc[i]['book_desc'])
        if random.random() <= self.legible_text_prob:
            legible_text = "legible text"
        # debug
        try:
            template = random.choice(self.templates)
            if self.include_desc:
                text = template.format(
                    legible_text, author, title, description)
            else:
                text = template.format(legible_text, author, title)
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
        image = image.resize((self.size, self.size),
                             resample=self.interpolation)
        image = np.array(image).astype(np.uint8)

        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        if self.test_speed:
            print("Used time=", time.time()-start_time)
            global used_times
            used_times.append(time.time()-start_time)
        return example


def create_dataloader(dataset, train_batch_size=1):
    return torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4)


def freeze_params(params):
    for param in params:
        param.requires_grad = False

def load_model(output_dir=None, wandb_run_id=None):
    assert output_dir or wandb_run_id, "Please provide either output_dir or wandb_run_id to load checkpoint from"
    output_dir = output_dir
    if os.path.isdir(output_dir):
        # load from local checkpoint
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                output_dir,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to('cuda')
            print(f"Built pipeline from {output_dir}")
        except:
            # manual pipeline
            accelerator = Accelerator()

            if not "text_encoder" in globals():
                text_encoder = CLIPTextModel.from_pretrained(
                    output_dir, subfolder="text_encoder"
                )
            if not "vae" in globals():
                vae = AutoencoderKL.from_pretrained(
                    output_dir, subfolder="vae"
                )
                vae.to(accelerator.device, dtype=torch.float32)
            if not "unet" in globals():
                unet = UNet2DConditionModel.from_pretrained(
                    output_dir, subfolder="unet"
                )
                unet.to(accelerator.device, dtype=torch.float32)
            if not "tokenizer" in globals():
                tokenizer = CLIPTokenizer.from_pretrained(
                    output_dir,
                    subfolder="tokenizer",
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
                feature_extractor=CLIPFeatureExtractor.from_pretrained(
                    "openai/clip-vit-base-patch32"),

            )
            print(f"Built pipeline from components from {output_dir}")
    else:
        # load from wandb checkpoint
        if wandb_run_id:
            run = wandb.init(project="book_cover_generation", id=wandb_run_id)
        else:
            run = wandb.init(project="book_cover_generation")            
        my_model_artifact = run.use_artifact("stable_diffusion_model:latest")
        # Download model weights to a folder and return the path
        model_dir = my_model_artifact.download()

        tokenizer = CLIPTokenizer.from_pretrained(
            model_dir,
            subfolder="tokenizer",
            Padding="max_length",
            Truncation=True,
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            safety_checker=None,
            tokenizer=tokenizer  # enable padding
        ).to('cuda')
        subprocess.Popen(["rm", "-r", "-f",  model_dir])
        print('Load model from wandb cloud checkpoint')
        
    return pipeline
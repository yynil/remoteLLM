import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKL,AltDiffusionPipeline,StableDiffusionPipeline,StableDiffusionImg2ImgPipeline
from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.optim import AdamW
import datetime
import os

class MLP(torch.nn.Module):

    def __init__(self,in_feats,hidden_feats,out_feats,layers) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if layers == 1:
            self.layers.append(torch.nn.Linear(in_feats,out_feats))
            return
        self.layers.append(torch.nn.Linear(in_feats,hidden_feats))
        for i in range(layers-1):
            self.layers.append(torch.nn.Linear(hidden_feats,hidden_feats))
        self.layers.append(torch.nn.Linear(hidden_feats,out_feats))
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def make_pil_images_grid(pil_images,col_num=2):
    width = pil_images[0].width
    height = pil_images[0].height
    length = len(pil_images)
    grid_width = width * col_num + 10 * (col_num - 1)
    grid_height = height * (length // col_num + (1 if length % col_num != 0 else 0)) + 10 * (len(pil_images) // col_num)
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    for i in range(len(pil_images)):
        grid.paste(pil_images[i], (i % col_num * (width + 10), i // col_num * (height + 10)))
    return grid

if __name__ == '__main__':
    output_dir = input("Please input the output dir: ")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_layers = 1
    device = "cuda"
    pipe = AltDiffusionPipeline.from_pretrained('F:/models/huggingface/AltDiffusion-m9/')
    print(pipe)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    noise_scheduler = pipe.scheduler
    unet = pipe.unet
    unet.requires_grad_(False)
    text_encoder = pipe.text_encoder
    text_encoder.requires_grad_(False)
    wte = text_encoder.get_input_embeddings()
    vae = pipe.vae
    vae.requires_grad_(False)
    
    if 'project_dim' in text_encoder.config.__dict__:
        num_emb_dim = text_encoder.config.project_dim
    elif 'hidden_size' in text_encoder.config.__dict__:
        num_emb_dim = text_encoder.config.hidden_size
    else:
        print("No projection dim found")
        exit()
    embedding_ckpt = "F:\\data\\Arknights-Diffusion-Dataset\\last-checkpoint-mlp-altm9-single-layer-1-18301.pt"
    mlp = MLP(num_emb_dim,num_emb_dim,num_emb_dim,num_layers).to(device)
    mlp.load_state_dict(torch.load(embedding_ckpt))
    mlp.eval()
    # additional_prompts_emb = torch.load(embedding_ckpt)
    # additional_prompts_emb.load_state_dict(torch.load(embedding_ckpt))
    print(mlp)
    center_crop = False
    random_flip = True
    tokenizer = pipe.tokenizer
    
    index = 0
    steps = 200
    while True:
        text_prompt = input("Please input the text prompt: ")
        steps = int(input("Please input the number of steps: "))
        sample_size = int(input("Please input the sample size: "))
        texts = [text_prompt]*sample_size
        text_prompt_save_path = os.path.join(output_dir,f"text_prompt_{index}.txt")
        with open(text_prompt_save_path,"w") as f:
            f.write(text_prompt)
        inputs = tokenizer(
            texts, max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors="pt"
        )
        attention_mask = inputs.attention_mask.to(device)
        prompt_embs = text_encoder(inputs.input_ids.to(device),attention_mask=attention_mask)
        encoder_hidden_states = prompt_embs[0]
        encoder_hidden_states = mlp(encoder_hidden_states)
        print(encoder_hidden_states.shape)
        generated_images = pipe(None,prompt_embeds=encoder_hidden_states,num_inference_steps=steps,output_type='pil', guidance_scale=7.5).images
        grid = make_pil_images_grid(generated_images)
        grid_file_path = os.path.join(output_dir,f"grid_{index}_with_samples_{sample_size}_with_steps_{steps}.png")
        grid.save(grid_file_path)
        
        
        original_images = pipe(texts, num_inference_steps=steps,output_type='pil', guidance_scale=7.5).images
        grid = make_pil_images_grid(original_images)
        grid_file_path = os.path.join(output_dir,f"grid_{index}_with_samples_{sample_size}_with_steps_{steps}_original.png")
        grid.save(grid_file_path)
        index += 1

    
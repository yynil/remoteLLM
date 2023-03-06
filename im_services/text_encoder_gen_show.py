import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKL,AltDiffusionPipeline,StableDiffusionPipeline,StableDiffusionImg2ImgPipeline

from PIL import Image
import os

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
    device = "cuda"
    pipe = AltDiffusionPipeline.from_pretrained('F:\\models\\huggingface\\AltDiffusion-m9')
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
    print(vae)
    num_emb_dim = text_encoder.config.hidden_size
    num_prompts = 100
    print(num_emb_dim)
    class EmbModule(torch.nn.Module) :
        def __init__(self, nUser, edim_u):
            super(EmbModule, self).__init__()
            self.U = torch.nn.Embedding(nUser, edim_u)
        def forward(self, x):
            return self.U(x)

    embedding_ckpt = "F:\\data\\Arknights-Diffusion-Dataset\\last-checkpoint-altm9-prompt-embedding-100-24001.pt"
    additional_prompts_emb = EmbModule(num_prompts, num_emb_dim).to(device)
    additional_prompts_emb.load_state_dict(torch.load(embedding_ckpt))
    # additional_prompts_emb = torch.load(embedding_ckpt)
    # additional_prompts_emb.load_state_dict(torch.load(embedding_ckpt))
    print(additional_prompts_emb)

    tokenizer = pipe.tokenizer
    
    
    index = 0
    steps = 100
    while True:
        text_prompt = input("Please input the text prompt: ")
        steps = int(input("Please input the number of steps: "))
        sample_size_iter = int(input("Please input the sample size: "))
        
        text_prompt_save_path = os.path.join(output_dir,f"text_prompt_{index}.txt")
        with open(text_prompt_save_path,"w") as f:
            f.write(text_prompt)

        sample_size = 1
        generated_images = []
        texts = [text_prompt]*sample_size
        inputs = tokenizer(
                texts, max_length=tokenizer.model_max_length-num_prompts, padding="longest", truncation=True, return_tensors="pt"
        )
        additional_prompts = torch.arange(num_prompts).repeat(sample_size).reshape((sample_size,num_prompts)).to('cuda')
        my_prompts_embs = additional_prompts_emb(additional_prompts)
        inputs_ids_embs = wte(inputs.input_ids.to(device))
        inputs_ids_embs = torch.cat([my_prompts_embs,inputs_ids_embs], dim=1).to(device)
        attention_mask = inputs.attention_mask.to(device)
        attention_mask = torch.cat([torch.ones((sample_size,num_prompts)).to(device),attention_mask], dim=1).to(device)
        prompt_embs = text_encoder(None,inputs_embeds=inputs_ids_embs,attention_mask=attention_mask)
        encoder_hidden_states = prompt_embs[0]
        for i in range(sample_size_iter):
            generated_images.append(pipe(None,prompt_embeds=encoder_hidden_states,num_inference_steps=steps,output_type='pil', guidance_scale=7.5).images[0])
        grid = make_pil_images_grid(generated_images)
        grid_file_path = os.path.join(output_dir,f"grid_{index}_with_samples_{sample_size}_with_steps_{steps}.png")
        grid.save(grid_file_path)
        original_images = []
        for i in range(sample_size_iter):
            original_images.append(pipe(texts, num_inference_steps=steps,output_type='pil', guidance_scale=7.5).images[0])
        grid = make_pil_images_grid(original_images)
        grid_file_path = os.path.join(output_dir,f"grid_{index}_with_samples_{sample_size}_with_steps_{steps}_original.png")
        grid.save(grid_file_path)
        index += 1

    
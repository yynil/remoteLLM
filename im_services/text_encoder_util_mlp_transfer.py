import argparse
import logging
import math
import os
import datasets
import torch
from tqdm import tqdm
import transformers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.optim import AdamW
import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import numpy as np
from accelerate.logging import get_logger
logger = get_logger(__name__)


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



def main(args):
    model_path = args.model_path
    name = args.name
    num_layers = args.num_layers
    epoch_ckpt = args.epoch_ckpt
    ckpt_dir = args.ckpt_dir
    resolution = args.resolution
    center_crop = args.center_crop
    random_flip = args.random_flip
    caption_column = args.caption_column
    data_dir = args.data_dir
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    mixed_precision = args.mixed_precision
    logging_dir = args.logging_dir
    report_to = args.report_to
    
    
     # Load scheduler, tokenizer and models.
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    noise_scheduler = pipe.scheduler
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    print(text_encoder)
    print(text_encoder.config)
    if 'project_dim' in text_encoder.config.__dict__:
        num_emb_dim = text_encoder.config.project_dim
    elif 'hidden_size' in text_encoder.config.__dict__:
        num_emb_dim = text_encoder.config.hidden_size
    else:
        print("No projection dim found")
        exit()
    mlp = MLP(num_emb_dim,num_emb_dim,num_emb_dim,num_layers)
    print(mlp)
        
    mlp.requires_grad_(True)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids, inputs.attention_mask

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
        return result

    def preprocess_train(examples):
        images = [expand2square(image.convert("RGB"), Image.LANCZOS) for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"],examples["attention_mask"] = tokenize_captions(examples)
        
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}
    

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        logging_dir=logging_dir,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    mlp.to(accelerator.device, dtype=weight_dtype)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        mlp.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    from datasets import load_dataset
    train_dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
    train_dataset = train_dataset.with_transform(preprocess_train)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    mlp, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        mlp, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # Potentially load in the weights and states from a previous save
    path = None
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.ckpt_dir)
            dirs = [d for d in dirs if d.startswith(f"checkpoint-mlp-{name}")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.ckpt_dir, path))
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    # Only show the progress bar once on each machine.
    print("Starting training from epoch", first_epoch, "step", global_step)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    checkpointing_steps = 1000
    max_grad_norm=1.0
    for epoch in range(first_epoch, args.num_train_epochs):
        mlp.train()
        train_loss = 0.0
        count = 0
        for step, batch in enumerate(train_dataloader):
            count += 1
            with accelerator.accumulate(mlp):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                #add prefix to the prompt
                # additional_prompts = torch.arange(num_prompts).repeat(bsz).reshape((bsz,num_prompts)).to('cuda')
                # my_prompts_embs = additional_prompts_emb(additional_prompts)
                # inputs_ids_embs = wte(batch["input_ids"])
                # inputs_ids_embs = torch.cat([my_prompts_embs,inputs_ids_embs], dim=1).to('cuda', dtype=weight_dtype)
                # attention_mask = torch.cat([torch.ones((bsz,num_prompts)).to('cuda'),batch["attention_mask"]], dim=1).to('cuda')
                input_ids = batch["input_ids"].to('cuda')
                attention_mask = batch["attention_mask"].to('cuda')
                prompt_embs = text_encoder(input_ids,attention_mask=attention_mask)
                encoder_hidden_states = prompt_embs[0]
                encoder_hidden_states = mlp(encoder_hidden_states).to('cuda', dtype=weight_dtype)
                #Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item() /gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(mlp.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.ckpt_dir, f"checkpoint-mlp-{name}-{num_layers}-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        print(mlp.state_dict())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        print("Epoch", epoch, "loss", train_loss/count)
        

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        mlp = accelerator.unwrap_model(mlp)
        save_path = os.path.join(args.ckpt_dir, f"last-checkpoint-mlp-{name}-{num_layers}-{global_step}.pt")
        #save the additional prompts embeddings
        torch.save(mlp.state_dict(), save_path)
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--epoch_ckpt", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=12000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    args = parser.parse_args()
    main(args)


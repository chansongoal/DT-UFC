import os
import re
import time 
import math
import random
import numpy as np
from PIL import Image
from functools import partial
import torch
import argparse
from diffusers import StableDiffusion3Pipeline
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance


random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
np.random.seed(3407)


def get_captions(source_captions_name):
    image_names = []
    captions = []

    with open(source_captions_name, 'r') as f:
        for line in f:
            image_name, caption = line.split(" ", 1)
            image_names.append(image_name)
            captions.append(caption.strip())

    return captions, image_names

def extract_features_parallel(sd3_pipeline, source_captions, image_names, org_feature_path, org_image_path):
    # source_captions = source_captions[:16]
    batch_size = 16; num_batch = math.ceil(len(source_captions)/batch_size) # please keep the batch size as 16 to extract identical test features. different batch size or total number source_captions results in different features
    for bs_idx in range(num_batch):
        start = bs_idx * batch_size
        end = (bs_idx+1) * batch_size if (bs_idx+1) * batch_size < len(source_captions) else len(source_captions)
        batch_capations = source_captions[start:end]
        batch_images = image_names[start:end]

        # Generate original features when output_type="latent" 
        feats = sd3_pipeline(
                            prompt=batch_capations,
                            negative_prompt="",
                            num_inference_steps=28,
                            height=1024,
                            width=1024,
                            guidance_scale=7.0,
                            output_type="latent"    # output features rather than images
        )
        feats = feats[0].cpu().numpy(); print(feats.shape)
        for idx, caption in enumerate(batch_capations):
            feat_name = os.path.join(org_feature_path, batch_images[idx][:-4] + '.npy')
            feat = feats[idx,:,:,:]; feat = np.expand_dims(feat, axis=0)
            np.save(feat_name, feat)


        # # Generate images when output_type="pil" 
        # imgs = sd3_pipeline(
        #                     prompt=batch_capations,
        #                     negative_prompt="",
        #                     num_inference_steps=28,
        #                     height=1024,
        #                     width=1024,
        #                     guidance_scale=7.0,
        #                     output_type="pil"    # output features rather than images
        # )
        # imgs = imgs[0]
        # for idx, caption in enumerate(batch_capations):
        #     img_name = os.path.join(org_image_path, batch_images[idx][:-4] + '.png')
        #     img = imgs[idx]
        #     img.save(img_name)

def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().detach().permute(0, 2, 3, 1).float().numpy()
    return images

def numpy_to_pil(images: np.ndarray) -> list[Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def feat_to_image_parallel(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path):
    image_names = image_names[:10]
    batch_size = 4; num_batch = math.ceil(len(source_captions)/batch_size)
    num_batch = 1
    for bs_idx in range(num_batch):
        start = bs_idx * batch_size
        end = (bs_idx+1) * batch_size if (bs_idx+1) * batch_size < len(source_captions) else len(source_captions)
        batch_capations = source_captions[start:end]
        batch_images = image_names[start:end]
        
        batch_rec_feat = []
        for idx, image_name in enumerate(batch_images):
            rec_feat_name = os.path.join(rec_feature_path, image_name[:-4]+'.npy')
            rec_feat = np.load(rec_feat_name)
            batch_rec_feat.append(rec_feat)
        batch_rec_feat = np.asarray(batch_rec_feat)[:,0,:,:,:]    
        batch_rec_feat = torch.from_numpy(batch_rec_feat).to('cuda')

        batch_pt_image = sd3_pipeline.generate_image_from_latents(latents=batch_rec_feat, output_type="pt")

        for idx, image_name in enumerate(batch_images):
            rec_img_name = os.path.join(rec_image_path, image_name[:-4]+'.png')
            pt_image = batch_pt_image[idx]
            numpy_image = pt_to_numpy(pt_image)

            pil_image = numpy_to_pil(numpy_image)[0]
            pil_image.save(rec_img_name)

def feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path):
    start_time = time.time()
    # image_names = image_names[:16]
    for idx, image_name in enumerate(image_names): 
        rec_feat_name = os.path.join(rec_feature_path, image_name[:-4]+'.npy')
        rec_img_name = os.path.join(rec_image_path, image_name[:-4]+'.png')

        rec_feat = np.load(rec_feat_name)
        # print(idx, image_name, rec_feat.shape)
        rec_feat = torch.from_numpy(rec_feat).to('cuda')
        
        caption = source_captions[idx]

        # Generate the image
        pt_image = sd3_pipeline.generate_image_from_latents(latents=rec_feat, output_type="pil")
        # numpy_image = pt_to_numpy(pt_image)
        # pil_image = numpy_to_pil(numpy_image)[0]
        pil_image = pt_image[0]
        pil_image.save(rec_img_name)
    print(f"Feat to image time: {(time.time()-start_time):.2f}")

def tti_evaluate_clip_score(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path):
    start_time = time.time()
    # image_names = image_names[:10]
    rec_image_all = []
    for idx, caption in enumerate(source_captions): 
        rec_img_name = os.path.join(rec_image_path, image_names[idx][:-4]+'.png')
        rec_image = np.asarray(Image.open(rec_img_name))
        # rec_image = (rec_image*255).astype('uint8')   # DO NOT PERFORM THIS
        rec_image_all.append(rec_image)
        
    # Compute CLIP Score
    rec_image_all = np.asarray(rec_image_all)   # already in [0, 255], no need further conversion
    clip_score = clip_score_fn(torch.from_numpy(rec_image_all).permute(0, 3, 1, 2), source_captions[:rec_image_all.shape[0]]).detach()
    # print(clip_score)
    clip_score = round(float(clip_score), 4)
    print(f"CLIP Score: {clip_score:.4f}")
    # print(f"Feature MSE: {np.mean(mse_list):.8f}")
    print(f"CLIP Score evaluation time: {(time.time()-start_time):.2f}")

# please refer to https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html for more details
def tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path):
    start_time = time.time()
    # image_names = image_names[:10]
    rec_image_all = []
    for idx, caption in enumerate(source_captions): 
        rec_img_name = os.path.join(rec_image_path, image_names[idx][:-4]+'.png')
        rec_image = np.asarray(Image.open(rec_img_name).resize((299,299)))  # resize to 299x299
        rec_image_all.append(rec_image)
    
    org_image_all = []
    for idx, caption in enumerate(source_captions): 
        org_img_name = os.path.join(org_image_path, image_names[idx][:-4]+'.png')
        org_image = np.asarray(Image.open(org_img_name).resize((299,299)))  # resize to 299x299
        org_image_all.append(org_image)
        
    rec_image_all = np.asarray(rec_image_all)   # already in [0, 255], no need further conversion
    org_image_all = np.asarray(org_image_all)   # already in [0, 255], no need further conversion
    rec_image_tensor = torch.tensor(rec_image_all).permute(0,3,1,2)
    org_image_tensor = torch.tensor(org_image_all).permute(0,3,1,2)

    # Compute FID
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=False, input_img_size=(3, 299, 299))
    fid.update(org_image_tensor, real=True)
    fid.update(rec_image_tensor, real=False)

    print(f"FID: {float(fid.compute()):.4f}")
    print(f"FID evaluation time: {(time.time()-start_time):.2f}")

def tti_pipeline(source_captions_name, org_feature_path, org_image_path, rec_feature_path, rec_image_path, vae_checkpoint_path, sd3_checkpoint_path):
    # Obtain source captions
    source_captions, image_names = get_captions(source_captions_name)

    # Setup models
    sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(sd3_checkpoint_path, torch_dtype=torch.float16)
    sd3_pipeline.enable_model_cpu_offload()

    # Extract features
    os.makedirs(org_feature_path, exist_ok=True)
    os.makedirs(org_image_path, exist_ok=True)
    extract_features_parallel(sd3_pipeline, source_captions, image_names, org_feature_path, org_image_path)

    # # Generate images and evaluate 
    # os.makedirs(rec_image_path, exist_ok=True)
    # feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
    
    # tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path)

    # clip_score_fn = partial(clip_score, model_name_or_path=vae_checkpoint_path)
    # tti_evaluate_clip_score(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
    
def quantization_evaluation(trun_low, trun_high, quant_type, samples, source_file):
    # Setup related path
    dataset_root = '/gdata1/gaocs/FCM_LM_Test_Dataset/sd3/tti'
    source_captions_name = f"{dataset_root}/source/{source_file}"
    sd3_checkpoint_path = "/gdata/liuzj/Data/sd3/tti/pretrained_head/stable-diffusion-3-medium-diffusers"

    org_feature_path = f'{dataset_root}/feature'
    org_image_path = f'{dataset_root}/image'  # use the images generated from original features as the anchor

    root_path = f'/gdata1/gaocs/Data_FCM_NQ/sd3/tti'; print('root_path: ', root_path)

    # Obtain source captions
    source_captions, image_names = get_captions(source_captions_name)

    # Setup models
    sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(sd3_checkpoint_path, torch_dtype=torch.float16)
    sd3_pipeline.enable_model_cpu_offload()

    # Evaluate and print results
    bit_depth_all = [10, 8]

    for bit_depth in bit_depth_all:
        print(source_file, trun_low, trun_high, quant_type, samples, bit_depth)
        rec_feature_path = f"{root_path}/quantization/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}/bitdepth{bit_depth}"
        rec_image_path = f"{rec_feature_path}_image"

        # Generate images 
        os.makedirs(rec_image_path, exist_ok=True)
        feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
        
        # Evaluation
        tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path)

def compressai_evaluation(arch, trun_low, trun_high, quant_type, samples, bit_depth, train_task, lambda_value_all, epochs, learning_rate, batch_size, patch_size):
    # Setup related path
    dataset_root = '/gdata1/gaocs/FCM_LM_Test_Dataset/sd3/tti'
    source_captions_name = f"{dataset_root}/source/captions_val2017_select500.txt"
    sd3_checkpoint_path = "/gdata/liuzj/Data/sd3/tti/pretrained_head/stable-diffusion-3-medium-diffusers"

    org_feature_path = f'{dataset_root}/feature'
    org_image_path = f'{dataset_root}/image'  # use the images generated from original features as the anchor

    root_path = f'/gdata1/gaocs/Data_FCM_NQ/sd3/tti/{arch}'; print('root_path: ', root_path)

    # Obtain source captions
    source_captions, image_names = get_captions(source_captions_name)

    # Setup models
    sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(sd3_checkpoint_path, torch_dtype=torch.float16)
    sd3_pipeline.enable_model_cpu_offload()

    for lambda_value in lambda_value_all:
        print(arch, trun_low, trun_high, quant_type, samples, bit_depth, lambda_value, epochs, learning_rate, batch_size, patch_size)
        if train_task == 'tti':
            rec_feature_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/" \
                           f"lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}"
        else:
            rec_feature_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/" \
                            f"trained_{train_task}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}"
        rec_image_path = f"{rec_feature_path}_image"

        # Generate images 
        os.makedirs(rec_image_path, exist_ok=True)
        feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
        
        # Evaluation
        tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path)

def argument_parsing():
    parser = argparse.ArgumentParser(description="Train Evaluation Pipeline")
    parser.add_argument('--arch', type=str, default='bmshj2018-hyperprior', help='arch')
    parser.add_argument('--trun_low', type=float, default=-506.97, help='trun_low')
    parser.add_argument('--trun_high', type=float, default=105.95, help='trun_high')
    parser.add_argument('--quant_type', type=str, default='kmeans', help='quant_type')
    parser.add_argument('--samples', type=int, default=10, help='samples')
    parser.add_argument('--bit_depth', type=int, default=8, help='bit_depth')
    parser.add_argument('--train_task', type=str, default='seg', help='train_task')
    parser.add_argument('--lambda_value_all', nargs='+', type=float, help='lambda_value_all')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--patch_size', type=str, default='256-256', help='patch_size')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = argument_parsing()
    arch = args.arch
    trun_low = args.trun_low
    trun_high = args.trun_high
    quant_type = args.quant_type
    samples = args.samples
    bit_depth = args.bit_depth
    train_task = args.train_task
    lambda_value_all = args.lambda_value_all
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patch_size = args.patch_size

    compressai_evaluation(arch, trun_low, trun_high, quant_type, samples, bit_depth, train_task, lambda_value_all, epochs, learning_rate, batch_size, patch_size)


    # # for quantization evaluation
    # trun_high = 4.46; trun_low = -5.79 
    # source_file = 'captions_val2017_select100.txt'
    # quant_type = 'kmeans'; samples = 10
    # quantization_evaluation(trun_low, trun_high, quant_type, samples, source_file)
    # # quant_type = 'uniform'; samples = 0
    # # quantization_evaluation(trun_low, trun_high, quant_type, samples, source_file)
    # source_file = 'captions_val2017_select500.txt'
    # quant_type = 'kmeans'; samples = 10
    # quantization_evaluation(trun_low, trun_high, quant_type, samples, source_file)
    # # quant_type = 'uniform'; samples = 0
    # # quantization_evaluation(trun_low, trun_high, quant_type, samples, source_file)
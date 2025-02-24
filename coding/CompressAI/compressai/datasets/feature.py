# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset

import numpy as np 
import json



@register_dataset("FeatureFolder")
class FeatureFolder(Dataset):
    """Load an feature folder database. Training and testing feature samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", model_type="sd3", task="tti", trun_flag=False, trun_low=-20, trun_high=20, quant_type="uniform", qsamples=0, bit_depth=8, quant_points_name='quant_points.json', patch_size=(256, 256)):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        #gcs
        self.samples = self.samples[:5000]

        self.transform = transform

        #gcs
        self.model_type = model_type
        self.task = task
        self.trun_flag = trun_flag
        self.trun_low = trun_low
        self.trun_high = trun_high
        self.quant_type = quant_type
        self.qsamples = qsamples
        self.bit_depth = bit_depth
        self.quant_points_name = quant_points_name
        self.patch_size = patch_size    #(height, width), must be the multiple of 64

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # Load feature, use float32 for training
        feat = np.load(self.samples[index]).astype(np.float32)
        # #gcs, preprocessing
        # if self.trun_flag == True: feat = FeatureFolder.truncation(feat, self.trun_low, self.trun_high)
        # # feat = FeatureFolder.uniform_quantization(feat, self.trun_low, self.trun_high, self.bit_depth)
        # quantization_points = FeatureFolder.load_quantization_points(self.quant_points_name)
        # feat = FeatureFolder.nonlinear_quantization(feat, quantization_points, self.bit_depth)
        # feat = FeatureFolder.packing(feat, self.model_type)
        # feat = FeatureFolder.random_crop(feat, self.patch_size)   # (height, width), must be the multiple of 64
        feat = np.expand_dims(feat, axis=0) # (C,H,W)
        # print(feat.shape)
        return feat

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def truncation(feat, trun_low, trun_high):
        trun_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(trun_low, list):
            for idx in range(len(trun_low)):
                trun_feat[:,idx,:,:] = np.clip(feat[:,idx,:,:], trun_low[idx], trun_high[idx])
        else:
            trun_feat = np.clip(feat, trun_low, trun_high)
        
        return trun_feat

    @staticmethod
    def uniform_quantization(feat, min_v, max_v, bit_depth):
        quant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
                quant_feat[:,idx,:,:] = ((feat[:,idx,:,:]-min_v[idx]) * scale)
        else:
            scale = ((2**bit_depth) -1) / (max_v - min_v)
            quant_feat = ((feat-min_v) * scale)

        return quant_feat

    @staticmethod
    def uniform_dequantization(feat, min_v, max_v, bit_depth):
        feat = feat.astype(np.float32)
        dequant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
                dequant_feat[:,idx,:,:] = feat[:,idx,:,:] / scale + min_v[idx]
        else:
            scale = ((2**bit_depth) -1) / (max_v - min_v)
            dequant_feat = feat / scale + min_v
        return dequant_feat

    @staticmethod
    def load_quantization_points(file_path: str or list[str]):
        """
        Load quantization points from a file or a list of files.
        
        Parameters:
            file_path (Union[str, List[str]]): Path to load the quantization points from.
                Can be a single file path (str) or a list of file paths (List[str]).
        
        Returns:
            Union[numpy.ndarray, List[numpy.ndarray]]: Loaded quantization points. If `file_path`
                is a single path, returns a single numpy.ndarray. If `file_path` is a list of paths,
                returns a list of numpy.ndarray.
        """
        def load_file(path):
            with open(path, 'r') as f:
                quantization_points = np.array(json.load(f))
            # print(f"Quantization points loaded from {path}")
            return quantization_points

        if isinstance(file_path, list):
            # Load quantization points from each file in the list
            return [load_file(path) for path in file_path]
        elif isinstance(file_path, str):
            # Load quantization points from a single file
            return load_file(file_path)
        else:
            raise ValueError("file_path must be a string or a list of strings.")
    
    @staticmethod
    def nonlinear_quantization(data, quantization_points, bit_depth):
        """
        Apply quantization to data using a single or multiple sets of quantization points.
        
        Parameters:
            data (numpy.ndarray): Original floating-point array with shape (N, C, H, W).
            quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]): 
                A single numpy array of quantization points or a list of numpy arrays,
                one for each channel (C).
        
        Returns:
            numpy.ndarray: Quantized integer array with the same shape as the input data.
        """
        if isinstance(quantization_points, np.ndarray):
            # If quantization_points is a single array, apply it to all channels
            num_levels = len(quantization_points)
            data_flat = data.flatten()
            quantized_data_flat = np.digitize(data_flat, quantization_points) - 1
            quantized_data_flat = np.clip(quantized_data_flat, 0, num_levels - 1)
            quantized_data = quantized_data_flat.reshape(data.shape)
        elif isinstance(quantization_points, list):
            if len(quantization_points) != data.shape[1]:
                raise ValueError("Length of quantization_points list must match the number of channels (C) in data.")
            
            quantized_data = np.zeros_like(data, dtype=int)
            # Apply different quantization points to each channel
            for i, qp in enumerate(quantization_points):
                num_levels = len(qp)
                channel_data = data[:, i, :, :]
                channel_data_flat = channel_data.flatten()
                quantized_channel_flat = np.digitize(channel_data_flat, qp) - 1
                quantized_channel_flat = np.clip(quantized_channel_flat, 0, num_levels - 1)
                quantized_data[:, i, :, :] = quantized_channel_flat.reshape(channel_data.shape)
        else:
            raise ValueError("quantization_points must be a numpy array or a list of numpy arrays.")
        
        # quantized_data = quantized_data.astype(np.uint16) if bit_depth>8 else quantized_data.astype(np.uint8)
        quantized_data = quantized_data.astype(np.float32) / (2**bit_depth) # normalize to [0,1)
        return quantized_data

    @staticmethod
    def nonlinear_dequantization(quantized_data, quantization_points, bit_depth):
        """
        Dequantize quantized data back to its approximate original floating-point values.
        
        Parameters:
            quantized_data (numpy.ndarray): Quantized integer array with shape (N, C, H, W).
            quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]): 
                A single numpy array of quantization points or a list of numpy arrays,
                one for each channel (C).
        
        Returns:
            numpy.ndarray: Dequantized floating-point array with the same shape as the input data.
        """
        # scale quantized_data to [0,2**bit_depth-1]
        quantized_data = np.clip(np.round(quantized_data * (2**bit_depth)), 0, 2**bit_depth-1)
        quantized_data = quantized_data.astype(np.uint16) if bit_depth>8 else quantized_data.astype(np.uint8)

        if isinstance(quantization_points, np.ndarray):
            # If quantization_points is a single array, apply it to all channels
            quantization_points = np.sort(quantization_points)  # Ensure points are sorted
            dequantized_data = quantization_points[quantized_data]
        elif isinstance(quantization_points, list):
            if len(quantization_points) != quantized_data.shape[1]:
                raise ValueError("Length of quantization_points list must match the number of channels (C) in quantized_data.")
            
            dequantized_data = np.zeros_like(quantized_data, dtype=np.float32)
            # Apply different quantization points to each channel
            for i, qp in enumerate(quantization_points):
                qp = np.sort(qp)  # Ensure points are sorted
                channel_data = quantized_data[:, i, :, :]
                dequantized_data[:, i, :, :] = qp[channel_data]
        else:
            raise ValueError("quantization_points must be a numpy array or a list of numpy arrays.")
        
        # print(dequantized_data.dtype)
        dequantized_data = dequantized_data.astype(np.float32)
        return dequantized_data

    @staticmethod
    def packing(feat, model_type):
        N, C, H, W = feat.shape
        if model_type == 'llama3':
            feat = feat[0,0,:,:]
        elif model_type == 'dinov2':
            feat = feat.transpose(0,2,1,3).reshape(N*H,C*W)
        elif model_type == 'sd3':
            feat = feat.reshape(int(C/4), int(C/4), H, W).transpose(0, 2, 1, 3).reshape(int(C/4*H), int(C/4*W)) 
        return feat

    @staticmethod
    def unpacking(feat, shape, model_type):
        N, C, H, W = shape
        if model_type == 'llama3':
            feat = np.expand_dims(feat, axis=0); feat = np.expand_dims(feat, axis=0)
        elif model_type == 'dinov2':
            feat = feat.reshape(N,H,C,W).transpose(0, 2, 1, 3) 
        elif model_type == 'sd3':
            feat = feat.reshape(int(C/4), H, int(C/4), W).transpose(0,2,1,3).reshape(N,C,H,W)
        return feat

    @staticmethod
    def random_crop(feat, crop_shape): # (hight, width)
        max_row = feat.shape[0] - crop_shape[0]
        max_col = feat.shape[1] - crop_shape[1]
        
        if max_row < 0 or max_col < 0:
            print(feat.shape[0], crop_shape[0])
            print(feat.shape[1], crop_shape[1])
            raise ValueError("crop_shape exceeds the feature shape")

        start_row = np.random.randint(0, max_row + 1)
        start_col = np.random.randint(0, max_col + 1)
        
        end_row = start_row + crop_shape[0]
        end_col = start_col + crop_shape[1]
        
        return feat[start_row:end_row, start_col:end_col]
import os
from typing import List

import torch
from torch.utils.data import Dataset

from PIL import Image
from model_utils import load_json_file


class GridDataset(Dataset):
    """
    Dataset class for grid images and their labels.
    Redefined here to ensure compatibility with complex model inputs (like Qwen grid_thw).
    """
    def __init__(self, image_paths, label_dir, processor):
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Return full dictionary (tensors) from processor
        inputs = self.processor(images=[image], return_tensors="pt")
        
        # Squeeze batch dimension added by processor (since DataLoader adds it back)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        image_filename = os.path.basename(image_path)
        label_filename = image_filename.replace('.png', '.json')
        label_path = os.path.join(self.label_dir, label_filename)
        label_data = load_json_file(label_path)
        label_matrix = torch.LongTensor(label_data['matrix'])

        return inputs, label_matrix.flatten()

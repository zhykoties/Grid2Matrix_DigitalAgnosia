import argparse
import os
import json
import glob
import re
import ast
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    AutoImageProcessor, 
    AutoModel, 
    AutoModelForCausalLM, 
    AutoConfig
)

# For Monkeypatching
from transformers.modeling_utils import PreTrainedModel

import wandb
from model_utils import get_model_name_from_path, name_with_datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- SPEED OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

# --- CONSTANTS ---
COLOR_NAMES = [
    "White", "Red", "Blue", "Green", "Orange", 
    "Purple", "Yellow", "Cyan", "Pink", "Brown"
]

class InferenceDataset(Dataset):
    def __init__(self, image_paths, label_paths, num_colors):
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        # Create mapping: {'White': 0, 'Red': 1, ...}
        self.class_mapping = {COLOR_NAMES[i]: i for i in range(num_colors)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        
        with open(lbl_path, 'r') as f:
            label_data = json.load(f)
            
        return {
            "image_path": img_path,
            "image": Image.open(img_path).convert("RGB"),
            "grid_size": label_data.get("grid_size", [3, 3]),
            "class_mapping": self.class_mapping,
            "matrix": np.array(label_data["matrix"], dtype=int) if "matrix" in label_data else None
        }

def collate_fn(batch):
    return batch[0] 

class ZeroShotExperiment:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dtype = torch.bfloat16 if (self.device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        self.model_name = config['model_name']
        
        self.heatmap_sum = None
        self.heatmap_count = 0
        
        # Create output directory immediately
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        print(f"Initializing Model: {self.model_name} on {self.device}...")
        self._init_model()

    def _init_model(self):
        """
        Robust loading logic with Defensive Smart Property.
        Fixes compatibility issues between InternVL custom code and newer transformers library.
        """
        self.is_qwen = "qwen" in self.model_name.lower() and "vl" in self.model_name.lower()
        print(f"Loading model: {self.model_name}...")

        # Patch 1: avoid meta tensor crashes during model init.
        # Force torch.linspace to CPU so .item() works during init
        original_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_linspace(*args, **kwargs)
        torch.linspace = patched_linspace

        # Patch 2: guard all_tied_weights_keys for model compatibility.
        # 1. Defensive Getter: Never return None
        def _get_all_tied_weights_keys(self):
            # Check storage (Qwen Write Path)
            val = getattr(self, "_patched_all_tied_keys_storage", None)
            if val is not None: 
                return val
            
            # Check legacy attribute (InternVL Read Path)
            val = getattr(self, "_tied_weights_keys", None)
            if val is not None:
                return val
            
            # Fallback: Must return empty dict, NEVER None
            return {}

        # 2. Setter: Allow Qwen to save its value
        def _set_all_tied_weights_keys(self, value):
            self._patched_all_tied_keys_storage = value

        # 3. Inject Property
        original_property = getattr(PreTrainedModel, "all_tied_weights_keys", None)
        setattr(PreTrainedModel, "all_tied_weights_keys", property(_get_all_tied_weights_keys, _set_all_tied_weights_keys))

        model_class = AutoModelForCausalLM if self.is_qwen else AutoModel
        
        # Try to resolve architecture from config just to be sure
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            archs = getattr(config, "architectures", [])
            if archs and hasattr(transformers, archs[0]):
                model_class = getattr(transformers, archs[0])
                print(f"Resolved architecture to: {model_class.__name__}")
        except Exception:
            pass

        # Loading options tuned for large VLM checkpoints.
        load_kwargs = {
            "device_map": "auto", 
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True
        }

        try:
            self.model = model_class.from_pretrained(self.model_name, **load_kwargs)
        except Exception as e:
            print(f"Loading failed: {e}. Trying fallback...")
            self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)

        self.model.eval()

        # --- RESTORE ORIGINAL STATE ---
        torch.linspace = original_linspace
        if original_property:
            setattr(PreTrainedModel, "all_tied_weights_keys", original_property)
        else:
            delattr(PreTrainedModel, "all_tied_weights_keys")

        # Load Processors
        if self.is_qwen:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)

    def format_class_mapping(self, mapping):
        """
        Converts dictionary {'White': 0, 'Red': 1} to string "{White: 0, Red: 1}"
        """
        if not mapping: return "{White: 0, Red: 1, Blue: 2}"
        
        # Sort by index (value) to ensure consistent order: 0, 1, 2...
        sorted_items = sorted(mapping.items(), key=lambda item: item[1])
        
        items = [f"{k}: {v}" for k, v in sorted_items]
        return "{" + ", ".join(items) + "}"

    def create_prompt(self, grid_size, class_mapping):
        h, w = grid_size
        mapping_str = self.format_class_mapping(class_mapping)
        return (
            f"You are a precise grid serialization engine.\n"
            f"Task: Transcribe the {h}x{w} pixel grid from the image into a numerical matrix.\n"
            f"Color Mapping: {mapping_str}.\n"
            "\n"
            "Instructions:\n"
            "1. Scan the grid row by row, from top to bottom.\n"
            "2. For each row, map every cell using the color mapping.\n"
            f"3. Ensure the output has exactly {h} rows and {w} columns.\n"
            "\n"
            "Output Format:\n"
            "Return ONLY a Python list of lists (e.g., [[0, 1], [2, 0]]). "
            "Do not use markdown, code blocks, or explanations."
        )

    def predict(self, image, prompt, grid_size):
        h, w = grid_size
        estimated_tokens = (h * w * 4) + (h * 20) + 50 
        max_tokens = self.config.get('max_new_tokens') or min(estimated_tokens, 2048)

        text_output = ""
        
        with torch.inference_mode():
            try:
                if self.is_qwen:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image}, 
                            {"type": "text", "text": prompt},
                        ],
                    }]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Direct PIL image passing
                    inputs = self.processor(
                        text=[text], 
                        images=[image], 
                        videos=None, 
                        padding=True, 
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get("input_ids"), generated_ids)
                    ]
                    text_output = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                else:
                    pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.to(self.model.device, dtype=self.dtype)
                    text_output = self.model.chat(
                        self.tokenizer, pixel_values=pixel_values, question=prompt,
                        generation_config=dict(max_new_tokens=max_tokens, do_sample=False)
                    )
            except Exception as e:
                print(f"Inference error: {e}")
                return ""

        return text_output

    def parse_output(self, output_str, expected_shape):
        if self.config.get("verbose", False):
            print(f"Raw output: {output_str}", flush=True)
        s = output_str.strip().replace("```python", "").replace("```", "").strip()
        H, W = expected_shape
        try:
            m = re.search(r"\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]", s, flags=re.DOTALL)
            if m:
                arr = np.array(ast.literal_eval(m.group(0)), dtype=int)
                if arr.shape == (H, W): return arr
        except: pass

        if "ROW1=" in s:
            rows = []
            try:
                for i in range(1, H + 1):
                    m = re.search(rf"ROW{i}\=\[([0-9,\s-]+)\]", s)
                    if m:
                        nums = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
                        if len(nums) == W: rows.append(nums)
                if len(rows) == H: return np.array(rows)
            except: pass

        nums = re.findall(r"\d+", s)
        if len(nums) == H * W:
            return np.array([int(x) for x in nums]).reshape(H, W)

        return None

    def _format_matrix(self, mat):
        if mat is None: return "None"
        l = mat.tolist() if isinstance(mat, np.ndarray) else mat
        s = str(l)
        return s.replace("],", "],\n      ")

    def _update_and_save_heatmap(self, pred_mat, gt_mat, img_size, step):
        if self.heatmap_sum is None:
            self.heatmap_sum = np.zeros(gt_mat.shape, dtype=float)
        
        if pred_mat is not None and pred_mat.shape == gt_mat.shape:
            match_mask = (pred_mat == gt_mat).astype(float)
        else:
            match_mask = np.zeros(gt_mat.shape, dtype=float)
        
        if match_mask.shape == self.heatmap_sum.shape:
            self.heatmap_sum += match_mask
            self.heatmap_count += 1
        
        if (step + 1) % self.config['log_interval'] == 0:
            self._save_heatmap_file(img_size, step + 1)

    def _save_heatmap_file(self, img_size, step):
        try:
            spatial_acc = self.heatmap_sum / max(self.heatmap_count, 1)
            H, W = spatial_acc.shape
            img_w, img_h = img_size
            
            plt.figure(figsize=(10, 8))
            plt.imshow(spatial_acc, cmap='RdYlGn', vmin=np.min(spatial_acc), vmax=np.max(spatial_acc), extent=[0, W, H, 0])
            
            scale_h = H / img_h
            scale_w = W / img_w
            
            patch_size = 16 
            if H > 5: 
                for y in range(patch_size, img_h, patch_size):
                    plt.axhline(y * scale_h, color='blue', linestyle=':', linewidth=1.0, alpha=0.3)
                for x in range(patch_size, img_w, patch_size):
                    plt.axvline(x * scale_w, color='blue', linestyle=':', linewidth=1.0, alpha=0.3)

            merge_size = 32
            for y in range(merge_size, img_h, merge_size):
                plt.axhline(y * scale_h, color='black', linestyle='--', linewidth=2.0, alpha=0.5)
            for x in range(merge_size, img_w, merge_size):
                plt.axvline(x * scale_w, color='black', linestyle='--', linewidth=2.0, alpha=0.5)

            plt.colorbar()
            plt.title(f"Zero-Shot Spatial Accuracy (N={self.heatmap_count})\nAvg: {spatial_acc.mean():.2%}")
            
            filename = f"heatmap_step{step}_res{img_h}x{img_w}.png"
            save_path = os.path.join(self.config['output_dir'], filename)
            
            plt.savefig(save_path)
            plt.close()
            
            npy_path = os.path.join(self.config['output_dir'], f"heatmap_step{step}_res{img_h}x{img_w}.npy")
            np.save(npy_path, spatial_acc)
            
        except Exception as e:
            print(f"Heatmap generation failed: {e}")

    def run(self, loader):
        results = []
        metrics = {
            "total": 0, "exact_match": 0, "cell_correct": 0, "cell_total": 0, 
            "parse_error": 0, "shape_mismatch": 0
        }
        
        print(f"\n{'='*50}")
        print(f"Starting inference on {len(loader)} samples...")
        print(f"{'='*50}\n")
        
        try:
            for i, sample in enumerate(loader):
                img = sample['image']
                gt_mat = sample['matrix']
                grid_size = sample['grid_size']
                fname = os.path.basename(sample['image_path'])
                
                prompt = self.create_prompt(grid_size, sample['class_mapping'])
                if self.config.get("verbose", False):
                    print(f"Prompt: {prompt}", flush=True)
                raw_output = self.predict(img, prompt, grid_size)
                
                pred_mat = self.parse_output(raw_output, grid_size)
                
                is_exact = False
                cell_acc = 0.0
                status = "[WRONG]"

                current_correct_cells = 0

                if pred_mat is None:
                    metrics['parse_error'] += 1
                    pred_display = "PARSE ERROR"
                elif pred_mat.shape != gt_mat.shape:
                    metrics['shape_mismatch'] += 1
                    pred_display = f"SHAPE MISMATCH {pred_mat.shape}"
                else:
                    is_exact = np.array_equal(pred_mat, gt_mat)
                    current_correct_cells = (pred_mat == gt_mat).sum().item()
                    metrics['exact_match'] += int(is_exact)
                    if is_exact:
                        status = "[CORRECT]"
                    pred_display = self._format_matrix(pred_mat)

                if gt_mat.size > 0:
                    cell_acc = current_correct_cells / gt_mat.size
                else:
                    cell_acc = 0.0
                    
                metrics['cell_correct'] += current_correct_cells
                metrics['cell_total'] += int(gt_mat.size)
                metrics['total'] += 1

                self._update_and_save_heatmap(pred_mat, gt_mat, img.size, i)

                if self.config.get("verbose", False):
                    gt_display = self._format_matrix(gt_mat)
                    print(f"Sample {i+1}/{len(loader)}: {status} {fname}", flush=True)
                    print(f"Acc:  {cell_acc:.2%}", flush=True)
                    print(f"GT:   {gt_display}", flush=True)
                    print(f"Pred: {pred_display}", flush=True)
                    print("-" * 30, flush=True)

                res_entry = {
                    "image_path": sample['image_path'],
                    "status": "success" if is_exact else "failed",
                    "exact_match": bool(is_exact),
                    "cell_acc": cell_acc,
                    "prediction": pred_mat.tolist() if pred_mat is not None else [],
                    "ground_truth": gt_mat.tolist(),
                    "raw_output": raw_output
                }
                results.append(res_entry)

                if self.config['wandb']:
                    if (i + 1) % self.config['log_interval'] == 0:
                        wandb.log({
                            "eval/test_exact": metrics['exact_match'] / metrics['total'], 
                            "eval/test_cell": metrics['cell_correct'] / max(metrics['cell_total'], 1),
                        })
                    
        except KeyboardInterrupt:
            print("\n\nStopping early...")

        final_exact = metrics['exact_match'] / metrics['total'] if metrics['total'] > 0 else 0
        final_cell = metrics['cell_correct'] / metrics['cell_total'] if metrics['cell_total'] > 0 else 0
        
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"Exact Match: {final_exact:.2%}")
        print(f"Cell Acc:    {final_cell:.2%}")
        print(f"Parse Errors:{metrics['parse_error']}")
        print(f"total: {metrics['total']}")
        
        out_file = os.path.join(self.config['output_dir'], f"{self.config['run_name']}_results.json")
        
        with open(out_file, 'w') as f:
            json.dump({"config": self.config, "metrics": metrics, "results": results}, f, indent=2)
        print(f"Saved results to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_dir', required=True)
    
    parser.add_argument('--split', default='test', help="Which split to run inference on")
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=None)    
    parser.add_argument('--num_colors', type=int, default=3)
    parser.add_argument('--device', type=str)
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='grid2matrix')
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--verbose', action='store_true', help='Enable per-sample logging.')

    args = parser.parse_args()
    config = vars(args)

    img_dir = os.path.join(args.dataset_path, args.split, 'images')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    
    if args.max_samples is not None:
        print(f"Limiting inference to first {args.max_samples} samples.")
        img_paths = img_paths[:args.max_samples]
        
    lbl_paths = [p.replace('images', 'labels').replace('.png', '.json') for p in img_paths]
    
    if len(img_paths) == 0:
        raise ValueError(f"No images found in {img_dir}.")

    name_with_date = name_with_datetime()
    if args.wandb_name is None:
        model_short = get_model_name_from_path(args.model_name)
        data_short = os.path.basename(args.dataset_path)
        args.wandb_name = f"ZeroShot-{model_short}-{data_short}-{name_with_date}"
    config['output_dir'] = config['output_dir'] + f'_{name_with_date}'
    config['run_name'] = args.wandb_name

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)

    dataset = InferenceDataset(img_paths, lbl_paths, args.num_colors)
    
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, 
        pin_memory=True, prefetch_factor=2, collate_fn=collate_fn
    )
    
    experiment = ZeroShotExperiment(config)
    experiment.run(loader)

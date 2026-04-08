"""Train linear/spatial probes on frozen VLM visual features."""

import argparse
import os
import math
import glob
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoModel, AutoConfig, AutoImageProcessor
import transformers
from tqdm import tqdm
import wandb

from data_utils import GridDataset
from model_utils import load_json_file, get_model_name_from_path, get_cosine_schedule_with_warmup, name_with_datetime


# --- Feature Extractor ---

class VLMFeatureExtractor(nn.Module):
    """
    Unified wrapper to extract features from Point A (Vision Tower) or Point B (Projector).
    Handles:
    1. Architecture differences (Internal vs External Projector).
    2. Input quirks (filtering kwargs like grid_thw vs num_patches).
    3. Output quirks (Reshaping 2D Qwen features to [B, N, D]).
    """
    def __init__(self, model_name, target_component, device, dtype):
        super().__init__()
        self.target_component = target_component
        self.device = device
        self.dtype = dtype
        self._captured_features = None # Storage for hooked features
        
        # 1. Load Model
        self.model = self._load_model(model_name)
        self.config = self.model.config
        
        # 2. Locate Components
        self.vision_tower = self._find_module(self.model, ['vision_tower', 'vision_model', 'visual'])
        self.projector = self._find_module(self.model, ['mm_projector', 'multi_modal_projector', 'mlp1', 'resampler', 'attn_pool', 'merger'])
        
        # Check if projector is actually inside vision tower (Qwen style)
        self.internal_projector = False
        if not self.projector and self.vision_tower:
            self.projector = self._find_module(self.vision_tower, ['merger', 'attn_pool'])
            if self.projector: self.internal_projector = True

        # 3. Configure Extraction
        if target_component == 'projector' and not self.projector:
            raise ValueError("Target is 'projector' but no projector found.")
        
        # InternVL specific: Pixel Shuffle factor
        self.downsample_ratio = getattr(self.config, 'downsample_ratio', 0.5) if 'InternVL' in model_name else 1.0

        # --- QWEN Specific Fix: Register Hook for Vision Encoder Probe ---
        # If we want the raw vision features (Point A) and the model has an internal merger (like Qwen),
        # we must hook the INPUT to the merger, because the forward pass automatically applies it.
        if self.target_component == 'vision' and self.internal_projector:
            if hasattr(self.vision_tower, 'merger'):
                print("Registering hook to capture Qwen pre-merger states...")
                self.vision_tower.merger.register_forward_hook(self._hook_fn)
        
        # Freeze everything
        for p in self.model.parameters(): p.requires_grad = False

        # Disable Gradient Checkpointing (Silences warnings and optimizes for inference)
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        
        print(f"Extractor Ready: Target={target_component} | InternalProjector={self.internal_projector}")

    def _load_model(self, model_name):
        """
        Robust loading logic with Defensive Smart Property.
        Fixes:
        1. 'meta tensor' crash (via linspace patch).
        2. 'NoneType' crash (via defensive getter).
        3. Read/Write conflict (via setter).
        """
        from transformers.modeling_utils import PreTrainedModel
        
        print(f"Loading model: {model_name}...", flush=True)

        # --- PATCH 1: Fix 'meta tensor' crash ---
        original_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_linspace(*args, **kwargs)
        torch.linspace = patched_linspace

        # --- PATCH 2: Fix 'all_tied_weights_keys' crash ---
        
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

        model = None
        load_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "device_map": None, 
            "low_cpu_mem_usage": False
        }

        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            except Exception as e:
                print(f"AutoModelForCausalLM failed ({e}). Trying architecture from config...")
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                arch = getattr(config, "architectures", [])[0]
                if hasattr(transformers, arch):
                    cls = getattr(transformers, arch)
                    model = cls.from_pretrained(model_name, **load_kwargs)
                else:
                    raise ValueError("Architecture not found in transformers")

        except Exception as e:
            print(f"Fallback to AutoModel due to: {e}")
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
        
        finally:
            # --- RESTORE ORIGINAL STATE ---
            torch.linspace = original_linspace
            
            if original_property:
                setattr(PreTrainedModel, "all_tied_weights_keys", original_property)
            else:
                delattr(PreTrainedModel, "all_tied_weights_keys")

        # Manually move to GPU
        return model.to(self.device)

    def _find_module(self, parent, names):
        for name in names:
            if hasattr(parent, name): return getattr(parent, name)
            # Check one level deep
            if hasattr(parent, 'model') and hasattr(parent.model, name): return getattr(parent.model, name)
            if hasattr(parent, 'transformer') and hasattr(parent.transformer, name): return getattr(parent.transformer, name)
        return None

    def _pixel_shuffle(self, x, scale_factor=0.5):
        """InternVL Pixel Shuffle with CLS handling: [B, N, C] -> [B, N/4, 4C]"""
        b, n, c = x.shape
        
        # Check for CLS token (N = H*W + 1)
        h = w = int(math.sqrt(n))
        if h * w != n:
            # Check if it's square + 1 (CLS)
            h = w = int(math.sqrt(n - 1))
            if h * w == n - 1:
                # Drop CLS token at index 0
                x = x[:, 1:, :]
                n -= 1
            else:
                raise ValueError(f"Invalid shape: {x.shape}")
        
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = F.pixel_unshuffle(x, int(1/scale_factor))
        return x.permute(0, 2, 3, 1).contiguous().view(b, -1, c * int(1/scale_factor)**2)

    def _unshuffle_qwen_blocks(self, x, h, w):
        """
        Converts Qwen3-VL 2x2 Block-Major order to Standard Raster order.
        Input: [B, N, D]
        Output: [B, N, D] (Rasterized)
        """
        b, n, d = x.shape
        # Qwen blocks are 2x2. 
        # Structure in memory: [Block0_0, Block0_1, ...]
        # Each Block has 4 tokens: [TL, TR, BL, BR]
        
        # 1. View as blocks
        # Shape: [B, H//2, W//2, 2, 2, D]
        x = x.view(b, h // 2, w // 2, 2, 2, d)
        
        # 2. Permute to raster order: [B, H//2, 2, W//2, 2, D] -> [B, H, W, D]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, n, d)
        return x

    def _hook_fn(self, module, input, output):
        """Hook to capture the input to the Qwen merger layer."""
        # input[0] is (Total_Pixels, Hidden_Dim) - Pre-merge
        self._captured_features = input[0]

    def forward(self, pixel_values, **kwargs):
        # 1. Filter Kwargs (Whitelist)
        safe_kwargs = {}
        if 'grid_thw' in kwargs: safe_kwargs['grid_thw'] = kwargs['grid_thw']

        # Clear captured features before forward pass
        self._captured_features = None

        # 2. Run Vision Tower
        outputs = self.vision_tower(pixel_values, output_hidden_states=True, **safe_kwargs)
        
        # 3. Parse Output (Handle tuple vs Object)
        last_hidden_state = None
        hidden_states = None
        
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
            if len(outputs) > 1: hidden_states = outputs[1]
        else:
            last_hidden_state = getattr(outputs, 'last_hidden_state', outputs)
            hidden_states = getattr(outputs, 'hidden_states', None)

        # 4. Select Feature Level based on Target
        features = last_hidden_state
        
        if self.target_component == 'vision':
            # Priority 1: Check if we hooked the merger input (Qwen Fix)
            if self._captured_features is not None:
                features = self._captured_features
                
                # Reshape flattened features back to [B, N, D].
                # The hook returns [Total_Pixels, D], causing the ValueError on unpacking later.
                # We must reshape to [B, N, D] immediately.
                if features.dim() == 2:
                    # Infer batch size from input (grid_thw or pixel_values)
                    if 'grid_thw' in safe_kwargs:
                        B = safe_kwargs['grid_thw'].shape[0]
                    else:
                        B = pixel_values.shape[0]
                    features = features.view(B, -1, features.shape[-1])

                # Match merger normalization path when available.
                if hasattr(self.vision_tower.merger, 'norm'):
                    features = self.vision_tower.merger.norm(features)
                elif hasattr(self.vision_tower.merger, 'ln_q'):
                    features = self.vision_tower.merger.ln_q(features)

                # Qwen stores tokens in 2x2 blocks; convert to raster order.
                B, N, D = features.shape
                H = int(math.sqrt(N))
                if H * H == N: # Only unshuffle if valid square
                    features = self._unshuffle_qwen_blocks(features, H, H)

            # Priority 2: Standard hidden states logic (InternVL / fallback)
            elif self.internal_projector and hidden_states is not None:
                features = hidden_states[-1]
            # else: features is already vision output (InternVL Point A)
            else:
                # If no features captured (e.g. model mismatch), this might trigger later errors,
                # but we leave existing logic intact.
                pass

        elif self.target_component == 'projector':
            if not self.internal_projector:
                # Apply External Projector (InternVL/LLaVA Point B)
                if self.downsample_ratio < 1.0:
                    features = self._pixel_shuffle(features, self.downsample_ratio)
                features = self.projector(features)
            # else: features is already projected output (Qwen Point B)

        # 5. Reshape Flattened 2D Outputs [B*N, D] -> [B, N, D]
        # This acts as a safety catch for any path that didn't already reshape
        if features.dim() == 2:
            # Infer batch size from input if possible, else rely on grid_thw
            B = kwargs.get('grid_thw', pixel_values).shape[0]
            features = features.view(B, -1, features.shape[-1])
            
        return features


# --- Probe Models ---

class SpatialProbeModel(nn.Module):
    def __init__(self, feature_extractor, input_dim, grid_size, num_colors, probe_hidden_dim=512, **kwargs):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_cells = grid_size[0] * grid_size[1]
        
        self.probe_head = nn.Sequential(
            nn.Conv2d(input_dim, probe_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(probe_hidden_dim),
            nn.GELU(),
            nn.Conv2d(probe_hidden_dim, num_colors, kernel_size=1),
        )

    def forward(self, pixel_values, **kwargs):
        if self.feature_extractor.training:  # force backbone to eval mode to disable Dropout/BatchNorm updates
            self.feature_extractor.eval()

        with torch.no_grad():
            features = self.feature_extractor(pixel_values, **kwargs)
            B, N, D = features.shape
            
            # Infer spatial structure
            H = int(math.sqrt(N))
            if H * H != N:
                if int(math.sqrt(N-1))**2 == N-1:
                    features = features[:, 1:, :] # Remove CLS
                    N -= 1
                    H = int(math.sqrt(N))
                else:
                    raise ValueError(f"Feature map is not square ({N} tokens).")

        spatial_features = features.permute(0, 2, 1).view(B, D, H, H)

        # Interpolate
        H_grid, W_grid = self.grid_size
        if (H != H_grid) or (H != W_grid):
            spatial_features = F.interpolate(  # gradient in float16 could be noisy
                spatial_features.float(), 
                size=(H_grid, W_grid), 
                mode='bilinear', 
                align_corners=False
            ).to(dtype=spatial_features.dtype)
        
        logits = self.probe_head(spatial_features) 
        return logits.permute(0, 2, 3, 1).view(B, self.num_cells, self.num_colors)


class AggregationProbeModel(nn.Module):
    def __init__(self, feature_extractor, input_dim, num_cells, num_colors, mode='mean', **kwargs):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_cells = num_cells
        self.num_colors = num_colors
        self.mode = mode
        self.probe_head = nn.Linear(input_dim, num_cells * num_colors)

    def forward(self, pixel_values, **kwargs):
        if self.feature_extractor.training:
            self.feature_extractor.eval()

        with torch.no_grad():
            features = self.feature_extractor(pixel_values, **kwargs)
            
            if self.mode == 'cls':
                pooled = features[:, 0, :]
            else: # mean
                start_idx = 1 if features.shape[1] % 2 != 0 else 0
                pooled = torch.mean(features[:, start_idx:, :], dim=1)
        
        normalized = F.normalize(pooled, p=2, dim=1)
        logits = self.probe_head(normalized)
        return logits.view(-1, self.num_cells, self.num_colors)


# --- Experiment Logic ---

class ProbeExperiment:
    def __init__(self, config):
        self.config = config
        # Number of discrete colors in the label grid (classes)
        self.num_colors = config.get('num_colors', 3)
        self.device = config.get('device')
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Capture explicit image dimensions for heatmap scaling.
        self.image_h = config.get('image_height', 512)
        self.image_w = config.get('image_width', 512)
        
        if 'cuda' in self.device:
            self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.compute_dtype = torch.float32
        print(f"Using device: {self.device} | Dtype: {self.compute_dtype}")

        # Initialize Extractor
        self.extractor = VLMFeatureExtractor(config['model_name'], config['target_component'], self.device, self.compute_dtype)
        self.target_component = config['target_component']
        
    def _prepare_batch(self, batch):
        pixel_values = batch.pop('pixel_values').to(self.device, dtype=self.compute_dtype)
        kwargs = {}
        for k, v in batch.items():
            if k == 'image_grid_thw': k = 'grid_thw'
            # Don't cast LongTensor (like grid_thw) to float
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                kwargs[k] = v.to(self.device, dtype=self.compute_dtype)
            elif isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self.device)
            else:
                kwargs[k] = v
        return pixel_values, kwargs

    def train(self, train_loader, valid_loader, test_loader):
        # 1. Inspect Data & Dimensions
        print("Determining input dimensions...")
        dummy_inputs, _ = next(iter(train_loader))
        pv, kw = self._prepare_batch(dummy_inputs)
        
        with torch.no_grad():
            self.extractor.eval()
            feats = self.extractor(pv, **kw)
            input_dim = feats.shape[-1]
        
        # 2. Setup Labels
        # Prefer explicit grid size args; fall back to a label file.
        if self.config['grid_rows'] is not None and self.config['grid_cols'] is not None:
             grid_h, grid_w = self.config['grid_rows'], self.config['grid_cols']
        else:
            sample_label_path = train_loader.dataset.image_paths[0].replace('.png', '.json')
            sample_label = load_json_file(sample_label_path.replace('images', 'labels'))
            grid_h, grid_w = len(sample_label['matrix']), len(sample_label['matrix'][0])
        
        # Persist grid dimensions for evaluate()/heatmap generation.
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Use configured number of colors instead of inferring from data
        num_colors = self.num_colors
        
        print(f"Dims: {input_dim} | Grid: {grid_h}x{grid_w} | Colors: {num_colors} | Image: {self.image_h}x{self.image_w}")

        # 3. Setup Model
        if self.config['mode'] == 'spatial':
            model = SpatialProbeModel(self.extractor, input_dim, (grid_h, grid_w), num_colors, self.config['probe_hidden_dim'])
        else:
            model = AggregationProbeModel(self.extractor, input_dim, grid_h*grid_w, num_colors, self.config['mode'])
        
        model.probe_head.to(self.device, dtype=torch.float32)
        
        # 4. Optimization 
        optimizer = optim.AdamW(model.probe_head.parameters(), 
                                lr=self.config['learning_rate'], 
                                weight_decay=self.config.get('weight_decay', 1e-3))
        
        total_updates = self.config['max_iters']
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    int(total_updates * self.config['warmup_ratio']), 
                                                    total_updates)
        criterion = nn.CrossEntropyLoss()
        
        device_type = self.device.split(':')[0]
        use_mixed_precision = (self.compute_dtype != torch.float32)
        scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
        
        # 5. Loop
        iterator = iter(train_loader)
        best_metric = 0.0
        patience_counter = 0
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        print(f"Starting training: Max Iters={self.config['max_iters']}")

        for step in tqdm(range(1, self.config['max_iters'] + 1)):
            try:
                batch, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch, labels = next(iterator)

            pixel_values, kwargs = self._prepare_batch(batch)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type, dtype=self.compute_dtype):
                logits = model(pixel_values, **kwargs)
                loss = criterion(logits.reshape(-1, num_colors), labels.view(-1))
            
            scaler.scale(loss).backward()

            if self.config.get('grad_clip_norm', 1.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.probe_head.parameters(), self.config.get('grad_clip_norm', 1.0))
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Logging
            metrics = {}
            if self.config['wandb'] and step % 10 == 0:
                metrics['train/loss'] = loss.item()
                metrics['train/lr'] = scheduler.get_last_lr()[0]

            if step % self.config['eval_interval'] == 0:
                valid_exact, valid_cell, valid_color_accs = self.evaluate(model, valid_loader)
                print(f"Step {step} | Val Exact: {valid_exact:.4f} | Val Cell: {valid_cell:.4f}")
                
                metrics['eval/valid_exact'] = valid_exact
                metrics['eval/valid_cell'] = valid_cell
                
                for c, acc in valid_color_accs.items():
                    metrics[f'eval/valid_color_{c}_acc'] = acc
                
                if valid_cell > best_metric:
                    patience_counter = 0
                    best_metric = valid_cell
                    test_exact, test_cell, test_color_accs = self.evaluate(model, test_loader, save_heatmap=True, step=step, output_dir=output_dir)
                    metrics['eval/test_exact'] = test_exact
                    metrics['eval/test_cell'] = test_cell
                    
                    for c, acc in test_color_accs.items():
                        metrics[f'eval/test_color_{c}_acc'] = acc
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['patience']:
                        print(f"Early stopping at step {step}")
                        break

                metrics['eval/best_valid_cell'] = best_metric

            if self.config['wandb'] and metrics:
                wandb.log(metrics, step=step)

        return best_metric

    def evaluate(self, model, loader, save_heatmap=False, step=0, output_dir=None):
        model.eval()
        total_exact_match = 0
        total_cell_correct = 0
        total_cells = 0
        
        # Overall spatial tracking
        heatmap = None
        total_samples = 0
        
        # Per-color tracking (Aggregated and Spatial)
        color_correct = {c: 0 for c in range(self.num_colors)}
        color_total = {c: 0 for c in range(self.num_colors)}
        color_heatmaps = {c: None for c in range(self.num_colors)}
        color_counts = {c: None for c in range(self.num_colors)}
        
        device_type = self.device.split(':')[0]

        with torch.no_grad():
            for batch, labels in loader:
                pv, kw = self._prepare_batch(batch)
                labels = labels.to(self.device)
                
                with torch.amp.autocast(device_type=device_type, dtype=self.compute_dtype):
                    logits = model(pv, **kw)
                
                preds = torch.argmax(logits.float(), dim=2)
                
                # 1. Exact Grid Accuracy
                correct_grids = (preds == labels).all(dim=1).sum().item()
                total_exact_match += correct_grids
                
                # 2. Overall Cell Accuracy
                mask = (preds == labels)
                total_cell_correct += mask.sum().item()
                total_cells += labels.numel()
                
                # 3. Spatial & Color Tracking
                if save_heatmap:
                    batch_heat = mask.float().sum(dim=0)
                    heatmap = batch_heat if heatmap is None else heatmap + batch_heat
                    total_samples += labels.shape[0]

                for c in range(self.num_colors):
                    c_mask = (labels == c)
                    
                    # Aggregate counts
                    color_total[c] += c_mask.sum().item()
                    color_correct[c] += (mask & c_mask).sum().item()
                    
                    # Spatial counts
                    if save_heatmap:
                        c_correct_heat = (mask & c_mask).float().sum(dim=0)
                        c_total_heat = c_mask.float().sum(dim=0)
                        
                        if color_heatmaps[c] is None:
                            color_heatmaps[c] = c_correct_heat
                            color_counts[c] = c_total_heat
                        else:
                            color_heatmaps[c] += c_correct_heat
                            color_counts[c] += c_total_heat

        exact_acc = total_exact_match / len(loader.dataset)
        cell_acc = total_cell_correct / total_cells
        
        color_accs = {}
        for c in range(self.num_colors):
            color_accs[c] = color_correct[c] / color_total[c] if color_total[c] > 0 else 0.0
        
        # --- NEW: Dynamic vmin/vmax and NaN handling ---
        if save_heatmap and output_dir:
            try:
                # 1. Pre-calculate overall accuracy as numpy array
                overall_acc_np = (heatmap / total_samples).cpu().numpy()
                global_min = np.nanmin(overall_acc_np)
                global_max = np.nanmax(overall_acc_np)
                
                # 2. Pre-calculate per-color accuracies and update global bounds
                color_acc_nps = {}
                for c in range(self.num_colors):
                    # Initialize with NaNs to handle empty cells gracefully
                    c_acc_np = np.full(color_heatmaps[c].shape, np.nan)
                    valid_mask = (color_counts[c] > 0).cpu().numpy()
                    
                    if valid_mask.any():
                        c_acc_np[valid_mask] = (color_heatmaps[c].cpu().numpy()[valid_mask] / 
                                                color_counts[c].cpu().numpy()[valid_mask])
                        
                        # Update global min/max bounds
                        global_min = min(global_min, np.nanmin(c_acc_np[valid_mask]))
                        global_max = max(global_max, np.nanmax(c_acc_np[valid_mask]))
                        
                    color_acc_nps[c] = c_acc_np

                # Safety fallback if all accuracies are identical
                if global_min == global_max:
                    global_min, global_max = 0.0, 1.0

                def _save_spatial_heatmap(spatial_acc_np, title_prefix, filename_suffix, vmin, vmax):
                    grid_h, grid_w = self.grid_h, self.grid_w
                    
                    if spatial_acc_np.size == grid_h * grid_w:
                        reshaped_acc = spatial_acc_np.reshape(grid_h, grid_w)
                    else:
                        dim = int(math.sqrt(spatial_acc_np.size))
                        reshaped_acc = spatial_acc_np.reshape(dim, dim)
                        grid_h, grid_w = dim, dim
                    
                    plt.figure(figsize=(10, 8))
                    
                    # --- Custom Colormap for NaNs ---
                    cmap = plt.cm.RdYlGn.copy()
                    cmap.set_bad(color='lightgray') # Color for cells with no data
                    
                    # Apply dynamic bounds
                    plt.imshow(reshaped_acc, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, grid_w, grid_h, 0])
                    
                    scale_h = grid_h / self.image_h
                    scale_w = grid_w / self.image_w
                    model_type = 'internvl' if 'internvl' in self.config['model_name'].lower() else 'qwen'
                    title_extra = ""

                    if model_type == 'qwen':
                        patch_px, merge_px = 16, 32
                        for y in range(patch_px, self.image_h, patch_px):
                            plt.axhline(y * scale_h, color='blue', linestyle=':', linewidth=1.0, alpha=0.3)
                        for x in range(patch_px, self.image_w, patch_px):
                            plt.axvline(x * scale_w, color='blue', linestyle=':', linewidth=1.0, alpha=0.3)
                        if self.target_component == 'projector':
                            for y in range(merge_px, self.image_h, merge_px):
                                plt.axhline(y * scale_h, color='black', linestyle='--', linewidth=2.0, alpha=0.5)
                            for x in range(merge_px, self.image_w, merge_px):
                                plt.axvline(x * scale_w, color='black', linestyle='--', linewidth=2.0, alpha=0.5)
                        title_extra = f"Absolute Steps: {patch_px}px / {merge_px}px"

                    elif model_type == 'internvl':
                        num_raw_patches = 32
                        step_h, step_w = self.image_h / num_raw_patches, self.image_w / num_raw_patches
                        for i in range(1, num_raw_patches):
                            plt.axhline(i * step_h * scale_h, color='blue', linestyle=':', linewidth=1.0, alpha=0.5)
                            plt.axvline(i * step_w * scale_w, color='blue', linestyle=':', linewidth=1.0, alpha=0.5)
                        if self.target_component == 'projector':
                            num_merged = 16
                            step_m_h, step_m_w = self.image_h / num_merged, self.image_w / num_merged
                            for i in range(1, num_merged):
                                plt.axhline(i * step_m_h * scale_h, color='black', linestyle='--', linewidth=2.0, alpha=0.5)
                                plt.axvline(i * step_m_w * scale_w, color='black', linestyle='--', linewidth=2.0, alpha=0.5)
                        title_extra = "Resized Steps: 1/32 (Patch) / 1/16 (Merge)"

                    plt.colorbar()
                    plt.title(f"{title_prefix}\nModel: {model_type} | Input Size: {self.image_h}x{self.image_w}\n({title_extra})")
                    
                    img_size_suffix = f"_{self.image_h}x{self.image_w}"
                    save_path_png = os.path.join(output_dir, f"heatmap_{step}{filename_suffix}{img_size_suffix}.png")
                    save_path_npy = os.path.join(output_dir, f"heatmap_{step}{filename_suffix}{img_size_suffix}.npy")
                    
                    plt.savefig(save_path_png)
                    plt.close()
                    
                    # Fill NaNs with -1.0 before saving to NPY to ensure compatibility, or just save as is
                    np.save(save_path_npy, spatial_acc_np)

                # 3. Generate all plots using the exact same vmin/vmax
                _save_spatial_heatmap(overall_acc_np, f"Overall Accuracy: {cell_acc:.2%}", "", global_min, global_max)

                for c in range(self.num_colors):
                    c_overall_acc = color_accs[c]
                    _save_spatial_heatmap(color_acc_nps[c], f"Color {c} Accuracy: {c_overall_acc:.2%}", f"_color_{c}", global_min, global_max)

            except Exception as e:
                print(f"Heatmap generation failed: {e}")
            
        model.train()
        return exact_acc, cell_acc, color_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_colors', type=int, default=3, help="Number of discrete colors/classes in the grid labels.")
    
    # Image dimensions used for heatmap overlay scaling.
    parser.add_argument('--image_height', type=int, default=512, help="Input image height for correct grid scaling.")
    parser.add_argument('--image_width', type=int, default=512, help="Input image width for correct grid scaling.")

    # Optional override for label-derived matrix dimensions.
    parser.add_argument('--grid_rows', type=int, default=None, help="Override label inference for grid height (rows).")
    parser.add_argument('--grid_cols', type=int, default=None, help="Override label inference for grid width (columns).")
    
    parser.add_argument('--target_component', default='vision', choices=['vision', 'projector'])
    parser.add_argument('--mode', type=str, choices=['mean', 'cls', 'spatial'], default='spatial')
    parser.add_argument('--probe_hidden_dim', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)

    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='grid2matrix')
    parser.add_argument('--wandb_name', default=None)
    
    args = parser.parse_args()
    config = vars(args)
    
    name_with_date = name_with_datetime()
    if args.wandb:
        if args.wandb_name is None:
            model_short = get_model_name_from_path(args.model_name)
            data_short = os.path.basename(args.dataset_path)
            # Add dimensions to run name for clarity
            size_str = f"{args.image_height}x{args.image_width}"
            args.wandb_name = f"{model_short}-{data_short}-{args.target_component}-{args.mode}-{size_str}-{name_with_date}"
            
        wandb.init(project=args.wandb_project, name=args.wandb_name, config={
            "model_name": Path(args.model_name).name,
            "dataset_path": Path(args.dataset_path).name,
            "target_component": args.target_component,
            "mode": args.mode,
            "probe_hidden_dim": args.probe_hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_iters": args.max_iters,
        })
    config['output_dir'] = config['output_dir'] + f'_{name_with_date}'

    processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    def create_loader(split, shuffle):
        path = os.path.join(args.dataset_path, split, 'images')
        labels = os.path.join(args.dataset_path, split, 'labels')
        files = sorted(glob.glob(os.path.join(path, '*.png')))
        ds = GridDataset(files, labels, processor)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    experiment = ProbeExperiment(config)
    experiment.train(create_loader('train', True), create_loader('valid', False), create_loader('test', False))

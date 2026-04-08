"""
Grid2Matrix VLM Benchmark - Data Generation
Generates datasets of grid images and ground-truth matrices with Strict Alignment.
Includes optional Patch/Merger Grid visualization for aliasing checks.
"""

import argparse
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Dict, Any, List
import random

# -------------------------
# Configuration & Constants
# -------------------------

GAP_COLOR = (60, 60, 60) # Dark Gray for grid lines

# Muted/Matte style (VLM-friendly)
PALETTE_DYNAMIC = [
    (255, 255, 255),  # 0: White
    (205, 78, 78),    # 1: Muted Red
    (72, 105, 200),   # 2: Muted Blue
    (85, 168, 104),   # 3: Muted Green
    (221, 163, 85),   # 4: Muted Orange
    (153, 102, 204),  # 5: Muted Purple
    (235, 215, 90),   # 6: Soft Yellow
    (100, 180, 190),  # 7: Muted Cyan/Teal
    (215, 120, 155),  # 8: Muted Pink
    (160, 130, 110)   # 9: Muted Brown/Taupe
]

def set_rw_permissions(path: str):
    """Force file permissions to rw-rw-rw- to avoid Docker/host lock issues."""
    try:
        os.chmod(path, 0o666)
    except Exception:
        pass

def save_json_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def create_output_directories(root_dir: str, splits: List[str]):
    for split in splits:
        os.makedirs(os.path.join(root_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root_dir, split, 'labels'), exist_ok=True)

# -------------------------
# Grid Logic
# -------------------------

def generate_grid_matrix(strategy: str, grid_size: Tuple[int, int], num_colors: int, seed: int) -> np.ndarray:
    """Generates the integer matrix based on the selected pattern strategy."""
    rng = np.random.default_rng(seed)
    height, width = grid_size
    
    if strategy == 'random':
        # Ensure all colors appear at least once if enough cells exist
        total_cells = height * width
        grid_flat = rng.integers(0, num_colors, size=total_cells)
        if num_colors <= total_cells:
            for c in range(num_colors):
                grid_flat[c] = c
            rng.shuffle(grid_flat)
        return grid_flat.reshape((height, width))
        
    elif strategy == 'stripes':
        grid = np.zeros((height, width), dtype=int)
        for i in range(height):
            grid[i, :] = i % num_colors
        return grid
        
    elif strategy == 'vertical_stripes':
        grid = np.zeros((height, width), dtype=int)
        for j in range(width):
            grid[:, j] = j % num_colors
        return grid
        
    elif strategy == 'checkerboard':
        grid = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                grid[i, j] = (i + j) % num_colors
        return grid
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# -------------------------
# Rendering Logic
# -------------------------

def get_color(index: int, num_colors: int) -> Tuple[int, int, int]:
    """Retrieves color from the unified muted palette."""
    colors = PALETTE_DYNAMIC[:]
    
    # If we need more colors than defined, generate random muted ones
    while len(colors) < num_colors:
        colors.append((random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    
    return colors[index] if 0 <= index < len(colors) else (255, 255, 255)

def overlay_patch_viz(image: Image.Image, patch_size: int = 16) -> Image.Image:
    """
    Overlays VLM patch boundaries for aliasing checks.
    - Light Green Lines (Thin): Patch Boundaries (e.g., 16x16)
    - Yellow Lines (Thick): Merger Boundaries (e.g., 32x32, usually 2x2 patches)
    """
    # Create an RGBA overlay to support transparency
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    
    merge_size = patch_size * 2 # Usually pooling layers combine 2x2 patches

    # 1. Patch Grid (Light Green, Thin)
    # RGBA: Bright Green with somewhat high opacity
    patch_color = (100, 255, 100, 200)
    
    # Draw Vertical Lines
    for x in range(patch_size, w, patch_size):
        draw.line([(x, 0), (x, h)], fill=patch_color, width=1)
    # Draw Horizontal Lines
    for y in range(patch_size, h, patch_size):
        draw.line([(0, y), (w, y)], fill=patch_color, width=1)

    # 2. Merger Grid (Yellow, Thicker)
    # RGBA: Bright Yellow with high opacity
    """
    merger_color = (255, 255, 0, 220)
    
    # Draw Vertical Lines
    for x in range(merge_size, w, merge_size):
        draw.line([(x, 0), (x, h)], fill=merger_color, width=3)
    # Draw Horizontal Lines
    for y in range(merge_size, h, merge_size):
        draw.line([(0, y), (w, y)], fill=merger_color, width=3)
    """

    # Composite the overlay onto the original image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    return Image.alpha_composite(image, overlay).convert('RGB')

def render_grid(matrix: np.ndarray, image_res: Tuple[int, int], num_colors: int) -> Image.Image:
    """
    Renders the grid using Strict Alignment.
    """
    H, W = matrix.shape
    target_W, target_H = image_res
    
    # Check for perfect divisibility
    if target_W % W != 0 or target_H % H != 0:
        pass 
        
    img = Image.new('RGB', (target_W, target_H), color='white')
    draw = ImageDraw.Draw(img)
    
    cell_w = target_W / W
    cell_h = target_H / H
    
    # Dynamic Line Width Calculation
    min_cell_dim = min(cell_w, cell_h)
    
    if min_cell_dim < 4:
        line_width = 0 
    elif min_cell_dim < 20:
        line_width = 2
    else:
        line_width = max(2, min(int(min_cell_dim * 0.05), 10))

    for i in range(H):
        for j in range(W):
            color = get_color(matrix[i, j], num_colors)
            
            x1 = int(round(j * cell_w))
            y1 = int(round(i * cell_h))
            x2 = int(round((j + 1) * cell_w))
            y2 = int(round((i + 1) * cell_h))
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=GAP_COLOR, width=line_width)
            
    return img

# -------------------------
# Main Generation Loop
# -------------------------

def generate_split(config: Dict[str, Any], split_name: str, num_samples: int, start_seed: int):
    """Generates a single split (train/valid/test)."""
    
    output_dir = config['output_dir']
    grid_size = tuple(config['grid_size'])
    num_colors = config['num_colors']
    image_res = tuple(config['image_resolution'])
    strategy = config['pattern_strategy']
    visualize = config.get('visualize_patches', False) # Check flag

    print(f"Generating {split_name}: {num_samples} samples | Strategy: {strategy} | Colors: {num_colors}")

    for i in range(num_samples):
        current_seed = start_seed + i
        
        # 1. Generate Matrix
        matrix = generate_grid_matrix(strategy, grid_size, num_colors, current_seed)
        
        # 2. Render Image (Strict Alignment)
        img = render_grid(matrix, image_res, num_colors)
        
        # 3. [OPTIONAL] Overlay Patch Viz
        if visualize:
            # We assume patch_size=16 as per your snippet. 
            img = overlay_patch_viz(img, patch_size=16)

        # 4. Save Files
        filename = f"{i:04d}"
        img_path = os.path.join(output_dir, split_name, 'images', f"{filename}.pdf")
        lab_path = os.path.join(output_dir, split_name, 'labels', f"{filename}.json")
        
        img.save(img_path)
        set_rw_permissions(img_path)
        
        label_data = {
            'matrix': matrix.tolist(),
            'grid_size': grid_size,
            'num_colors': num_colors,
            'pattern_strategy': strategy,
            'image_resolution': image_res,
            'split': split_name,
            'seed': current_seed
        }
        save_json_file(label_data, lab_path)
        set_rw_permissions(lab_path)

def main():
    parser = argparse.ArgumentParser(description='Generate aligned grid datasets for VLM benchmark')

    # Core Args
    parser.add_argument('--output_dir', type=str, required=True, help='Root output directory')
    
    # Overrides / Direct Configuration
    parser.add_argument('--grid_size', type=int, nargs=2, default=[32, 32], help='Grid H W')
    parser.add_argument('--image_resolution', type=int, nargs=2, default=[512, 512], help='Image W H')
    parser.add_argument('--num_colors', type=int, default=3)
    parser.add_argument('--pattern_strategy', type=str, default='random', choices=['random', 'stripes', 'checkerboard', 'vertical_stripes'])
    
    # Debug / Visualization
    parser.add_argument('--visualize_patches', action='store_true', help='Overlay 16x16 patch grid (Green) and 32x32 merger grid (Yellow) for debugging alignment.')
    
    # Split Sizes
    parser.add_argument('--train_count', type=int, default=1)
    parser.add_argument('--val_count', type=int, default=1)
    parser.add_argument('--test_count', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    config = vars(args) 

    # 1. Setup Directories
    splits = ['train', 'valid', 'test']
    create_output_directories(args.output_dir, splits)

    # 2. Check for Alignment (Warning Only)
    h, w = args.grid_size
    img_w, img_h = args.image_resolution
    if img_w % w != 0 or img_h % h != 0:
        print(f"\n[WARNING] Strict Alignment Check:")
        print(f"Grid {args.grid_size} does not divide evenly into Res {args.image_resolution}.")
        print(f"Cells will be approx ({img_w/w:.2f} x {img_h/h:.2f}). This may introduce minor aliasing.\n")
    else:
        print(f"\n[INFO] Strict Alignment Check Passed.")
        print(f"Each cell will be exactly {img_w//w}x{img_h//h} pixels with dynamic borders.\n")

    # 3. Generate Splits
    seed_cursor = args.seed
    
    if args.train_count > 0:
        generate_split(config, 'train', args.train_count, seed_cursor)
        seed_cursor += args.train_count
        
    if args.val_count > 0:
        generate_split(config, 'valid', args.val_count, seed_cursor)
        seed_cursor += args.val_count
        
    if args.test_count > 0:
        generate_split(config, 'test', args.test_count, seed_cursor)

    print(f"\n✅ Dataset generation complete at {args.output_dir}")

if __name__ == '__main__':
    main()

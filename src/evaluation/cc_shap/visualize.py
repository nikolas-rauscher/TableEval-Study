import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from typing import Tuple


def image_patch_heatmap_overlay(image: Image.Image, shap_values, num_patches_p: int, alpha: float = 0.5) -> Image.Image:
    """Create a simple overlay heatmap for image patches from SHAP values.

    - Aggregates absolute SHAP values over outputs for image features (first p^2 features)
    - Reshapes to p×p, upsamples to image size, overlays heatmap
    """
    if not hasattr(shap_values, 'values'):
        return image
    vals = shap_values.values
    if vals.ndim == 2:
        vals = np.expand_dims(vals, axis=2)
    if vals.ndim != 3:
        return image
    p2 = num_patches_p * num_patches_p
    img_contrib = np.abs(vals[0, :p2, :]).sum(axis=1)  # (p2,)
    if img_contrib.sum() == 0:
        heat = np.zeros((num_patches_p, num_patches_p), dtype=float)
    else:
        heat = img_contrib.reshape(num_patches_p, num_patches_p)
        heat = heat / (heat.max() + 1e-9)

    # Upsample heatmap to image size
    W, H = image.size
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
    heat_rgb = plt.cm.jet(np.array(heat_img) / 255.0)[:, :, :3]  # drop alpha
    heat_rgb = (heat_rgb * 255).astype(np.uint8)
    heat_overlay = Image.fromarray(heat_rgb)
    return Image.blend(image.convert('RGB'), heat_overlay, alpha)


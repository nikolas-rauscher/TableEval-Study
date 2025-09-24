import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


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


def save_text_contribution_plot(
    shap_plot_info: Dict[str, Any],
    out_path: str,
    title: Optional[str] = None,
    max_tokens: Optional[int] = 80,
):
    """Save a side-by-side bar plot for token contributions (prediction vs. explanation).

    Expects shap_plot_info with keys: 'input_labels', 'ratios_prediction', 'ratios_explanation'.
    """
    labels = shap_plot_info.get('input_labels', [])
    rp = shap_plot_info.get('ratios_prediction', [])
    re = shap_plot_info.get('ratios_explanation', [])

    if not labels or not rp or not re:
        return

    n = min(len(labels), len(rp), len(re))
    labels = labels[:n]
    rp = rp[:n]
    re = re[:n]

    if max_tokens is not None:
        labels = labels[:max_tokens]
        rp = rp[:max_tokens]
        re = re[:max_tokens]

    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(max(8, len(labels) * 0.15), 4))
    plt.bar(x - width/2, rp, width, label='Prediction', alpha=0.8)
    plt.bar(x + width/2, re, width, label='Explanation', alpha=0.8)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('Contribution ratio (%)')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


import numpy as np
from typing import Tuple, Dict, Any
from scipy import spatial, stats, special
from sklearn import metrics


def compute_mm_score(num_text_tokens: int, shap_values) -> float:
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        return 0.5
    vals = shap_values.values
    if vals.ndim != 3:
        return 0.5
    num_image_features = vals.shape[1] - int(num_text_tokens)
    if num_image_features < 0:
        return 0.5
    image_contrib = np.abs(vals[0, :num_image_features, :]).sum()
    text_contrib = np.abs(vals[0, num_image_features:, :]).sum()
    denom = image_contrib + text_contrib
    if denom == 0 or not np.isfinite(denom):
        return 0.5
    return float(text_contrib / denom)


def aggregate_values_prediction(shap_values) -> np.ndarray:
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        return np.array([])
    vals = shap_values.values
    if vals.ndim == 2:
        vals = np.expand_dims(vals, axis=2)
    if vals.ndim != 3:
        return np.array([])
    mean_per_input = np.mean(vals[0], axis=1)
    total_abs = np.abs(mean_per_input).sum()
    if total_abs == 0 or not np.isfinite(total_abs):
        return np.zeros_like(mean_per_input)
    return mean_per_input / (total_abs + 1e-9) * 100.0


def aggregate_values_explanation(shap_values, tokenizer, to_marginalize: str) -> np.ndarray:
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        return np.array([])
    vals = shap_values.values
    if vals.ndim == 2:
        vals = np.expand_dims(vals, axis=2)
    if vals.ndim != 3:
        return np.array([])

    # Derive number of tokens to marginalize by encoding without specials
    try:
        marginal_ids = tokenizer.encode(to_marginalize, add_special_tokens=False)
        L = len(marginal_ids)
    except Exception:
        L = 0
    if L > 0:
        add_to_base = np.abs(vals[:, -L:]).sum(axis=1)
        denom = (np.abs(vals).sum(axis=1) - add_to_base)
        denom = np.where(denom == 0, 1.0, denom)
        ratios = vals / (denom + 1e-9) * 100.0
        out = np.mean(ratios, axis=2)[0, :-L]
    else:
        total_abs = np.abs(vals).sum(axis=1)
        total_abs = np.where(total_abs == 0, 1.0, total_abs)
        ratios = vals / (total_abs + 1e-9) * 100.0
        out = np.mean(ratios, axis=2)[0]
    return out


def cc_shap_score(ratios_prediction: np.ndarray, ratios_explanation: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    rp = np.asarray(ratios_prediction)
    re = np.asarray(ratios_explanation)
    if rp.size == 0 or re.size == 0 or rp.shape != re.shape:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if not np.all(np.isfinite(rp)) or not np.all(np.isfinite(re)):
        rp = np.nan_to_num(rp)
        re = np.nan_to_num(re)

    try:
        cosine_dist = spatial.distance.cosine(rp, re)
    except Exception:
        cosine_dist = np.nan
    try:
        if np.std(rp) > 1e-9 and np.std(re) > 1e-9:
            dist_corr = spatial.distance.correlation(rp, re)
        else:
            dist_corr = np.nan
    except Exception:
        dist_corr = np.nan
    try:
        mse = metrics.mean_squared_error(rp, re)
    except Exception:
        mse = np.nan
    try:
        var_diff = np.var(rp - re)
    except Exception:
        var_diff = np.nan
    try:
        kl_div = stats.entropy(special.softmax(re), special.softmax(rp))
    except Exception:
        kl_div = np.nan
    try:
        js_div = spatial.distance.jensenshannon(special.softmax(rp), special.softmax(re))
    except Exception:
        js_div = np.nan

    return cosine_dist, dist_corr, mse, var_diff, kl_div, js_div


def compute_cc_shap(
    values_prediction,
    values_explanation,
    tokenizer,
    num_patches_p: int,
    num_text_tokens_pred: int,
    num_text_tokens_expl: int,
    marg_pred_str: str,
    marg_expl_str: str,
) -> Tuple[float, float, float, float, float, float, Dict[str, Any]]:
    if marg_pred_str:
        ratios_pred = aggregate_values_explanation(values_prediction, tokenizer, marg_pred_str)
    else:
        ratios_pred = aggregate_values_prediction(values_prediction)
    ratios_expl = aggregate_values_explanation(values_explanation, tokenizer, marg_expl_str)

    cosine_dist, dist_corr, mse, var_diff, kl_div, js_div = cc_shap_score(ratios_pred, ratios_expl)

    # Build minimal plot info
    input_ids_pred = getattr(values_prediction, 'data', None)
    input_ids_expl = getattr(values_explanation, 'data', None)
    labels = []
    try:
        if input_ids_pred is not None:
            ids = input_ids_pred[0].tolist()[-num_text_tokens_pred:]
            labels = [tokenizer.decode([x], skip_special_tokens=False) for x in ids]
    except Exception:
        labels = []

    shap_plot_info = {
        'input_labels': labels,
        'ratios_prediction': ratios_pred.astype(float).round(2).tolist() if ratios_pred.size else [],
        'ratios_explanation': ratios_expl.astype(float).round(2).tolist() if ratios_expl.size else [],
        'num_patches_p': num_patches_p,
        'num_text_tokens_pred': num_text_tokens_pred,
        'num_text_tokens_expl': num_text_tokens_expl,
    }

    return cosine_dist, dist_corr, mse, var_diff, kl_div, js_div, shap_plot_info


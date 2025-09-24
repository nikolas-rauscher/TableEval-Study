import math
import numpy as np
import torch
from typing import Optional, Tuple, List
from PIL import Image
from tqdm.auto import tqdm

import shap


def _find_image_token_sequence(tokenizer, input_ids: List[int], image_token_text: str) -> List[int]:
    """Find the actual token id sequence for the special image token text in a list of input ids."""
    try:
        seq = tokenizer.encode(image_token_text, add_special_tokens=False)
    except Exception:
        return []
    if not seq:
        return []
    for k in range(0, len(input_ids) - len(seq) + 1):
        if input_ids[k : k + len(seq)] == seq:
            return seq
    return []


def _build_masker(num_patches: int,
                  num_text_tokens: int,
                  pad_token_id: int,
                  image_sequence_ids: List[int],
                  bos_id: Optional[int],
                  eos_id: Optional[int]):
    """Create a SHAP masker that preserves BOS/EOS and the image token sequence, masks image patches and text tokens.

    The feature vector x is expected to be: [patch_placeholders (num_patches), text_token_ids (num_text_tokens)].
    """
    def custom_masker(mask, x):
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).bool()
        else:
            mask_tensor = mask.bool()

        if x.ndim == 1:
            x = x.unsqueeze(0)
        masked_X = x.clone()  # CPU tensors

        # Align mask dims to X
        if mask_tensor.ndim == 1:
            current_mask = mask_tensor.unsqueeze(0).expand_as(masked_X)
        elif mask_tensor.shape[0] == 1 and masked_X.shape[0] > 1:
            current_mask = mask_tensor.expand_as(masked_X)
        elif mask_tensor.shape[0] != masked_X.shape[0]:
            current_mask = mask_tensor[0:1].expand_as(masked_X)
        else:
            current_mask = mask_tensor

        # Do-not-mask flags for special text tokens
        dont_mask_flags = torch.zeros_like(current_mask, dtype=torch.bool)
        original_text_tokens = x[:, num_patches:]
        tokens_to_keep = []
        if bos_id is not None and bos_id >= 0:
            tokens_to_keep.append(bos_id)
        if eos_id is not None and eos_id >= 0:
            tokens_to_keep.append(eos_id)
        tokens_to_keep.extend(image_sequence_ids or [])
        if tokens_to_keep:
            keep_ids = torch.tensor(sorted(set(tokens_to_keep)))
            # Broadcast compare across batch
            keep_mask = (original_text_tokens.unsqueeze(-1) == keep_ids).any(dim=-1)
            dont_mask_flags[:, num_patches:] |= keep_mask

        # Final keep mask: keep if SHAP says keep OR token is special
        final_keep_mask = current_mask | dont_mask_flags

        # Apply masking
        mask_value_text = 0 if pad_token_id == -100 else pad_token_id
        mask_value_img = 0
        # Default: mask all positions where final_keep_mask is False
        masked_X.masked_fill_(~final_keep_mask, mask_value_text)
        # Specifically ensure image placeholders are 0 when masked
        masked_X[:, :num_patches].masked_fill_(~final_keep_mask[:, :num_patches], mask_value_img)

        return masked_X

    return custom_masker


def _build_predictor(model,
                     device: str,
                     original_inputs_cpu,
                     num_patches: int,
                     num_text_tokens: int,
                     target_output_ids_cpu: torch.Tensor,
                     pad_token_id: int):
    """Create the SHAP prediction function. It constructs masked inputs and returns logits for target tokens.

    - x has shape (B, num_patches + num_text_tokens)
    - first num_patches entries encode which image patches are present (non-zero) or masked (zero)
    - remaining entries are text token ids; masked ones should be turned to PAD
    """
    # Cache vision features on device
    vision_keys = [k for k in original_inputs_cpu.keys() if 'pixel' in k or 'image' in k or 'vision' in k]
    if 'pixel_values' not in vision_keys and 'pixel_values' in original_inputs_cpu:
        vision_keys.append('pixel_values')

    original_vision_on_device = {}
    for v_key in vision_keys:
        val = original_inputs_cpu.get(v_key, None)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            try:
                target_dtype = getattr(model, 'dtype', None) or val.dtype
                original_vision_on_device[v_key] = val.to(device).to(target_dtype)
            except Exception:
                original_vision_on_device[v_key] = val.to(device)
        else:
            original_vision_on_device[v_key] = val

    def get_model_prediction(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to('cpu')

        num_permutations = x.shape[0]
        num_output_tokens = target_output_ids_cpu.shape[1]
        results = np.zeros((num_permutations, num_output_tokens))

        with torch.no_grad():
            for i in range(num_permutations):
                current_x = x[i : i + 1, :]
                # Text
                item_input_ids = current_x[:, num_patches:].to(torch.long)
                item_attn_mask = (item_input_ids != pad_token_id).long()
                # Determine if image is included (any non-zero patch placeholder)
                include_image = (current_x[0, :num_patches] != 0).any().item()

                # Build batch
                batch = {
                    'input_ids': item_input_ids.to(device),
                    'attention_mask': item_attn_mask.to(device),
                }
                if include_image:
                    # Recreate pixel_values with patches masked by zeros
                    if 'pixel_values' in original_vision_on_device:
                        pv = original_vision_on_device['pixel_values'].clone()

                        original_shape = pv.shape
                        if pv.ndim == 4:
                            views = [pv]
                            restore = lambda: pv
                        elif pv.ndim == 5:
                            num_images = pv.shape[1]
                            views = [pv[:, img_idx] for img_idx in range(num_images)]
                            restore = lambda: pv  # modifications already applied to pv
                        else:
                            pv_reshaped = pv.reshape(pv.shape[0], pv.shape[-3], pv.shape[-2], pv.shape[-1])
                            views = [pv_reshaped]
                            restore = lambda: pv_reshaped.reshape(original_shape)

                        p_now = int(round(math.sqrt(num_patches)))
                        for view in views:
                            if view.ndim != 4:
                                continue
                            _, _, H, W = view.shape
                            patch_h = H // p_now
                            patch_w = W // p_now
                            for k in range(num_patches):
                                if current_x[0, k].item() == 0:
                                    m = k // p_now
                                    n = k % p_now
                                    h0, h1 = m * patch_h, (m + 1) * patch_h
                                    w0, w1 = n * patch_w, (n + 1) * patch_w
                                    view[:, :, h0:h1, w0:w1] = 0

                        pv = restore()
                        batch['pixel_values'] = pv
                    # Copy any other vision keys if present
                    for v_key, v_tensor in original_vision_on_device.items():
                        if v_key == 'pixel_values':
                            continue
                        batch[v_key] = v_tensor

                outputs = model(**batch, return_dict=True)
                if not hasattr(outputs, 'logits') or outputs.logits is None:
                    results[i] = 0.0
                    continue

                item_logits = outputs.logits.detach().cpu()[0]
                input_len = item_input_ids.shape[1]
                # Next-token logits start at input_len-1
                start = input_len - 1
                max_len = item_logits.shape[0]
                available = max_len - start
                eff = min(num_output_tokens, max(0, available))
                if start < 0 or eff <= 0:
                    results[i] = 0.0
                    continue
                sliced = item_logits[start : start + eff, :]
                target_ids = target_output_ids_cpu.numpy()[0]
                try:
                    vals = np.array([sliced[t, target_ids[t]].item() for t in range(eff)])
                    results[i, :eff] = vals
                    if eff < num_output_tokens:
                        results[i, eff:] = 0.0
                except Exception:
                    results[i] = 0.0

        return results

    return get_model_prediction


def explain_vlm_with_patches(
    prompt_text: str,
    raw_image: Image.Image,
    model_wrapper,
    target_output_ids: Optional[torch.Tensor] = None,
    p: Optional[int] = None,
    num_evals: Optional[int] = 600,
    max_new_tokens: int = 1,
    patch_grid: Optional[int] = None,
    max_patch_grid: Optional[int] = None,
):
    """Explain a VLM (HFModel) prediction using CC-SHAP-style patch masking.

    Returns (shap_values, mm_score, p, num_text_tokens, target_output_ids)
    """
    model = model_wrapper.model
    processor = model_wrapper.processor
    tokenizer = getattr(model_wrapper, 'tokenizer', None) or processor.tokenizer
    device = model_wrapper.device
    image_token_text = model_wrapper.image_token

    # Build original inputs once
    full_prompt = image_token_text + prompt_text
    original_inputs_cpu = processor(
        text=full_prompt,
        images=raw_image,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(device)

    input_ids_list = original_inputs_cpu['input_ids'][0].tolist()
    # Identify image sequence ids for protection
    image_seq_ids = _find_image_token_sequence(tokenizer, input_ids_list, image_token_text)

    # Derive p from text length if not provided
    num_text_tokens = original_inputs_cpu['input_ids'].shape[1]
    if p is None:
        if patch_grid is not None:
            p = max(1, int(patch_grid))
        else:
            # Heuristic from CC-SHAP: balance feature counts
            p = max(1, int(math.ceil(math.sqrt(max(1, num_text_tokens - len(image_seq_ids))))))
            if max_patch_grid is not None:
                p = min(p, max(1, int(max_patch_grid)))
    num_patches = p * p

    # Prepare target outputs: if not provided, generate with the model
    if target_output_ids is None:
        gen_kwargs = dict(max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True)
        try:
            gen_out = model.generate(**original_inputs_cpu, **gen_kwargs)
            # Extract generated ids after the input length
            gen_ids = gen_out.sequences[:, original_inputs_cpu['input_ids'].shape[1]:]
            if gen_ids is None or gen_ids.shape[1] == 0:
                # Fallback to single token
                target_output_ids = tokenizer(" ", return_tensors='pt', add_special_tokens=False).input_ids[:, :1]
            else:
                target_output_ids = gen_ids.to('cpu')
        except Exception:
            # Fallback to a single token to avoid failure
            target_output_ids = tokenizer(" ", return_tensors='pt', add_special_tokens=False).input_ids[:, :1]
    target_output_ids_cpu = target_output_ids.to('cpu')

    # Build feature vector X = [patch placeholders; input_ids]
    patch_placeholders = torch.arange(-1, -num_patches - 1, -1).unsqueeze(0)  # negative ids just as placeholders
    X = torch.cat((patch_placeholders, original_inputs_cpu['input_ids'].to('cpu')), dim=1)
    num_features_total = X.shape[1]
    min_evals_needed = 2 * num_features_total + 1
    
    if num_evals is None or num_evals < min_evals_needed:
        new_evals = min_evals_needed
        print(
            f"[CC-SHAP] Feature count {num_features_total} requires at least {min_evals_needed} evaluations. "
            f"Adjusting max_evals from {num_evals if num_evals is not None else 'None'} to {new_evals}."
        )
        num_evals = new_evals
    elif num_evals == -1:  # Special flag to use exact minimum
        num_evals = min_evals_needed
        print(f"[CC-SHAP] Using exact minimum: {min_evals_needed} evaluations for {num_features_total} features.")
    else:
        print(f"[CC-SHAP] Feature count {num_features_total}, using max_evals={num_evals}.")

    # Special token ids
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    pad_id = getattr(tokenizer, 'pad_token_id', -100) or -100

    # SHAP components
    masker = _build_masker(
        num_patches=num_patches,
        num_text_tokens=num_text_tokens,
        pad_token_id=pad_id,
        image_sequence_ids=image_seq_ids,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    predictor = _build_predictor(
        model=model,
        device=device,
        original_inputs_cpu=original_inputs_cpu,
        num_patches=num_patches,
        num_text_tokens=num_text_tokens,
        target_output_ids_cpu=target_output_ids_cpu,
        pad_token_id=pad_id,
    )

    # Wrap predictor to count evaluations
    eval_counter = {'count': 0}
    
    def counting_predictor(*args, **kwargs):
        result = predictor(*args, **kwargs)
        eval_counter['count'] += 1
        
        # Update progress for every evaluation
        progress = (eval_counter['count'] / num_evals) * 100
        print(f"\rSHAP Progress: {eval_counter['count']}/{num_evals} ({progress:.1f}%)", end="", flush=True)
        
        return result
    
    print(f"Starting SHAP evaluation with {num_evals} evaluations...")
    explainer = shap.Explainer(counting_predictor, masker, silent=False)
    shap_values = explainer(X, max_evals=num_evals)
    print(f"\nSHAP evaluation completed! ({eval_counter['count']} evaluations used)")

    # Normalize SHAP values shape to (1, num_features, num_output_tokens)
    if hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray):
        if shap_values.values.ndim == 2:
            shap_values.values = np.expand_dims(shap_values.values, axis=2)
        if shap_values.values.ndim == 3 and shap_values.values.shape[0] != 1:
            shap_values.values = shap_values.values[0:1]
            if hasattr(shap_values, 'base_values') and isinstance(shap_values.base_values, np.ndarray):
                if shap_values.base_values.ndim > 0 and shap_values.base_values.shape[0] > 1:
                    shap_values.base_values = shap_values.base_values[0:1]
            if hasattr(shap_values, 'data') and isinstance(shap_values.data, np.ndarray):
                if shap_values.data.ndim > 1 and shap_values.data.shape[0] > 1:
                    shap_values.data = shap_values.data[0:1]
    else:
        # build dummy in worst case
        dummy_vals = np.zeros((1, X.shape[1], target_output_ids_cpu.shape[1]))
        dummy_base = np.zeros(target_output_ids_cpu.shape[1])
        shap_values = shap.Explanation(values=dummy_vals, base_values=dummy_base, data=X.numpy())

    # MM score: share of text contribution
    mm_score = compute_mm_score(num_text_tokens=num_text_tokens, shap_values=shap_values)

    return shap_values, mm_score, p, num_text_tokens, target_output_ids_cpu


def compute_mm_score(num_text_tokens: int, shap_values) -> float:
    """Compute Multimodality Score = text_contrib / (image + text contrib)."""
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        return 0.5
    vals = shap_values.values
    if vals.ndim != 3:
        return 0.5
    num_input_features = vals.shape[1]
    num_image_features = num_input_features - int(num_text_tokens)
    if num_image_features < 0:
        return 0.5
    image_contrib = np.abs(vals[0, :num_image_features, :]).sum()
    text_contrib = np.abs(vals[0, num_image_features:, :]).sum()
    denom = image_contrib + text_contrib
    if denom == 0 or not np.isfinite(denom):
        return 0.5
    return float(text_contrib / denom)

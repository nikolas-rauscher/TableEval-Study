## Interpretability analyses

### Inseq (Text-only)
Models tested:
* [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
* [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

Example of generating saliency maps for ComTQA-PMC using Mistral-Nemo and constraining the analysis to two instances (IDs 49 and 101):  
```bash
python ./src/evaluation/explain_llm_predictions.py \
  --input_file comtqa_pmc_results.json \
  --model_id mistralai/Mistral-Nemo-Instruct-2407 \
  --source_data_path data/ComTQA_data/comtqa_pmc_updated_2025-03-07 \
  --instance_ids 49 \
  --instance_ids 101
```
Results are in subdirectories of explanations/inseq/

---

### MLLM (CC‑SHAP)
Models tested:
* llava-hf/llava-v1.6-mistral-7b-hf

Example for LogicNLG-Image with LLaVA (subset of 10):
```bash
python ./src/evaluation/explain_mllm_predictions.py \
  --input_file logicnlg_image_results.json \
  --model_id llava-hf/llava-v1.6-mistral-7b-hf \
  --source_data_path data/LogicNLG/logicnlg_updated_2025-03-13 \
  --subset_size 10 \
  --shap_num_evals 600 \
  --output_dir ./explanations/cc_shap
```
Results and optional overlays are saved in subdirectories of explanations/cc_shap/

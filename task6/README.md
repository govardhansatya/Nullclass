# Task 6: Stable Diffusion LoRA Fine-Tuning

Fine-tunes a Stable Diffusion model using Low-Rank Adaptation (LoRA) on a small custom (example) dataset (Pokemon captions). Produces adapted weights to specialize generation style/content.

## Files
- `task6_diffusion_finetune.ipynb`: Main LoRA training notebook.
- `finetune_stable_diffusion_lora.ipynb`: (If present) alternative / extended workflow.
- `metrics_task6.ipynb`: CLIP similarity & (sample) FID evaluation of fine-tuned outputs vs base model.

## Outputs
Directory `task6_sd_lora/` contains saved adapted UNet, tokenizer, and text encoder components. Use with `generate.py --backend diff --lora_unet task6_sd_lora/unet_lora`.

## Steps Summary
1. Load base pipeline & scheduler.
2. Inject LoRA layers to UNet attention projections.
3. Encode images -> latents; add noise and predict noise residual.
4. Optimize LoRA params only.
5. Save adapted components.
6. Evaluate with CLIP similarity & (small) FID.

## Extension Ideas
- Increase dataset size & steps.
- Add DreamBooth subject token.
- Track loss & learning rate scheduling.

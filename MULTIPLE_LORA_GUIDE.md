# Multiple LoRA Generation Guide in AI Toolkit

This guide explains how to use multiple LoRA (Low-Rank Adaptation) models simultaneously when generating images with the AI Toolkit.

## How it Works

The AI Toolkit has been modified to support loading and fusing multiple LoRA models into the base diffusion model before image generation. This means you can combine the effects of several LoRAs in a single generation pass.

Each LoRA can have an individual `weight` associated with it, allowing you to control its influence on the final image.

## Configuration

To use multiple LoRAs, you need to modify your `generate` job configuration in the `model` section. Instead of `lora_path`, you will now use `lora_paths`, which is a list of dictionaries. Each dictionary should contain:

- `path`: The path to your LoRA `.safetensors` file (can be a local path or an S3 path).
- `weight`: (Optional) A float value representing the strength of this specific LoRA. If not specified, it defaults to `1.0`.

### Example for FLUX Models:

```yaml
job: generate
config:
  name: generate_multiple_loras_flux
  process:
    - type: to_folder
      output_folder: output/generated_images_multi_lora_flux
      device: cuda:0

      generate:
        sampler: flowmatch
        width: 1024
        height: 1280
        guidance_scale: 3
        sample_steps: 30
        seed: 99
        walk_seed: false
        num_repeats: 1
        ext: png

        prompts:
          - "A futuristic city at sunset, cyberpunk style, neon lights, flying cars"
          - "A serene forest with a hidden waterfall, magical, fantasy art"

      model:
        name_or_path: black-forest-labs/FLUX.1-dev
        is_flux: true
        quantize: true
        dtype: float16
        lora_paths: # Use lora_paths for multiple LoRAs
          - path: s3://sh9zztnhd4/lora-dashboard/results/Trening_1_1/lora/Trening 1.0.safetensors
            weight: 0.8
          - path: /path/to/your/second_lora.safetensors # Replace with your actual LoRA path
            weight: 0.5
          - path: /path/to/your/third_lora.safetensors # Replace with your actual LoRA path
            weight: 1.0
        # The old 'lora_path' and 'lora_weight' parameters are now deprecated if 'lora_paths' is used.
        # For backward compatibility, if only 'lora_path' is provided, it will be converted to the new 'lora_paths' format internally.
```

### Example for SDXL Models:

```yaml
job: generate
config:
  name: generate_multiple_loras_sdxl
  process:
    - type: to_folder
      output_folder: output/generated_images_multi_lora_sdxl
      device: cuda:0

      generate:
        sampler: ddim
        width: 1024
        height: 1024
        guidance_scale: 7
        sample_steps: 25
        seed: 42
        walk_seed: false
        num_repeats: 1
        ext: png

        prompts:
          - "A majestic dragon flying over a medieval castle, highly detailed, fantasy art"
          - "A futuristic robot bartender serving drinks in a neon-lit bar, cinematic, sci-fi"

      model:
        name_or_path: stabilityai/stable-diffusion-xl-base-1.0
        is_xl: true
        dtype: float16
        lora_paths: # Use lora_paths for multiple LoRAs
          - path: /path/to/your/sdxl_lora1.safetensors # Replace with your actual SDXL LoRA path
            weight: 0.7
          - path: /path/to/your/sdxl_lora2.safetensors # Replace with your actual SDXL LoRA path
            weight: 0.9
```

## Backward Compatibility

If you use the old `lora_path` parameter (without `lora_paths`), the system will automatically convert it into the new `lora_paths` format with a default weight of `1.0`. This ensures that your existing configurations will continue to work without modification.

## Important Notes

- **Fusing LoRAs**: The LoRAs are fused directly into the base model during the `load_model()` phase. This means that the `network_multiplier` in the `GenerateImageConfig` will act as a global multiplier for the combined effect of all loaded LoRAs.
- **Performance**: Fusing multiple LoRAs can increase model loading time, especially in `low_vram` mode for FLUX models, where LoRAs are fused in parts on the CPU. However, once loaded, generation performance should be similar to using a single fused LoRA.
- **LoRA Conflicts**: Be mindful of potential conflicts or unexpected results when combining multiple LoRAs, as their effects might interact in complex ways. Experiment with different weights to achieve desired outcomes.

## Recommended Weight Values

Based on testing, here are recommended weight ranges for different types of LoRAs:

- **Main subject/character LoRA**: 0.8 - 1.0
- **Artistic style LoRA**: 0.5 - 0.7
- **Quality enhancement LoRA**: 0.3 - 0.5
- **Subtle effect LoRA**: 0.2 - 0.4

## Technical Details

### Implementation

The multiple LoRA support is implemented at two levels:

1. **Configuration Level** (`toolkit/config_modules.py`):
   - Added `lora_paths` field to `ModelConfig`
   - Automatic conversion of old `lora_path` format
   - Validation of LoRA configuration structure

2. **Model Loading Level** (`toolkit/stable_diffusion_model.py`):
   - Modified `load_model()` to iterate through multiple LoRAs
   - Support for FLUX models (with low_vram mode)
   - Support for SDXL and SD1.5 models
   - Individual weight application for each LoRA

### FLUX Low VRAM Mode

For FLUX models in `low_vram` mode, LoRAs are fused in two parts to avoid out-of-memory errors:
1. Double transformer blocks
2. Single transformer blocks

Each part is loaded to GPU, fused, and then moved back to CPU before processing the next part.

## Troubleshooting

### Issue: Out of Memory Errors

**Solution**: Try reducing the weights or using fewer LoRAs. For FLUX models, ensure `low_vram` mode is enabled in your configuration.

### Issue: Unexpected Results

**Solution**: LoRAs may conflict with each other. Try:
- Adjusting individual weights
- Using fewer LoRAs
- Testing LoRAs individually first to understand their effects

### Issue: Loading Errors

**Solution**: Ensure all LoRA paths are correct and the `.safetensors` files exist. Check that S3 paths are accessible if using remote storage.

## Examples

### Example 1: Character + Style + Quality

```yaml
lora_paths:
  - path: /models/character_lora.safetensors
    weight: 1.0  # Main character, full strength
  - path: /models/anime_style_lora.safetensors
    weight: 0.6  # Style influence
  - path: /models/quality_boost_lora.safetensors
    weight: 0.4  # Subtle quality enhancement
```

### Example 2: Multiple Styles Blending

```yaml
lora_paths:
  - path: /models/watercolor_style.safetensors
    weight: 0.7
  - path: /models/impressionist_style.safetensors
    weight: 0.5
```

### Example 3: S3 + Local LoRAs

```yaml
lora_paths:
  - path: s3://bucket/trained_models/my_character.safetensors
    weight: 0.9
  - path: /local/path/style_lora.safetensors
    weight: 0.6
```

## Support

For issues or questions, please refer to the main AI Toolkit documentation or open an issue on GitHub.


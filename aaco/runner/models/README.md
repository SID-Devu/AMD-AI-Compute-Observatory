# AACO Model Configurations

This directory contains model configuration files for profiling with AACO.

## Structure

Each model has a JSON configuration file with:

- **Model metadata**: ID, name, category, framework
- **Architecture**: Layers, parameters, shapes
- **Default config**: Batch size, precision, etc.
- **Profiling**: Key kernels, warmup/profile iterations
- **Benchmarks**: Expected performance per GPU
- **Sources**: Where to load the model from

## Models

| Model | Category | Parameters | File |
|-------|----------|------------|------|
| ResNet-50 | Vision | 25M | `resnet50.json` |
| BERT Base | NLP | 110M | `bert_base.json` |
| LLaMA 2 7B | LLM | 7B | `llama2_7b.json` |

## Usage

```python
from aaco.runner import ModelRegistry

# Load model config
config = ModelRegistry.load("llama2-7b")

# Profile with defaults
from aaco.runner import LLMProfiler
profiler = LLMProfiler(config)
results = profiler.profile()
```

## Adding Models

1. Create a new JSON file in this directory
2. Follow the schema in existing files
3. Add benchmark expectations for target GPUs
4. Register with the model registry

## Schema

```json
{
  "model_id": "string (unique identifier)",
  "display_name": "string (human readable)",
  "category": "vision|nlp|llm|audio|multimodal",
  "framework": "pytorch|onnx|tensorflow",
  "architecture": { ... },
  "default_config": { ... },
  "profiling": { ... },
  "benchmarks": { ... },
  "sources": { ... }
}
```

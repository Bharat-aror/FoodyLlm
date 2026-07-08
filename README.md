# FoodyLlm
# FoodyLLM

A fine-tuned Llama 3 8B nutrition assistant that estimates ingredients and per-100g nutritional values from a natural language dish description.

## Overview

FoodyLLM fine-tunes Meta's Llama 3 8B Instruct model using QLoRA to build a domain-specific nutrition assistant. Given a dish name (e.g. "butter chicken"), the model returns a structured JSON response listing likely ingredients and estimated nutrition per 100g (calories, protein, carbs, fat).

## Approach

**Data preparation**
- Sourced from the Nutrition5k dataset (via Kaggle), combining dish ingredient lists with measured nutrition values
- Nutrition values normalised to a per-100g basis for consistency across dish sizes
- Dish names auto-generated from the top 3 ingredients by weight
- Training examples filtered to remove implausible values (e.g. extreme calorie or macro values, missing data)
- Formatted into Llama 3 instruction/response pairs using the official chat template

**Fine-tuning**
- Base model: Meta-Llama-3-8B-Instruct, loaded in 4-bit via Unsloth
- QLoRA / PEFT applied (rank 8, alpha 16) targeting attention and MLP projection layers
- Trained with the TRL `SFTTrainer`, using gradient checkpointing and 8-bit AdamW for memory efficiency
- Single epoch over the training set, with mixed precision (bf16/fp16 depending on GPU support)

**Evaluation**
- Tested against a fixed set of dishes (e.g. mac and cheese, spaghetti bolognese) to check ingredient and nutrition output quality
- Before/after comparison run to assess the effect of fine-tuning on response structure and accuracy
- Output validated as parseable JSON, with fallback error handling for malformed generations

**Deployment**
- Fine-tuned model and tokenizer pushed to the Hugging Face Hub
- Served via a FastAPI backend for integration into the FoodyLLM mobile app, with the USDA FoodData Central API used as the primary nutrition source and this model as fallback for Indian, Asian, and homemade dishes not well covered by USDA data

## Tech Stack

- Python, PyTorch
- Unsloth (efficient 4-bit fine-tuning)
- Hugging Face Transformers, PEFT, TRL (SFTTrainer)
- Meta-Llama-3-8B-Instruct
- FastAPI (serving)
- Nutrition5k dataset (Kaggle)

## Dataset

[Nutrition5k dataset](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset), combining dish-level ingredient and nutrition records.

## Notes

- Requires a Hugging Face access token to download the base model and push the fine-tuned version — set as an environment variable, never hardcoded
- Requires a Kaggle API key to download the dataset — same rule applies
- Designed to run on a GPU-backed environment (developed on Google Colab)

## Model

Fine-tuned model available at [`Bharat773/FoodyLLM-global`](https://huggingface.co/Bharat773/FoodyLLM-global) on Hugging Face.

## Author

Bharat Arora
[LinkedIn](http://www.linkedin.com/in/bharat-arora-2a13772b9)

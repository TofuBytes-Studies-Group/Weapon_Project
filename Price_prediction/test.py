from transformers import AutoConfig

gemma_model_name = "DavidAU/Gemma-The-Writer-Mighty-Sword-9B-GGUF"
config = AutoConfig.from_pretrained(gemma_model_name)
print(config)

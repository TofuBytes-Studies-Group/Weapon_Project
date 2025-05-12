from transformers import AutoModelForCausalLM, AutoTokenizer

# Try loading a known CausalLM model (GPT-2 as an example)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test text generation
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "deepseek-ai/deepseek-llm-7b-chat"
local_dir = "./models/deepseek-7b-chat"  # or any path you prefer

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_dir)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(local_dir)

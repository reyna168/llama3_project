# from transformers import LlamaForCausalLM, LlamaTokenizer

# # Load tokenizer and model
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# # Tokenize input
# input_text = "Hello, how are you?"
# inputs = tokenizer(input_text, return_tensors="pt")

# # Generate response
# outputs = model.generate(inputs["input_ids"])
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(response)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8, lora_alpha=32, lora_dropout=0.1
)

peft_model = get_peft_model(model, peft_config)
peft_model.train()
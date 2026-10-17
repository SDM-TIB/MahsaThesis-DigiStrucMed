"""Reusable fixed-adapter inference used by prompt optimization."""


def load_fixed_model(base_model, adapter, dtype='float32', device='cpu'):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    device_map = {'': device}
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=getattr(torch, dtype),
                                               device_map=device_map, low_cpu_mem_usage=True)
    # PEFT otherwise defaults to "auto" when redispatching a CPU-loaded model,
    # which can unexpectedly introduce disk offloading during adapter loading.
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False,
                                     device_map=device_map)
    model.eval()
    return tokenizer, model


def generate_fixed(tokenizer, model, messages, max_new_tokens=1024):
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        # Greedy decoding on this 1B model loops into repeated phrases/duplicated JSON
        # objects without a repetition guard; no_repeat_ngram_size hard-blocks exact
        # repeats and repetition_penalty discourages near-repeats, while do_sample stays
        # False so output remains deterministic.
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                repetition_penalty=1.3, no_repeat_ngram_size=3,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()

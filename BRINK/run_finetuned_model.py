import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Choose one adapter:
ADAPTER_PATH = "model/qKG/finetuned_LLaMA_3.2_1B_with_rules_CoT2"
# ADAPTER_PATH = "model/qKG/finetuned_LLaMA_3.2_1B_baseline"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    # Explicit CPU loading prevents Accelerate from trying to disk-offload
    # the whole base model before PEFT attaches the LoRA adapter.
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)
model.eval()

SYSTEM_MESSAGE = "Answer the question based on the provided knowledge."
MAX_TURNS = 10


def new_conversation():
    return [{"role": "system", "content": SYSTEM_MESSAGE}]


messages = new_conversation()

print("\nModel loaded. Start chatting!")
print("Commands: /clear resets the conversation, /exit quits.\n")

while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not question:
        continue
    if question.lower() in {"/exit", "/quit"}:
        print("Goodbye!")
        break
    if question.lower() == "/clear":
        messages = new_conversation()
        print("Conversation cleared.\n")
        continue

    messages.append({"role": "user", "content": question})

    # Keep the system message and only the most recent conversation turns.
    if len(messages) > 1 + (MAX_TURNS * 2):
        messages = [messages[0]] + messages[-(MAX_TURNS * 2):]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    print("Assistant: ", end="", flush=True)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(answer, "\n")

    messages.append({"role": "assistant", "content": answer})

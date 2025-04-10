from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def load_conversation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def count_tokens(text):
    return len(tokenizer.encode(text))

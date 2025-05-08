import tiktoken

def load_conversation(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def truncate_text(text, token_limit, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:token_limit])

def truncate_text_safe(text, token_limit, model_name="gpt-3.5-turbo"):
    from nltk.tokenize import sent_tokenize
    encoding = tiktoken.encoding_for_model(model_name)
    sentences = sent_tokenize(text)

    out = []
    total = 0
    for sent in sentences:
        sent_tokens = encoding.encode(sent)
        if total + len(sent_tokens) > token_limit:
            break
        out.append(sent)
        total += len(sent_tokens)
    return " ".join(out)

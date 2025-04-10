from transformers import pipeline
from utils.text_helpers import split_sentences

# Load the T5 model for question generation
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-e2e-qg", framework="pt")

def generate_questions(text, max_questions=5):
    """
    Generate question-answer style outputs from input conversation using T5.
    """
    # Split and join to clean up formatting
    sentences = split_sentences(text)
    joined_text = " ".join(sentences)

    # Prompt the model to generate questions
    prompt = f"generate questions: {joined_text}"

    outputs = qg_pipeline(prompt, max_length=128, do_sample=True, num_return_sequences=max_questions)

    # Format results
    formatted = []
    for i, out in enumerate(outputs):
        qa = out['generated_text'].strip()
        formatted.append(f"Q{i+1}: {qa}")

    return "\n\n".join(formatted)
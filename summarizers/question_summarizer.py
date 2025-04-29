from transformers import pipeline
from utils.text_helpers import split_sentences
from utils.io import truncate_text

# Load question generation model (T5)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=-1)

# Load question answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", framework="pt")

def generate_questions(text, max_questions=5, token_limit=3000):
    
    #Generate question-answer pairs from the input conversation.
    
    # Truncate if needed
    context = truncate_text(text, token_limit)

    # Clean and prep input for question generation
    sentences = split_sentences(context)
    joined_text = " ".join(sentences)
    prompt = f"generate questions: {joined_text}"

    # Generate questions
    outputs = qg_pipeline(prompt, max_length=128, do_sample=True, num_return_sequences=max_questions)
    questions = [out['generated_text'].strip() for out in outputs]

    # Generate answers using QA model
    qa_pairs = []
    for i, question in enumerate(questions):
        try:
            result = qa_pipeline(question=question, context=context)
            answer = result['answer'].strip()
        except:
            answer = "[Answer not found]"
        qa_pairs.append(f"Q{i+1}: {question}\nA{i+1}: {answer}")

    return "\n\n".join(qa_pairs)

import streamlit as st
from openai import OpenAI, RateLimitError, APIError
import json
import time

# --- Configuration ---
SUMMARIZER_MODEL = "gpt-4o"
EVALUATOR_MODEL = "gpt-4o-mini"

# --- Summarization Strategies (Prompts) ---
# These define how we ask the summarizer model to perform its task
SUMMARIZATION_STRATEGIES = {
    "Standard Concise": {
        "system_prompt": "You are an expert summarizer. Condense the following text, focusing on the main points and key information, while maintaining clarity and coherence.",
        "user_prompt_template": "Please summarize the following text concisely:\n\n---\n{text}\n---"
    },
    "Key Bullet Points": {
        "system_prompt": "You are an expert summarizer. Extract the key bullet points or main takeaways from the following text.",
        "user_prompt_template": "Extract the key bullet points from this text:\n\n---\n{text}\n---"
    },
    "Abstractive TL;DR": {
        "system_prompt": "You are an expert at creating very short, high-level summaries (like a TL;DR). Capture the absolute essence of the text in 1-2 sentences.",
        "user_prompt_template": "Provide a 1-2 sentence TL;DR summary for this text:\n\n---\n{text}\n---"
    },
    "Question-Focused (Who, What, Why)": {
        "system_prompt": "You are an expert summarizer. Summarize the text by primarily answering the questions: Who was involved? What happened? Why did it happen?",
        "user_prompt_template": "Summarize this text, focusing on Who, What, and Why:\n\n---\n{text}\n---"
    }
}

# --- Evaluation Criteria ---
# This defines how we ask the evaluator model to assess the summary
EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the quality of a text summary.
Compare the 'Generated Summary' against the 'Original Text'.

Please evaluate the summary based on the following criteria, providing a score from 1 (Poor) to 5 (Excellent) for each, along with a brief justification:

1.  **Relevance:** Does the summary accurately capture the main topic and key points of the original text? (Score 1-5)
2.  **Coherence:** Is the summary well-organized, easy to understand, and does it flow logically? (Score 1-5)
3.  **Conciseness:** Does the summary effectively condense the original text without unnecessary jargon or repetition? (Is it significantly shorter?) (Score 1-5)
4.  **Information Preservation:** Does the summary retain the most critical information from the original text without introducing inaccuracies or misinterpretations? (Score 1-5)

Provide your evaluation in JSON format like this example:

```json
{{
  "evaluation": {{
    "relevance": {{"score": 4, "justification": "Captures most key points but missed one minor detail."}},
    "coherence": {{"score": 5, "justification": "The summary is clear and flows well."}},
    "conciseness": {{"score": 5, "justification": "Significantly shorter while retaining core message."}},
    "information_preservation": {{"score": 3, "justification": "Omitted a key statistic mentioned in the original."}}
  }},
  "overall_comment": "A good summary, but could improve on preserving specific details."
}}
```

**Original Text:**
---
{original_text}
---

**Generated Summary:**
---
{summary_text}
---

Your JSON Evaluation:
"""

# --- Helper Functions ---

def get_openai_client(api_key):
    "Initializes and returns the OpenAI client."
    if not api_key:
        st.error("API Key is missing!")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

@st.cache_data(show_spinner=False)
def generate_summary(_client, original_text, strategy_name, max_tokens=500):
    "Calls OpenAI API to generate a summary using a specific strategy."
    if not _client or not original_text or strategy_name not in SUMMARIZATION_STRATEGIES:
        return "Error: Invalid input for summarization."

    strategy = SUMMARIZATION_STRATEGIES[strategy_name]
    system_prompt = strategy["system_prompt"]
    user_prompt = strategy["user_prompt_template"].format(text=original_text)

    try:
        response = _client.chat.completions.create(
            model=SUMMARIZER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5 # Slightly creative but still focused
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except RateLimitError:
        return "Error: OpenAI API rate limit exceeded. Please try again later."
    except APIError as e:
        return f"Error: OpenAI API error: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred during summarization: {e}"

@st.cache_data(show_spinner=False)
def evaluate_summary(_client, original_text, summary_text):
    "Calls OpenAI API to evaluate the summary against the original text."
    if not _client or not original_text or not summary_text:
        return {"error": "Invalid input for evaluation."}

    eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        original_text=original_text,
        summary_text=summary_text
    )

    try:
        response = _client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1, # Low temp for consistent evaluation
            response_format={"type": "json_object"} # Request JSON output
        )
        evaluation_raw = response.choices[0].message.content
        
        # Try parsing the JSON response
        try:
            evaluation_data = json.loads(evaluation_raw)
            return evaluation_data
        except json.JSONDecodeError:
            st.warning("Evaluation model didn't return valid JSON. Displaying raw output.")
            return {"raw_evaluation": evaluation_raw}
            
    except RateLimitError:
        return {"error": "OpenAI API rate limit exceeded during evaluation."}
    except APIError as e:
        return {"error": f"OpenAI API error during evaluation: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during evaluation: {e}"}

# --- Streamlit App UI ---

def main():
    st.set_page_config(layout="wide", page_title="Summary Strategy Evaluator")
    st.title("📝 Summary Strategy Evaluator")
    st.markdown(f"Generate summaries using **{SUMMARIZER_MODEL}** and evaluate them with **{EVALUATOR_MODEL}**.")

    # --- Session State Initialization ---
    if 'original_text' not in st.session_state: st.session_state.original_text = ""
    if 'selected_strategy' not in st.session_state: st.session_state.selected_strategy = list(SUMMARIZATION_STRATEGIES.keys())[0]
    if 'generated_summary' not in st.session_state: st.session_state.generated_summary = None
    if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None

    # --- Sidebar for API Key ---
    st.sidebar.header("🔑 Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Required to use the summarization and evaluation models.")
    client = get_openai_client(api_key)

    if not client:
        st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
        st.stop()

    # --- Main Layout --- 
    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Input & Strategy")
        st.session_state.original_text = st.text_area(
            "Enter Text to Summarize:", 
            height=300, 
            value=st.session_state.original_text,
            key="text_input"
        )
        
        st.session_state.selected_strategy = st.selectbox(
            "Choose Summarization Strategy:", 
            options=list(SUMMARIZATION_STRATEGIES.keys()),
            index=list(SUMMARIZATION_STRATEGIES.keys()).index(st.session_state.selected_strategy), # Maintain selection
            key="strategy_select",
            help="Each strategy uses a different prompt to guide the summarizer."
        )
        
        # Update strategy description dynamically
        strategy_desc = SUMMARIZATION_STRATEGIES[st.session_state.selected_strategy]["system_prompt"]
        st.caption(f"*Strategy Goal:* {strategy_desc}")
        
        submit_summary = st.button("✨ Generate Summary", disabled=(not st.session_state.original_text))
        
        if submit_summary:
            st.session_state.generated_summary = None # Clear previous summary
            st.session_state.evaluation_results = None # Clear previous evaluation
            with st.spinner(f"Generating summary using '{st.session_state.selected_strategy}' strategy..."):
                summary = generate_summary(client, st.session_state.original_text, st.session_state.selected_strategy)
                st.session_state.generated_summary = summary
            if "Error:" in (st.session_state.generated_summary or ""):
                 st.error(st.session_state.generated_summary)
            else:
                 st.success("Summary generated!")

    with col2:
        st.header("2. Summary & Evaluation")
        if st.session_state.generated_summary and "Error:" not in st.session_state.generated_summary:
            st.subheader("Generated Summary")
            st.text_area("Summary Output", value=st.session_state.generated_summary, height=200, key="summary_output", disabled=True)
            
            submit_evaluate = st.button("📊 Evaluate Summary")
            if submit_evaluate:
                st.session_state.evaluation_results = None # Clear previous evaluation
                with st.spinner(f"Evaluating summary using {EVALUATOR_MODEL}..."):
                    evaluation = evaluate_summary(client, st.session_state.original_text, st.session_state.generated_summary)
                    st.session_state.evaluation_results = evaluation
                if "error" in (st.session_state.evaluation_results or {}):
                    st.error(st.session_state.evaluation_results["error"])
                else:
                    st.success("Evaluation complete!")

        elif st.session_state.generated_summary and "Error:" in st.session_state.generated_summary:
            st.error(f"Could not generate summary: {st.session_state.generated_summary}")
        else:
            st.info("Generate a summary using the button on the left to enable evaluation.")
        
        # Display Evaluation Results
        if st.session_state.evaluation_results and "error" not in st.session_state.evaluation_results:
            st.subheader("Evaluation Results")
            if "evaluation" in st.session_state.evaluation_results: 
                eval_data = st.session_state.evaluation_results['evaluation']
                cols = st.columns(len(eval_data)) # Create columns for each metric
                for idx, (metric, details) in enumerate(eval_data.items()):
                    with cols[idx]:
                        st.metric(label=metric.capitalize(), value=f"{details['score']}/5")
                        st.caption(details['justification'])
                st.markdown("**Overall Comment:**")
                st.write(st.session_state.evaluation_results.get('overall_comment', "N/A"))
            elif "raw_evaluation" in st.session_state.evaluation_results:
                # Display if JSON parsing failed but we got text back
                st.code(st.session_state.evaluation_results["raw_evaluation"], language=None)
            else:
                st.warning("Evaluation results seem incomplete or malformed.")
        elif st.session_state.evaluation_results and "error" in st.session_state.evaluation_results:
             # Error already displayed above, could add more detail here if needed
             pass 

if __name__ == "__main__":
    main() 
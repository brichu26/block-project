import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from context_window_manager import ContextWindowManager
import pandas as pd
from datetime import datetime
import time

def create_token_usage_plot(context_manager):
    """Create a plot showing token usage over time"""
    history = context_manager.get_context_history()
    if not history:
        return None
    
    # Prepare data
    timestamps = [entry['timestamp'] for entry in history]
    token_counts = [entry['token_count'] for entry in history]
    cumulative_tokens = []
    current_cumulative = 0
    for entry in history:
        if entry.get('is_summary', False):
            current_cumulative = entry['token_count']
        else:
            current_cumulative += entry['token_count']
        cumulative_tokens.append(current_cumulative)
        
    is_summary = [entry.get('is_summary', False) for entry in history]
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add token count bars
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=token_counts,
            name="Token Count per Message/Summary",
            marker_color=['red' if s else 'blue' for s in is_summary]
        ),
        secondary_y=False
    )
    
    # Add cumulative line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_tokens,
            name="Effective Token Count",
            line=dict(color='green')
        ),
        secondary_y=True
    )
    
    # Add context limit line
    fig.add_hline(
        y=context_manager.max_context_length,
        line_dash="dash",
        line_color="orange",
        name="Context Limit",
        annotation_text="Context Limit",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        title="Context Window Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Tokens Added (Blue=Msg, Red=Summary)",
        yaxis2_title="Effective Context Tokens",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_context_history_table(context_manager):
    """Create a table showing the context history"""
    history = context_manager.get_context_history()
    if not history:
        return None
    
    # Prepare data
    data = []
    for entry in history:
        content_preview = entry['text']
        if entry.get('is_summary', False):
            content_preview = content_preview.replace("[SUMMARY] ", "", 1)
        content_preview = content_preview[:150] + '...' if len(content_preview) > 150 else content_preview

        data.append({
            'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Type': 'Summary' if entry.get('is_summary', False) else 'Message',
            'Token Count': entry['token_count'],
            'Content Preview': content_preview
        })
    
    # Display newest first
    df = pd.DataFrame(data)
    return df.iloc[::-1]

def main():
    st.set_page_config(layout="wide")
    st.title("GPT-4 Context Window Management Visualizer")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")

    # API Key Input (Moved to top and made mandatory)
    api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use GPT models.",
        value=st.session_state.get("api_key", "")
    )

    if not api_key:
        st.error("🚫 Please enter your OpenAI API key in the sidebar to initialize the application.")
        st.stop()

    # Store API key safely in session state if entered
    st.session_state.api_key = api_key

    # Model Selection
    model_name = st.sidebar.selectbox(
        "🤖 Model",
        ("gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"),
        help="Select the OpenAI model for summarization."
    )

    # Context Length Slider (Dynamically adjust based on model if needed)
    max_context_limit = 128000
    default_context = 8000 if "gpt-4" in model_name and "turbo" not in model_name and "o" not in model_name else 16000

    max_context_length = st.sidebar.slider(
        "📐 Max Context Length (tokens)",
        min_value=1000,
        max_value=max_context_limit,
        value=min(default_context, max_context_limit),
        step=1000,
        help="The maximum number of tokens allowed before summarization."
    )

    # Summary Length Slider
    summary_length = st.sidebar.slider(
        "📝 Summary Target Length (tokens)",
        min_value=100,
        max_value=4000,
        value=min(1000, max_context_length // 4),
        step=100,
        help="The target token length for the generated summary."
    )

    # Initialize or Reinitialize Context Manager if settings change
    manager_key = f"{api_key}-{model_name}-{max_context_length}-{summary_length}"

    if 'manager_key' not in st.session_state or st.session_state.manager_key != manager_key:
        try:
            with st.spinner("Initializing Context Manager with new settings..."):
                st.session_state.context_manager = ContextWindowManager(
                    api_key=st.session_state.api_key,
                    model_name=model_name,
                    max_context_length=max_context_length,
                    summary_length=summary_length
                )
            st.session_state.manager_key = manager_key
            st.sidebar.success("Context Manager Initialized!")
        except Exception as e:
            st.sidebar.error(f"Initialization Failed: {e}")
            st.stop()


    # --- Main Page Layout ---
    st.header("💬 Context Interaction")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add Single Message")
        new_message = st.text_area("Enter a message to add:", height=100, key="single_message")
        if st.button("Add Message to Context"):
            if new_message:
                with st.spinner("Adding message and checking context..."):
                    st.session_state.context_manager.add_to_context(new_message)
                st.success("Message added!")
                st.rerun()
            else:
                st.warning("Please enter a message.")

        st.divider()

        st.subheader("Process Pasted Text")
        pasted_text = st.text_area("Paste a large block of text:", height=200, key="pasted_text")
        if st.button("Process Pasted Text"):
            if pasted_text:
                with st.spinner("Processing text, may require summarization..."):
                    st.session_state.context_manager.add_to_context(pasted_text)
                st.success("Pasted text processed!")
                st.rerun()
            else:
                st.warning("Please paste some text.")


    with col2:
        st.subheader("Current Context State")
        current_context = st.session_state.context_manager.get_current_context()
        st.text_area("📜 Current Context (Summary or recent messages)", value=current_context, height=400, disabled=True, key="current_context_display")

        # Display metrics
        st.subheader("📊 Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                "Token Count",
                st.session_state.context_manager.get_token_count()
            )
        with m_col2:
            usage_ratio = st.session_state.context_manager.get_context_usage_ratio()
            st.metric(
                "Context Usage",
                f"{usage_ratio:.1%}"
            )
        with m_col3:
             st.metric(
                "History Entries",
                len(st.session_state.context_manager.get_context_history())
            )

        # Progress Bar for Context Usage
        st.progress(usage_ratio)


    # --- Visualization and History ---
    st.divider()
    st.header("📈 Visualization & History")

    # Visualization
    st.subheader("Context Usage Visualization")
    fig = create_token_usage_plot(st.session_state.context_manager)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add messages to see the context usage visualization.")

    # History table
    st.subheader("📋 Context History (Newest First)")
    df = create_context_history_table(st.session_state.context_manager)
    if df is not None:
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("History will appear here as messages are added.")

    # Clear context button (moved to sidebar for less clutter)
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Context & History"):
        st.session_state.context_manager.clear_context()
        if 'manager_key' in st.session_state:
             del st.session_state.manager_key
        st.success("Context cleared!")
        st.rerun()


if __name__ == "__main__":
    main() 
import streamlit as st
import time

# Title for the page
st.title('RAG for PDF Text Retrieval')

# Initialize or reset the session state keys
if 'question' not in st.session_state or 'reset' in st.session_state:
    st.session_state['question'] = ""
    st.session_state.pop('reset', None)  # Remove the reset flag if it exists

# Text input for the question, linked to session state for dynamic update
question = st.text_input("Enter your question here:", value=st.session_state['question'], key='question')


# Function to handle reset
def reset_form():
    st.session_state['reset'] = True


# Button to process the question
if st.button('Get Answer'):
    # Simulating a popup-like notification
    with st.spinner('Processing your request...'):
        # Simulate a background task with a sleep delay
        time.sleep(10)
    # You would replace this with the function call that processes the question
    # For example: answer = process_question(question)
    answer = "This is a placeholder answer for the question: " + question
    # Display the answer
    st.text_area("Answer", answer, height=300)

st.write("")
st.write("")

# Button to reset the inputs and outputs, calling the reset_form function on click
if st.button('Reset', on_click=reset_form):
    pass  # The actual reset logic is handled in the reset_form function

# Adjust button placement with custom CSS
st.markdown("""
<style>
.stButton>button {
  float: right;
}
</style>
""", unsafe_allow_html=True)

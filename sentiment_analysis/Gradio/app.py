import gradio as gr
from transformers import pipeline

pipe = pipeline(model="anjan-k/Sentiment-Analysis-FineTune-HuggingFace")

# Function to analyze sentiment
def analyze_text(inputs):
    result = pipe(inputs, top_k=1)
    label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    output = []
    label = result[0]["label"]
    sentiment = label_mapping[label]
    return sentiment

# Gradio interface
input_box = gr.Textbox(lines=3,
                     placeholder="Enter a text to analyze sentiment (e.g., 'I love this product.', 'I am not feeling well' )",
                     label="Input Text")

output_box = gr.Textbox(label="Sentiment")

iface = gr.Interface(
    fn=analyze_text, 
    inputs=input_box,
    outputs=output_box,
    title="Sentiment Analysis",
    flagging_mode="never",
)

# Launch the interface
iface.launch()

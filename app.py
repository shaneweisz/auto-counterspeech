from decode import DEFAULT_CONFIG, DEFAULT_MODEL, get_decoding_config
from response_generation import ResponseGenerator
import gradio as gr


decoding_config = get_decoding_config(DEFAULT_CONFIG, config_overrides="")
model = ResponseGenerator(DEFAULT_MODEL, decoding_config)


def respond(hate_speech_input_text):
    return model.respond(hate_speech_input_text)


demo = gr.Interface(fn=respond, inputs="text", outputs="text")

demo.launch()

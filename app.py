from flask import Flask
from flask import request
from response_generation import ResponseGenerator

app = Flask(__name__)

print("Loading counterspeech model...")

model = "models/dialoGPT-mtconan"
decoding_config = {
    "max_new_tokens": 100,
    "num_beams": 10,
    "no_repeat_ngram_size": 4,
    "early_stopping": True,
    "exponential_decay_length_penalty": (15, 0.9),
}
response_generator = ResponseGenerator(model, decoding_config)


@app.route("/")
def index():
    hate_speech_input_text = request.args.get("hate_speech_input_text", "")
    if hate_speech_input_text:
        # response = "Dummy response"
        response = response_generator.generate_response(hate_speech_input_text)
    else:
        response = ""

    output = ""
    if hate_speech_input_text:
        output = "Hate speech: " + hate_speech_input_text + "<br>" + "Response: " + response

    return (
        """<form id="frm" action="" method="get">
                Enter hate speech:
            </form>
            <textarea name="hate_speech_input_text" form="frm" rows="4" cols="50"></textarea>
            <br><br>
            <button type="submit" value="Submit" form="frm">Submit</button>
            <br><br>
            """
        + output
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

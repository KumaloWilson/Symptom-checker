from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load the pre-trained GPT-2 model and tokenizer
model_path = 'SmallMedLM.pt'
model = torch.load(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Set up the model to run on CPU
device = torch.device('cpu')
model.to(device)

# Define the endpoint for generating disease symptoms or disease
@app.route('/generate', methods=['POST'])
def generate_output():
    # Get the input text from the POST request
    input_str = request.json['input_text']

    # Encode the input text using the tokenizer
    input_ids = tokenizer.encode(input_str, return_tensors='pt').to(device)

    # Generate output using the model
    output = model.generate(
        input_ids,
        max_length=20,
        num_return_sequences=1,
        do_sample=True,
        top_k=8,
        top_p=0.95,
        temperature=0.5,
        repetition_penalty=1.2
    )

    # Decode the generated output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated output as a JSON response
    return jsonify({'generated_output': decoded_output})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

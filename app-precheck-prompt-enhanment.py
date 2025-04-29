from flask import Flask, request, jsonify
from openai import OpenAI
import os
import re
import logging
from dotenv import load_dotenv

# .env file load
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=api_key)

# Create Flask app
app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Allowed models
VALID_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]

def precheck_name(name):
    # 1) Invalid characters
    if re.search(r"[^A-Za-z\s\-.]", name):
        return False, "Invalid characters"
    # 2) At least 3 letters
    if len(re.findall(r"[A-Za-z]", name)) < 3:
        return False, "Too few letters"
    # 3) No consecutive hyphens/dots
    if '--' in name or '..' in name:
        return False, "Consecutive punctuation"
    # 4) Dot must have spaces around it
    if re.search(r"(?<=\w)\.(?=\w)", name):
        return False, "Invalid dot formatting"
    return True, None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    model = data.get('model', 'gpt-4o-mini')
    name = re.sub(r'\s+', ' ', name)

    # Camel-case conversion
    name = ' '.join(word[:1].upper() + word[1:].lower() for word in name.split())

    if not name:
        return jsonify({"error": "No name provided"}), 400
    if model not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from {VALID_MODELS}"}), 400

    # Pre-validation
    valid, reason = precheck_name(name)
    if not valid:
        logger.info(f"Pre-check failed for '{name}': {reason}")
        return jsonify({
            "name": name,
            "prediction": "Not Realistic"
            #,"reason": reason
        }), 200

    # Minimal prompt for 1/0
    prompt = f"Classify '{name}' as a realistic human name (single-word OK). Return only 1 or 0."

    try:
        logger.info(f"Calling OpenAI with model={model} for name='{name}'")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,
            temperature=0.0
        )

        reply = response.choices[0].message.content.strip()
        digit = reply[0] if reply else ''
        logger.info(f"Model returned: '{reply}'")

        if digit == '1':
            return jsonify({"name": name, "prediction": "Realistic"}), 200
        elif digit == '0':
            return jsonify({"name": name, "prediction": "Not Realistic"}), 200
        else:
            logger.error(f"Unexpected model output: {reply}")
            return jsonify({"error": "Invalid response from model"}), 500

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return jsonify({"error": "OpenAI API error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

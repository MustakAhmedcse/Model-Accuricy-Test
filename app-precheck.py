from flask import Flask, request, jsonify
from openai import OpenAI
import os
import re
import json
import logging
from dotenv import load_dotenv

# .env file load
load_dotenv()

# Initialize OpenAI client with your API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Make sure you have a .env file with this key.")
client = OpenAI(api_key=api_key)

# Create Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Valid OpenAI models
VALID_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]

# Pre-check validation function
def precheck_name(name):
    # Invalid characters
    if re.search(r"[^A-Za-z\s\-.]", name):
        return False, "Invalid characters present"
    # Letter count
    letters = re.findall(r"[A-Za-z]", name)
    if len(letters) < 3:
        return False, "Too few letters"
    # Consecutive hyphens or dots
    if '--' in name or '..' in name:
        return False, "Consecutive punctuation"
    # Dot must have spaces: no letter.dot.letter patterns
    if re.search(r"(?<=\w)\.(?=\w)", name):
        return False, "Invalid dot formatting, must have spaces after dot(.)"
    return True, None

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict if a name is realistic using OpenAI."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        name = data.get('name', '').strip()
        model = data.get('model', 'gpt-4.1-nano')  # Default to gpt-4.1-nano
        name = re.sub(r'\s+', ' ', name).strip()  # Normalize spaces
        # Convert each word's first character to uppercase, rest to lowercase
        name = ' '.join([word[:1].upper() + word[1:].lower() for word in name.split()])

        if not name:
            logger.warning("Received request with no name.")
            return jsonify({"error": "No name provided"}), 400

        # Pre-check before OpenAI call
        valid, reason = precheck_name(name)
        if not valid:
            logger.info(f"'{name}' rejected from pre-check with reason: {reason}")
            return jsonify({
                "name": name,
                "prediction": "Not Realistic",
                "reason": reason
            }), 200
        
        logger.info(f"'{name}' passed local checks. Proceeding to OpenAI validation with model '{model}'.")

        if model not in VALID_MODELS:
            logger.warning(f"Invalid model specified: {model}")
            return jsonify({"error": f"Invalid model. Choose from {VALID_MODELS}"}), 400

        # Build prompt
        # prompt = f"""
        # You are an expert in name classification. Determine if the full name '{name}' is a realistic human full name, used in any culture. Consider the name regardless of its capitalization.
        # Respond in JSON only:
        # - If realistic: {{"prediction":"Realistic"}}
        # - If not: {{"prediction":"Not Realistic","reason":"<≤50‑char reason>"}}
        # """

        # Build prompt
        prompt = f"""
        You are an expert in name classification. Determine if the name '{name}' is realistic human name. Single-word names are allowed. Ignore the case.
        Examples of realistic names: 'Mst Nodi', 'Md Hafijul', 'Mst Taslima', 'Mr. Hanif Uddin', 'Beauty','Md Jewel', 'Mst Sonia'
               
        Respond only with JSON:
        - Realistic: {{"prediction":"Realistic"}}
        - Not Realistic: {{"prediction":"Not Realistic","reason":"<≤50‑char reason>"}}
        """

        try:
            logger.info(f"Calling OpenAI API for name: '{name}' with model: '{model}'")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise name classification assistant outputting only JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.0
            )

            reply_content = response.choices[0].message.content.strip()
            logger.info(f"Received raw response: {reply_content}")
            # Extract JSON
            match = re.search(r"```json\s*({.*?})\s*```", reply_content, re.DOTALL|re.IGNORECASE)
            json_str = match.group(1) if match else reply_content

            try:
                result = json.loads(json_str)
                pred = result.get("prediction")

                if pred not in ["Realistic", "Not Realistic"]:
                    raise ValueError(f"Invalid prediction value from model: {pred}")

                if pred == "Realistic":
                    return jsonify({"name": name, "prediction": "Realistic"}), 200
                else:
                    return jsonify({
                        "name": name,
                        "prediction": "Not Realistic",
                        "reason": result.get("reason", "Reason not provided by AI.")
                    }), 200

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: '{json_str}'")
                return jsonify({"error": "Invalid JSON response format from model"}), 500
            except ValueError as ve:
                logger.error(f"Prediction validation error: {str(ve)}")
                return jsonify({"error": str(ve)}), 500

        except Exception as api_error:
            logger.error(f"OpenAI API call failed: {str(api_error)}")
            return jsonify({"error": f"OpenAI API error: {str(api_error)}"}), 500

    except Exception as e:
        logger.exception(f"An unexpected server error occurred: {str(e)}")
        return jsonify({"error": "An unexpected server error occurred"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug=False for production

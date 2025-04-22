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

def extract_json_from_response(text):
    """Extracts JSON content, potentially removing markdown fences."""
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    return text

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

        if not name:
            logger.warning("Received request with no name.")
            return jsonify({"error": "No name provided"}), 400

        if model not in VALID_MODELS:
            logger.warning(f"Invalid model specified: {model}")
            return jsonify({"error": f"Invalid model. Choose from {VALID_MODELS}"}), 400

        # --- If Local Validation Passed, Proceed to OpenAI ---
        logger.info(f"'{name}' passed local checks. Proceeding to OpenAI validation with model '{model}'.")

        # prompt = f"""
        # You are an expert in name classification. Determine if the full name '{name}' is a realistic human full name, used in any culture.
        # Consider the name regardless of its capitalization. 

        # A name is considered unrealistic if:
        # * It contains characters other than letters (a-z, A-Z), spaces, hyphens (-), and dots (.).
        # * It is less than three letter in total (excluding spaces, hyphens, and dots).
        # * It contains consecutive hyphens (e.g., '--') or consecutive dots (e.g., '..').
        # * It has a dot ('.') without spaces around it, resembling an email address or username (e.g., 'm.ahmed', 'Ravi.kumar').

        # Examples of realistic names: 'Mohiuddin Mohi', 'Aisha Khan', 'Sheik Kaykaus', 'Mr. Hanif Uddin', 'John-Doe', 'Mary. Anne Smith', 'm. a. h. hashan', 'p. k. robi mullah'.
        # Examples of unrealistic names: 'Abdullah123'(contains numbers), 'Table Chair'(common phrase), 'Qwert' (Keyboard Pattern).

        # Respond in JSON format. If the full name is 'Realistic', the JSON should only contain:
        # {{
        #     "prediction": "Realistic"
        # }}
        # If the name is 'Not Realistic', the JSON should contain:
        # {{
        #     "prediction": "Not Realistic",
        #     "reason": "<brief reason (max 50 character) why the name is not realistic>"
        # }}
        # """
        prompt = f"""
        You are an expert in name classification. Determine if the full name '{name}' is a realistic human full name, used in any culture. Consider the name regardless of its capitalization.

        A full name is considered unrealistic if:
        * It contains characters other than letters (a-z, A-Z), spaces, hyphens (-), and dots (.).
        * It has fewer than three letters in total (excluding spaces, hyphens, and dots). For example, 'Ku' has two letters and is unrealistic.
        * It contains consecutive hyphens (e.g., '--') or consecutive dots (e.g., '..').
        * It has a dot ('.') without spaces around it, resembling an email address or username (e.g., 'm.ahmed', 'Ravi.kumar').
        * It is a single word without a clear first and last name, unless it is a widely recognized full name in a specific culture. For example, 'Baba' is a single word and unrealistic, while 'Leonardo da Vinci' is realistic.

        Examples of realistic full names: 'Mohiuddin Mohi', 'Aisha Khan', 'Sheik Kaykaus', 'Mr. Hanif Uddin', 'John-Doe', 'Mary. Anne Smith', 'm. a. h. hashan', 'p. k. robi mullah', 'Leonardo da Vinci'.
        Examples of unrealistic names: 'Ku' (too short), 'Ghi' (single word), 'Baba' (single word), 'Vut' (single word), 'San' (single word), 'TUTU' (single word), 'Ami' (single word), 'Ash' (single word), 'Nono' (single word), 'Pon' (single word), 'Nav' (single word), 'Sisi' (single word), 'Wert' (keyboard pattern), 'Ji' (too short), 'Zaza' (single word), 'Li' (too short), 'Jahanara--Begum' (consecutive hyphens), 'Fu' (too short), 'PAPA' (single word), 'Dede' (single word), 'Abdullah123' (contains numbers), 'Table Chair' (common phrase), 'Qwert' (keyboard pattern), 'Abdullah' (single word).

        Respond in JSON format. If the full name is 'Realistic', the JSON should only contain:
        {{
            "prediction": "Realistic"
        }}
        If the name is 'Not Realistic', the JSON should contain:
        {{
            "prediction": "Not Realistic",
            "reason": "<brief reason (max 50 character) why the name is not realistic>"
        }}
        """
        # prompt = f"""
        # Classify the full name "{name}" (any capitalization) as realistic or not.  
        # Respond in JSON format. If the full name is 'Realistic', the JSON should only contain:
        # {{
        #     "prediction": "Realistic"
        # }}
        # If the name is 'Not Realistic', the JSON should contain:
        # {{
        #     "prediction": "Not Realistic",
        #     "reason": "<≤50‑char reason"
        # }}

        # Unrealistic if any of:
        # • Non‑letter (A–Z) chars other than space, hyphen, dot  
        # • Fewer than 3 letters total (e.g., 'Ku')  
        # • “--” or “..” present (e.g., 'Ravi--Kumar')  
        # • A dot touching letters without spaces (e.g., 'm.ahmed')  
        # • Single word (unless a culturally recognized full name, e.g., 'Madonna')

        # Note: Titles like 'Mr. Smith' are realistic if the dot is followed by a space.
        # """

        
        try:
            logger.info(f"Calling OpenAI API for name: '{name}' with model: '{model}'")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise name classification assistant outputting only JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.5
            )

            reply_content = response.choices[0].message.content.strip()
            logger.info(f"Received raw response: {reply_content}")
            json_str_to_parse = extract_json_from_response(reply_content)

            try:
                result = json.loads(json_str_to_parse)
                prediction_value = result.get("prediction")

                if prediction_value not in ["Realistic", "Not Realistic"]:
                    raise ValueError(f"Invalid prediction value from model: {prediction_value}")

                if prediction_value == "Realistic":
                    return jsonify({
                        "name": name,
                        "prediction": "Realistic"
                    }), 200
                elif prediction_value == "Not Realistic":
                    return jsonify({
                        "name": name,
                        "prediction": "Not Realistic",
                        "reason": result.get("reason", "Reason not provided by AI.")
                    }), 200

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: '{json_str_to_parse}'")
                return jsonify({"error": "Invalid JSON response format from model"}), 500
            except ValueError as ve:
                logger.error(f"Prediction validation error: {str(ve)}")
                return jsonify({"error": str(ve)}), 500

        except Exception as api_error:
            logger.error(f"OpenAI API call failed: {str(api_error)}")
            return jsonify({"error": f"OpenAI API error: {str(api_error)}"}), 500

    except Exception as e:
        logger.exception(f"An unexpected server error occurred: {str(e)}")
        return jsonify({"error": f"An unexpected server error occurred"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug=False for production
from flask import Flask, request, jsonify
from inference import generate_response
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/generate_response', methods=['POST'])
def generate_response_api():
    """
    API to handle LLM requests.
    Expects JSON payload with keys: facial_emotion, speech_emotion, text, speaking.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload."}), 400

    try:
        facial_emotion = data.get("facial_emotion", {"emotion": "unknown", "confidence": 0.0})
        speech_emotion = data.get("speech_emotion", {"emotion": "unknown", "confidence": 0.0})
        text = data.get("text", "")
        speaking = data.get("speaking", False)

        # Call the LLM function
        response = generate_response(
            facial_emotion=facial_emotion,
            speech_emotion=speech_emotion,
            text=text,
            speaking=speaking
        )

        logging.info("Generated response from LLM.")
        return jsonify({"response": response}), 200

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)

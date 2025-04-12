# exposing localhost port using ngrok http <port> 

# 1. Vapi Sends POST Request: When a user interacts with your Vapi application, Vapi sends a POST request containing conversation context and metadata to the configured endpoint (your ngrok URL).
#
# 2. Local Server Processes Request: Your Python script receives the POST request and the chat_completions function is invoked.
#
# 3. Extract and Prepare Data: The script parses the JSON data, extracts relevant information (prompt, conversation history), and builds the prompt for the OpenAI API call.
#
# 4. Call to OpenAI API: The constructed prompt is sent to the gpt-3.5-turbo-instruct model using the openai.ChatCompletion.create method.
#
# 5. Receive and Format Response: The response from OpenAI, containing the generated text, is received and formatted according to Vapi’s expected structure.
#
# 6. Send Response to Vapi: The formatted response is sent back to Vapi as a JSON object.
#
# 7. Vapi Displays Response: Vapi receives the response and displays the generated text within the conversation interface to the user.

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, port=5050)  

from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from chatbot import * 
from ingestion import * 
from pinecone_utils import *  
from flask_cors import CORS  
import logging

# Set up logging globally
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  # Create logger

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

@app.route("/upload", methods=["POST"])
def upload_file():
    user = request.form.get("user")
    collection = request.form.get("collection")
    if not user:
        return jsonify({"error": "User identifier is required"}), 400
    if not collection:
        return jsonify({"error": "Collection identifier is required"}), 400
    
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    # Ensure user has a Pinecone index
    create_pinecone_index(user)
    
    success, message = ingest_file(file_path, user, collection)
    if not success:
        return jsonify({"error": message}), 500
    
    return jsonify({"message": "File processed and stored successfully"})

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user = data.get("user")
    collection = data.get("collection")
    query_text = data.get("query", "")
    
    if not user:
        return jsonify({"error": "User identifier is required"}), 400
    if not collection:
        return jsonify({"error": "Collection identifier is required"}), 400
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    try:
        logger.debug(f"Querying Pinecone with user: {user}, collection: {collection}, query: {query_text}")
        response = query_pinecone(query_text, f'{user}-index', collection)
        logger.debug(f"Query response: {response}")
        return jsonify(response)  # Return detailed response
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/index-info", methods=["GET"])
def get_index_info():
    """Retrieve metadata about the user's Pinecone index (namespaces, files, etc.)."""
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "User identifier is required"}), 400

    try:
        index_info = get_pinecone_index_info(user)
        return jsonify(index_info)
    except Exception as e:
        print(f"Error fetching index info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/delete", methods=["POST"])
def delete_file():
    """Delete vectors associated with a specific file in a namespace for a user."""
    data = request.get_json()
    user = data.get("user")
    namespace = data.get("collection")
    filename = data.get("filename")
    
    if not user:
        return jsonify({"error": "User identifier is required"}), 400
    if not namespace:
        return jsonify({"error": "Namespace identifier is required"}), 400
    # if not filename:
        # return jsonify({"error": "Filename is required"}), 400
    
    try:
        success, message = delete_file_vectors(user, namespace, filename)
        if not success:
            return jsonify({"error": message}), 404
        return jsonify({"message": message})
    except Exception as e:
        print(f"Error deleting file: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)   
import os
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import io
import uuid

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def create_pinecone_index(user):
    """Create a new Pinecone index for the user if it doesn't exist."""
    index_name = f"{user}-index"
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="euclidean",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """Extract images from a PDF and save them as separate files."""
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"{os.path.basename(pdf_path)}_image_{i}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def load_document(file_path):
    """Load a document from file."""
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            text_data = loader.load()
            image_paths = extract_images_from_pdf(file_path)
            return text_data, image_paths
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith((".mp3", ".wav")):
            return transcribe_audio(file_path), []
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            return [Document(page_content=extract_text_from_image(file_path))], []
        else:
            raise ValueError("Unsupported file format")
        return loader.load(), []
    except Exception as e:
        print(f"Error loading document: {e}")
        return [], []

def transcribe_audio(file_path):
    """Transcribe audio file into text using SpeechRecognition."""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return [Document(page_content=text)]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return []

def split_text(document, chunk_size=1000, chunk_overlap=100):
    """Split document into smaller chunks."""
    try:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(document)
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []

def create_embeddings():
    """Create an embedding model."""
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-gecko-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

def store_embeddings(texts, embeddings, user, collection, file_path):
    """Store embeddings in Pinecone with filename and unique ID in metadata."""
    index_name = f"{user}-index"
    try:
        # Add filename and unique ID to metadata for each document
        for doc in texts:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['filename'] = os.path.basename(file_path)
            doc.metadata['vector_id'] = str(uuid.uuid4())  # Unique ID for each vector
            
        vector_store = PineconeVectorStore.from_documents(
            texts, 
            embeddings, 
            index_name=index_name, 
            namespace=collection
        )
        print(f"Stored {len(texts)} embeddings in Pinecone index '{index_name}' under namespace '{collection}'")
        return [doc.metadata['vector_id'] for doc in texts]  # Return list of vector IDs
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return []
    
def ingest_file(file_path, user, collection):
    """Process and store a file's data in Pinecone."""
    try:
        create_pinecone_index(user)

        # Process file
        document, image_paths = load_document(file_path)
        if not document:
            return False, "Failed to load document"

        texts = split_text(document)
        if not texts:
            return False, "Failed to split document into chunks"

        print(f"Created {len(texts)} text chunks")

        # Create embeddings
        embeddings = create_embeddings()
        if embeddings is None:
            return False, "Failed to create embeddings"

        # Store text embeddings in Pinecone with filename and IDs
        vector_ids = store_embeddings(texts, embeddings, user, collection, file_path)
        
        # Process images if present
        for image_path in image_paths:
            extracted_text = extract_text_from_image(image_path)
            image_doc = Document(
                page_content=extracted_text, 
                metadata={
                    "type": "image", 
                    "filename": os.path.basename(image_path),
                    "vector_id": str(uuid.uuid4())
                }
            )
            image_vector_ids = store_embeddings([image_doc], embeddings, user, collection, image_path)
            vector_ids.extend(image_vector_ids)
            print(f"Stored OCR-extracted text from {image_path}")

        os.remove(file_path)
        return True, f"File processed and stored successfully with vector IDs: {vector_ids}"
    except Exception as e:
        return False, f"Error processing file: {e}"

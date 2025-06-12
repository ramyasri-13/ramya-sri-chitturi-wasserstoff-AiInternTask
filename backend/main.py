from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import chromadb
from chromadb.utils import embedding_functions
import openai
import uuid
import json
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize collection
try:
    collection = chroma_client.get_collection(
        name="documents",
        embedding_function=embedding_function
    )
except:
    collection = chroma_client.create_collection(
        name="documents",
        embedding_function=embedding_function
    )

# OpenAI API key (replace with your key or use environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        text_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():  # If text is extractable
                text_content.append({
                    'page': page_num + 1,
                    'text': text.strip()
                })
            else:  # If text is not extractable, use OCR
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'text': ocr_text.strip()
                    })
        
        doc.close()
        return text_content
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return []

def extract_text_from_image(file_path):
    """Extract text from image using OCR"""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return [{'page': 1, 'text': text.strip()}] if text.strip() else []
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return []

def extract_text_from_txt(file_path):
    """Extract text from plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return [{'page': 1, 'text': text.strip()}] if text.strip() else []
    except Exception as e:
        logger.error(f"Error reading text file: {str(e)}")
        return []

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size // 2:
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def get_gpt_response(prompt, max_tokens=1000):
    """Get response from OpenAI GPT"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides accurate information with citations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting GPT response: {str(e)}")
        return f"Error: Could not process request - {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_documents():
    """Upload and process documents"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_docs = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text based on file type
                file_ext = filename.rsplit('.', 1)[1].lower()
                if file_ext == 'pdf':
                    text_content = extract_text_from_pdf(file_path)
                elif file_ext in ['png', 'jpg', 'jpeg']:
                    text_content = extract_text_from_image(file_path)
                elif file_ext == 'txt':
                    text_content = extract_text_from_txt(file_path)
                else:
                    continue
                
                if not text_content:
                    logger.warning(f"No text extracted from {filename}")
                    continue
                
                # Store in ChromaDB
                doc_id = str(uuid.uuid4())
                for page_data in text_content:
                    chunks = chunk_text(page_data['text'])
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_page{page_data['page']}_chunk{i}"
                        
                        collection.add(
                            documents=[chunk],
                            metadatas=[{
                                'doc_id': doc_id,
                                'filename': filename,
                                'page': page_data['page'],
                                'chunk_id': i,
                                'upload_time': datetime.now().isoformat()
                            }],
                            ids=[chunk_id]
                        )
                
                uploaded_docs.append({
                    'doc_id': doc_id,
                    'filename': filename,
                    'pages': len(text_content),
                    'upload_time': datetime.now().isoformat()
                })
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_docs)} documents',
            'documents': uploaded_docs
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """Query documents and return answers with themes"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Search in ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=20  # Get top 20 relevant chunks
        )
        
        if not results['documents'][0]:
            return jsonify({
                'individual_answers': [],
                'themes': [],
                'message': 'No relevant documents found for your query.'
            })
        
        # Group results by document
        doc_answers = {}
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            doc_id = metadata['doc_id']
            filename = metadata['filename']
            page = metadata['page']
            
            if doc_id not in doc_answers:
                doc_answers[doc_id] = {
                    'doc_id': doc_id,
                    'filename': filename,
                    'answers': [],
                    'pages': set(),
                    'content': []
                }
            
            doc_answers[doc_id]['answers'].append(doc)
            doc_answers[doc_id]['pages'].add(page)
            doc_answers[doc_id]['content'].append(doc)
        
        # Generate individual answers for each document
        individual_answers = []
        all_content_for_themes = []
        
        for doc_id, doc_data in doc_answers.items():
            combined_content = ' '.join(doc_data['content'])
            pages_str = ', '.join(map(str, sorted(doc_data['pages'])))
            
            # Generate answer for this specific document
            prompt = f"""
            Based on the following content from document '{doc_data['filename']}', provide a concise answer to the query: "{query}"
            
            Content: {combined_content[:2000]}
            
            Provide only the most relevant answer based on this document's content. Be specific and cite the information accurately.
            """
            
            answer = get_gpt_response(prompt, max_tokens=200)
            
            individual_answers.append({
                'doc_id': doc_id,
                'filename': doc_data['filename'],
                'answer': answer,
                'citation': f"Page(s) {pages_str}"
            })
            
            all_content_for_themes.append(f"Document {doc_data['filename']}: {combined_content}")
        
        # Generate themes from all content
        themes_content = '\n\n'.join(all_content_for_themes[:5])  # Limit content for theme analysis
        
        theme_prompt = f"""
        Analyze the following document contents and identify the main themes related to the query: "{query}"
        
        Document Contents:
        {themes_content}
        
        Instructions:
        1. Identify 2-3 main themes that emerge across the documents
        2. For each theme, provide a clear title and explanation
        3. Reference which documents support each theme
        4. Format your response as:
        
        Theme 1 - [Title]:
        [Explanation with document references]
        
        Theme 2 - [Title]:
        [Explanation with document references]
        """
        
        themes_response = get_gpt_response(theme_prompt, max_tokens=500)
        
        # Parse themes (simple parsing - can be improved)
        themes = []
        theme_sections = themes_response.split('Theme ')
        for section in theme_sections[1:]:  # Skip first empty section
            if ' - ' in section:
                title_part = section.split(':')[0]
                content_part = ':'.join(section.split(':')[1:]).strip()
                theme_number = title_part.split(' - ')[0]
                theme_title = title_part.split(' - ')[1] if ' - ' in title_part else title_part
                
                themes.append({
                    'theme_id': theme_number,
                    'title': theme_title,
                    'explanation': content_part,
                    'supporting_docs': [doc['filename'] for doc in individual_answers]
                })
        
        return jsonify({
            'individual_answers': individual_answers,
            'themes': themes,
            'total_documents_found': len(individual_answers)
        })
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    try:
        # Get all unique documents from ChromaDB
        all_data = collection.get()
        
        documents = {}
        for metadata in all_data['metadatas']:
            doc_id = metadata['doc_id']
            if doc_id not in documents:
                documents[doc_id] = {
                    'doc_id': doc_id,
                    'filename': metadata['filename'],
                    'upload_time': metadata['upload_time'],
                    'pages': set()
                }
            documents[doc_id]['pages'].add(metadata['page'])
        
        # Convert to list and format pages
        doc_list = []
        for doc_data in documents.values():
            doc_list.append({
                'doc_id': doc_data['doc_id'],
                'filename': doc_data['filename'],
                'upload_time': doc_data['upload_time'],
                'total_pages': len(doc_data['pages'])
            })
        
        return jsonify({
            'documents': doc_list,
            'total_count': len(doc_list)
        })
    
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Document Research Chatbot API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
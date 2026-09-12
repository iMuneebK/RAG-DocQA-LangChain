import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    def process_file(self, uploaded_file):
        """Processes an uploaded file and returns document chunks."""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_path)
            elif file_extension == '.txt':
                loader = TextLoader(temp_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        finally:
            os.remove(temp_path)

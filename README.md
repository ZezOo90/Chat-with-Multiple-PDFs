# Chat with Multiple PDFs  

**Chat with Multiple PDFs** is a Streamlit-based application that allows users to interact with multiple PDF documents by asking natural language questions. The app extracts text from uploaded PDFs, processes the content into searchable chunks, and provides detailed responses based on the context of the documents. It leverages LangChain, Google Generative AI, and FAISS for efficient text processing and conversational interaction.  

## Features  
- **Upload and Process PDFs**: Users can upload one or more PDF files, and the app processes the text for querying.  
- **Google Generative AI Integration**: Uses advanced AI models to generate detailed and contextually relevant responses.  
- **Streamlit Interface**: Intuitive and interactive interface for seamless user interaction.  
- **Text Splitting and Chunking**: Handles large PDF files by splitting the text into manageable chunks for better processing.  
- **Efficient Document Retrieval**: Utilizes FAISS for fast and accurate similarity-based searches on the processed text.  

## Prerequisites  
Before running the app, ensure you have the following:  
1. Python 3.9.  
2. Google API Key for accessing Google Generative AI services.  
3. Required Python libraries (listed in `requirements.txt`).  

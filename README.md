# Name: Swaraj Bhanja | Student Id: st125052

# Welcome to Retrieval Augmented Generation!

This is a web-based end-to-end application named Retrieval Augmented Generation. It leverages the power of deep learning and web development to provide a website that generates text responses based on the input prompts provided to it.


# About the Deep Learning Model

The deep learning architecture implemented in this system is designed for retrieval-augmented generation (RAG), combining semantic search with a large language model (LLM)-based response generation. This architecture enhances information retrieval by embedding textual documents into vector space and retrieving the most relevant chunks based on user queries. The system then leverages a generative model to produce well-structured, contextually relevant responses. Here, the information is stored as a PDF which was created using ChatGPT-o1 using the memory functionality in it. Any missing information was provided explicitly to it, based on which a final PDF was generated and returned by it.

Access the PDF here: [Swaraj.pdf](https://github.com/st125052/a6-nlp-retrieval-augmented-generation-st125052/blob/main/notebooks/pdfs/Swaraj.pdf)


# Creating the Prompt Template

A prompt template is defined using the `PromptTemplate` class from LangChain, which is a framework used for working with large language models. The template is designed to guide an AI assistant in answering questions based on a given context while ensuring it does not fabricate information. The template consists of instructions specifying that the assistant should use the provided context to answer the question and, if the answer is unknown, should explicitly state that. The `PromptTemplate.from_template()` method converts the raw string into a structured prompt template, allowing dynamic insertion of values. The `.format()` function is then used to populate the placeholders `{context}` and `{question}` with actual values, effectively generating a complete prompt that can be fed into a language model. In this case, the context provides details about a person’s background in computer science and AI, while the question asks about their background, leading to a formatted prompt ready for use.


# Reading the Knowledge Base And Parsing It

A PDF document is loaded using `PyMuPDFLoader` from LangChain, which is specifically designed to extract text from PDF files. The variable `nlp_docs` holds the path to the PDF file (Swaraj.pdf) located inside the `../pdfs/` directory. The `PyMuPDFLoader` class is then initialized with this file path, creating an instance called `loader`. The `load()` method is invoked on loader, which reads the entire contents of the PDF and extracts its text as structured documents, storing them in the documents variable. This allows further processing, such as splitting the extracted text into chunks or passing it into a language model for analysis or answering queries. Essentially, this code automates the process of reading a PDF and converting its contents into a format that can be used for NLP tasks. The extracted text from the PDF is split  into smaller chunks using `RecursiveCharacterTextSplitter` from LangChain. Since language models often have token limitations, it is essential to divide large documents into manageable segments. The `RecursiveCharacterTextSplitter` is specifically designed to intelligently split text while preserving contextual meaning by prioritizing sentence and paragraph boundaries. The `chunk_size` parameter is set to 700 characters, meaning each chunk will contain a maximum of 700 characters, while `chunk_overlap` is set to 100 characters, ensuring that consecutive chunks share an overlapping portion of text to maintain coherence between them. Finally, the `split_documents(documents)` method is called, which applies this splitting strategy to the previously loaded documents, storing the resulting text chunks in the variable doc. These smaller text fragments can now be efficiently processed by language models for various tasks, such as answering queries or summarization.


# Loading the Embeddings Model

An embedding model is initialized using `HuggingFaceInstructEmbeddings` from LangChain, leveraging a pre-trained model from Hugging Face. The model used here is `hkunlp/instructor-base`, which is a transformer-based embedding model designed to generate numerical vector representations (embeddings) of textual data. These embeddings capture semantic meaning, making them useful for tasks like information retrieval, similarity search, and document classification. The `model_kwargs` argument specifies additional parameters, including setting `"device": device`, which ensures that the model runs on the appropriate hardware (e.g., GPU or CPU). If a CUDA-enabled GPU is available, PyTorch (torch) will allocate the model to the GPU for faster computations. This embedding model will later be used to convert text chunks (from the PDF) into numerical vectors, enabling efficient semantic search and retrieval in AI applications.


# Vector Store Creation

A local vector database using `FAISS (Facebook AI Similarity Search)` is created to store and retrieve document embeddings efficiently. First, it checks if the directory `vector_path` `(../models/vector-store-base)` exists; if not, it creates the directory using `os.makedirs()` and prints a confirmation message. This ensures that the location for saving the vector database is prepared.

The core functionality is handled by `FAISS.from_documents()`, which takes the previously split document chunks `(doc)` and embeds them using `embedding_model` (which is based on `hkunlp/instructor-base`). FAISS is an optimized similarity search library that enables fast and scalable vector-based retrieval by indexing these embeddings.

Finally, the FAISS database is saved locally under the directory defined by `vector_path`, inside a subfolder named `nlp_stanford`. The `.save_local()` method ensures the index is stored under this location with the name `nlp`. This means the vectorized document representations are now persistently stored, allowing for fast, local retrieval of semantically similar text chunks when queried later.


# Retriever Modelling

The previously stored `FAISS` vector database is loaded from local storage and is prepared for semantic search and retrieval. The `vector_path` variable specifies the directory where the vector database was saved `('../models/vector-store-base')`, and db_file_name refers to the subfolder `('nlp_stanford')` where the indexed vectors were stored. Using `FAISS.load_local()`, the stored vector index is reloaded into memory, ensuring that it can be used without needing to recompute the embeddings from the original documents.

The embeddings parameter is set to `embedding_model`, meaning the same embedding model `(hkunlp/instructor-base)` is used for querying and retrieving relevant documents. The index_name='nlp' ensures the correct index is loaded.

Finally, `vectordb.as_retriever()` converts the `FAISS` vector store into a retriever object, which allows efficient semantic search. When given a query, this retriever can find and return the most relevant document chunks based on their embeddings. This setup enables fast and intelligent retrieval of stored knowledge without reprocessing the original text documents.


# LLM Chaining

A quantized transformer model for text generation is loaded and configured, making it efficient for inference. The model used here is `fastchat-t5-3b-v1.0`, a large language model optimized for dialogue-based applications.

First, the `AutoTokenizer.from_pretrained(model_id)` loads the tokenizer, which is responsible for converting text into numerical tokens and vice versa. The `pad_token_id` is set to the `eos_token_id`, ensuring proper token padding during generation.

The `BitsAndBytesConfig` is then used to enable model quantization, specifically 4-bit quantization using `NF4 (Normal Float 4)`. This drastically reduces memory usage while maintaining reasonable model accuracy. The `bnb_4bit_compute_dtype = torch.float16` ensures computations are performed in half-precision floating point, balancing speed and efficiency. The `bnb_4bit_use_double_quant = True` further optimizes the quantization process. When loading the model, `device_map='auto'` allows automatic allocation to available hardware (CPU or GPU), while `load_in_8bit = True` provides another layer of quantization.

The `pipeline()` function initializes a text-to-text generation pipeline that takes input text and generates a response. The `max_new_tokens = 256` restricts the output length, while `"temperature": 0` ensures deterministic responses (lower temperature = less randomness). The `"repetition_penalty": 1.5` discourages repetitive outputs, making responses more natural.

Finally, `HuggingFacePipeline(pipeline=pipe)` wraps the pipeline inside a LangChain-compatible LLM object, enabling seamless integration with other LangChain components, such as retrievers or conversational agents. This setup allows efficient inference on a quantized model while minimizing memory usage, making it suitable for real-time applications.


# Generation Modelling

A question rephrasing mechanism is prepared using a large language model (LLM) chain in LangChain. The goal is to reframe user queries into clearer, more contextually relevant questions before retrieving relevant information.

The LLMChain class is used to create a chain where the LLM (llm, which was previously configured using `fastchat-t5-3b-v1.0`) processes inputs based on a specific prompt. Here, the `CONDENSE_QUESTION_PROMPT` is used as the predefined template for restructuring conversational queries. This prompt is especially useful in multi-turn conversations, where user queries may depend on previous interactions. The chain ensures that follow-up questions are rewritten to include necessary context, making retrieval more effective. By setting `verbose=True`, debugging information about the chain's execution will be displayed, helping in monitoring how questions are rephrased. This question generator will later be used to improve retrieval accuracy by converting vague or incomplete follow-up questions into well-structured standalone queries.

A document-based question-answering (QA) chain is then generated using LangChain’s `load_qa_chain()` function. The QA chain is responsible for retrieving relevant document chunks and using the LLM (llm) to generate answers based on those chunks. The `chain_type='stuff'` option specifies how retrieved documents are processed. The "stuff" strategy means that all retrieved document chunks are concatenated together and fed into the LLM as context for answering the query. While this is efficient for small amounts of data, it may not be ideal for large documents due to token limitations.

The `prompt = PROMPT` ensures that the LLM answers questions using the structured template defined earlier, reinforcing strict factual retrieval (i.e., the model should not make up answers). Setting `verbose=True` allows logging of intermediate steps, helping debug how the model processes input documents. Once created, `doc_chain` becomes a fully functional question-answering component, capable of retrieving relevant documents and generating informed responses based on them.


# Deployment

For deployment purposes, the retriever model is replaced with `OpenAIEmbeddings`, and the generation model is replaced with `GPT-4o-mini`, ensuring improved scalability, efficiency, and coherence in responses. By leveraging OpenAI’s embedding model, the document retrieval process benefits from highly optimized, dense vector representations that enhance semantic search accuracy. This means that even if a query is phrased differently from the original text, the retriever can still locate the most relevant document chunks with higher precision.

Replacing `fastchat-t5-3b-v1.0` with `GPT-4o-mini` for response generation ensures greater fluency, contextual awareness, and factual consistency. Unlike fine-tuned models that may struggle with generalization beyond their training data, `GPT-4o-mini` can generate well-structured, logically consistent responses across a broad range of queries. The zero-shot reasoning capability of GPT-4o-mini further allows it to understand nuanced user inputs, dynamically rephrase queries for better retrieval, and synthesize answers in a way that feels natural and human-like.

Moreover, OpenAI’s API-based approach eliminates the need for manual quantization, model hosting, and fine-tuning, leading to faster deployment, lower maintenance overhead, and better scalability in production environments. The integration with `ConversationalRetrievalChain` ensures that responses are context-aware, meaning the model remembers prior interactions within a conversation window, making follow-up responses more coherent and engaging. This architecture ultimately elevates the user experience, providing a fast, reliable, and accurate AI-driven conversational assistant.


# Analysis

Both variants of Generators and Retrievers were analyzed [here](https://github.com/st125052/a6-nlp-retrieval-augmented-generation-st125052/blob/main/notebooks/pdfs/Analysis%20of%20RAG%20Models.pdf).

# Website Creation

The model was then hosted over the Internet with Flask as the backend, HTML, CSS, JS as the front end, and Docker as the container. The end-user is presented with a UI wherein a search input box is present. Once the user types in the first set of words, they click on the `Go` button or hit the 'Return' key on their keyboard. The input texts are sent to the JS handler which makes an API call to the Flask backend. The Flask backend has the GET route which intercepts the HTTP request. The input text is then fed to the model to generate the response. The result is then returned back to the JS handler as a list by the Flask backend. The JS handler then appends each token in the received list into the result container's inner HTML and finally makes it visible for the output to be shown. Any further interaction is captured by appending a new container to the existing document via JS. 

A Vanilla architecture was chosen due to time constraints. In a more professional scenario, the ideal approach would be used frameworks like React, Angular and Vue for Frontend and ASP.NET with Flask or Django for Backend.

The following describes the key points of the hosting discussion.
> **1. DigitalOcean (Hosting Provider)**
> 
>> - **Role:** Hosting and Server Management
>> - **Droplet:** Hosts the website on a virtual server, where all files, databases, and applications reside.
>> - **Dockerized Container:** The website is hosted in a Dockerized container running on the droplet. The container is built over a Ubuntu Linux 24.10 image.
>> - **Ports and Flask App:** The Dockerized container is configured to host the website on port 8000. It forwards requests to port 5000, where the Flask app serves the backend and static files. This flask app contains the pickled model, which is used for prediction.
>> - **IP Address:** The droplet’s public IP address directs traffic to the server.
>
>  **In Summary:** DigitalOcean is responsible for hosting the website within a Dockerized container, ensuring it is online and accessible via its IP address.
> 
>  **2. GoDaddy (Domain Registrar)**
>
>> - **Role:** Domain Registration and Management
>> - **Domain Purchase:** Registers and manages the domain name.
>> - **DNS Management:** Initially provided DNS setup, allowing the domain to be pointed to the DigitalOcean droplet’s IP address.
> 
> **In Summary:** GoDaddy ensures the domain name is registered and correctly points to the website’s hosting server.
>
>  **3. Cloudflare (DNS and Security/Performance Optimization)**
>
>> - **Role:** DNS Management, Security, and Performance Optimization
>> - **DNS Management:** Resolves the domain to the correct IP address, directing traffic to the DigitalOcean droplet.
>> - **CDN and Security:** Caches website content globally, enhances performance, and provides security features like DDoS protection and SSL encryption.
> 
> **In Summary:** Cloudflare improves the website’s speed, security, and reliability.
>
> **How It Works Together:**
> 
>> - **Domain Resolution:** The domain is registered with GoDaddy, which points it to Cloudflare's DNS servers. Cloudflare resolves the domain to the DigitalOcean droplet's IP address.
>> - **Content Delivery:** Cloudflare may serve cached content or forward requests to DigitalOcean, which processes and serves the website content to users.
> 
> **Advantages of This Setup:**
>
>> - **Security:** Cloudflare provides DDoS protection, SSL/TLS encryption, and a web application firewall.
>> - **Performance:** Cloudflare’s CDN reduces load times by caching content globally, while DigitalOcean offers scalable hosting resources.
>> - **Reliability:** The combination of GoDaddy, Cloudflare, and DigitalOcean ensures the website is always accessible, with optimized DNS resolution and robust hosting.

# Demo
https://github.com/user-attachments/assets/1cb59880-b713-42e5-985b-115f145b16fb

# Access The Final Website

You can access the website [here](https://aitmltask.online). 


# Limitations

Note that the model predicts only responses to a specific set of questions, as directed by the content in the PDF. Also, it may generate unwanted or hallucinated content for some contents, which is a known limitation.


# Important Notes

To improve safety, the existing OpenAI key used for training has been decommissioned, and a new key has been added in its place.


# How to Run the Retrieval Augmented Generation Docker Container Locally

### Step 1: Clone the Repository
> - First, clone the repository to your local machine.
### Step 2: Install Docker
> - If you don't have Docker installed, you can download and install it from the [Docker](https://www.docker.com) website.
### Step 3: Build and Run the Docker Container
Once Docker is installed, navigate to the app folder in the project directory. Delete the docker-compose-deployment.yml file and run the following commands to build and run the Docker container:
> - `docker compose up -d`

### Important Notes
> - The above commands will serve the Docker container on port 5000 and forward the requests to the Flask application running on port 5000 in the containerized environment.
> - Ensure Ports Are Free: Make sure that port 5000 is not already in use on your machine before running the container.
> - Changing Flask's Port: If you wish to change the port Flask runs on (currently set to 5000), you must update the port in the app.py file. After making the change, remember to rebuild the Docker image in the next step. Execute the following command to stop the process: `docker compose down`. Then goto Docker Desktop and delete the container and image from docker. 

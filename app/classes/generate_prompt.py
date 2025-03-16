import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain


def get_openai_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def get_retriever():
    embedding_model = OpenAIEmbeddings()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    vector_path = os.path.join(base_dir, '..', 'models', 'vector-store-openai')
    
    db_file_name = 'nlp_stanford'
    
    vectordb = FAISS.load_local(
        folder_path=os.path.join(vector_path, db_file_name),
        embeddings=embedding_model,
        index_name='nlp'
    ) 
    
    return vectordb.as_retriever()

def get_llm():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  
        temperature=0.8  
    )
    return llm

def get_question_generator(llm):
    question_generator = LLMChain(
        llm = llm,
        prompt = CONDENSE_QUESTION_PROMPT,
        verbose = True
    )
    return question_generator

def get_doc_chain(llm):
    prompt_template = """
        You are a helpful assistant that answers questions about the person based on their personal documents.
        Use the following context to answer the question. If you don't know the answer, just say you don't know.
        Don't make things up.    

        Context: {context}
        Question: {question}
        Answer:
        """.strip()

    PROMPT = PromptTemplate.from_template(
        template = prompt_template
    )

    doc_chain = load_qa_chain(
        llm = llm,
        chain_type = 'stuff',
        prompt = PROMPT,
        verbose = True
    )

    return doc_chain
    
def get_conversation_chain():
    llm = get_llm()
    
    retriever = get_retriever()
    question_generator = get_question_generator(llm)
    doc_chain = get_doc_chain(llm)

    memory = ConversationBufferWindowMemory(
        k=3, 
        memory_key = "chat_history",
        return_messages = True,
        output_key = 'answer'
    )

    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory=memory,
        verbose=True,
        get_chat_history=lambda h : h
    )

    return chain


def get_prediction(prompt_question, chain):
    answer = chain({"question":prompt_question}) 

    final_answer = {
        "answer": answer['answer'],
        "source_documents": answer['source_documents'][0].metadata['source'],
    }   

    return final_answer
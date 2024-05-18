import os
from dotenv import load_dotenv
load_dotenv()
import nest_asyncio
nest_asyncio.apply()
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

import streamlit as st

### api keys 
# LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# from llama_parse import LlamaParse  

# parser = LlamaParse(
#     api_key=LLAMA_CLOUD_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY or set it up here
#     result_type="markdown", ## text
#     language='en'
# )
# documents = LlamaParse(result_type="markdown").load_data("./HAI_AI-Index-Report-20242.pdf")

loader = PyPDFLoader("HAI_AI-Index-Report-20242.pdf")
pages = loader.load_and_split()

# documents = LlamaParse(result_type="markdown").load_data("./HAI_AI-Index-Report-20242.pdf")

gpt4all_embd = GPT4AllEmbeddings()
semantic_chunker1 = SemanticChunker(gpt4all_embd)
semantic_chunks1 = semantic_chunker1.create_documents([d.page_content for d in pages]) ## documents.text

### vector database
vectorstoreS1 = Chroma.from_documents(semantic_chunks1, 
                                     collection_name="rag",
                                     embedding=gpt4all_embd)
retriever1 = vectorstoreS1.as_retriever() ## search_kwargs={"k": 1}

#### chain

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768",groq_api_key = GROQ_API_KEY )


### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever1, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def main():
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.title('RAG App')
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        session_id = str(uuid.uuid4())

        with st.chat_message("assistant"):
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


if __name__ == "__main__":
    main()
"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging

import wandb
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from prompts import load_chat_prompt


logger = logging.getLogger(__name__)


def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    # load vector store artifact
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config.vector_store_artifact, type="search_index"
    ).download()
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )
    # i = 0
    # print("\n\n EXIBIR TODOS OS DOCUMENTOS\n\n ================================================")
    # for doc in vector_store.get()["documents"]:
    #     i = i+1
    #     if i < 30:
    #         print(f"========\n{doc}\n======")
    #     else:
    #         break

    return vector_store


def load_chain(wandb_run: wandb.run, vector_store: Chroma, openai_api_key: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=wandb_run.config.model_name,
        temperature=wandb_run.config.chat_temperature,
        max_retries=wandb_run.config.max_fallback_retries,
    )
    chat_prompt_dir = wandb_run.use_artifact(
        wandb_run.config.chat_prompt_artifact, type="prompt"
    ).download()
    qa_prompt = load_chat_prompt(f"{chat_prompt_dir}/prompt.json")
    print(f"\n QA PROMPT ==================================================== {qa_prompt}\n")
    adjusted_prompt = qa_prompt.format(question="{question}", context="{context}")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    print(f"\n QA chain ==================================================== {qa_chain}\n")
    return qa_chain


def get_answer(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: list[tuple[str, str]],
    retrieved_docs: list,
    context: str = "",
    
):
    """Get an answer from a ConversationalRetrievalChain
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """

    inputs={"question": "contexto: "+ context+ "Dado o contexto, responda:" +question, "chat_history": chat_history }
    print(type(inputs)) 
    
    result = chain(inputs=inputs, return_only_outputs=True)

    # Print retrieved fragments
    source_documents = result.get("source_documents", [])
    print("\nDocumentos recuperados\n")

    for idx, doc in enumerate(source_documents, start=1):
        print(f"Fragment {idx}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}\n")

    # Return the answer
    response = f"\t{result['answer']}"

    return response

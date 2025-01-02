"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import os
from types import SimpleNamespace
import wandb
from langchain_openai import ChatOpenAI
from prompts import load_chat_prompt
from config import default_config
from chain import get_answer, load_chain, load_vector_store


class Chat:
    """A chatbot interface that persists the vectorstore and chain between calls."""

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        """Initialize the chatbot.
        Args:
            config (SimpleNamespace): The configuration.
        """
        self.config = config
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.vector_store = None
        self.chain = None

    def __call__(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        openai_api_key: str = None,
    ):
        """Answer a question about wandb documentation using the LangChain QA chain and vector store retriever.
        Args:
            question (str): The question to answer.
            history (list[tuple[str, str]] | None, optional): The chat history. Defaults to None.
            openai_api_key (str, optional): The OpenAI API key. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """
        if openai_api_key is not None:
            openai_key = openai_api_key
        elif os.environ["OPENAI_API_KEY"]:
            openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                "Please provide your OpenAI API key as an argument or set the OPENAI_API_KEY environment variable"
            )

        if self.vector_store is None:
            self.vector_store = load_vector_store(
                wandb_run=self.wandb_run, openai_api_key=openai_key
            )
        
            
        self.retriever = self.vector_store.as_retriever()
        history = history or []
        question = question.lower()
        # Recupere os documentos uma única vez
        retrieved_docs = self.retriever.get_relevant_documents(question)
        print("\n\n retrieved_docs ========================================================= \n\n")
        print(retrieved_docs)
        
        # Obtenha o contexto extra manualmente
        unique_sources = set(doc.metadata["source"] for doc in retrieved_docs)
        print("\n\n unique source ========================================================= \n\n")
        print(unique_sources)
        context = []
        for source_file in unique_sources:
            with open(source_file, "r") as f:
                context.append(f.read())
        combined_context = "\n\n".join(context)

        print("\n\n COMBINADO O TEXTO ========================================================= \n\n")
        print(combined_context)
        
        self.chat_prompt_dir = self.wandb_run.use_artifact(
            self.wandb_run.config.chat_prompt_artifact, type="prompt"
        ).download()
        
        self.qa_prompt = load_chat_prompt(f"{self.chat_prompt_dir}/prompt.json")
        
        self.adjusted_prompt = self.qa_prompt.format(question=f"{question}", context=f"{combined_context}")
        
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=self.wandb_run.config.model_name,
            temperature=self.wandb_run.config.chat_temperature,
            max_retries=self.wandb_run.config.max_fallback_retries,
        )
        
        response = self.llm.invoke(self.adjusted_prompt)
        
        print(f"\n\nCHAT HISTORY =================================== {history}\n\n")
        history.append((question, response))
        return history, history
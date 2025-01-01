"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import os
from types import SimpleNamespace

import gradio as gr
import wandb
from chain import get_answer, load_chain, load_vector_store
from config import default_config


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
        if self.chain is None:
            self.chain = load_chain(
                self.wandb_run, self.vector_store, openai_api_key=openai_key
            )

        history = history or []
        question = question.lower()
        # Recupere os documentos uma única vez
        retriever = self.vector_store.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(question)
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

        # Obtenha a resposta, passando os documentos e o contexto
        response = get_answer(
            chain=self.chain,
            question=question,
            chat_history=history,
            retrieved_docs=retrieved_docs,
            context=combined_context,
        )
        print(f"\n\nCHAT HISTORY =================================== {history}\n\n")
        history.append((question, response))
        return history, history



with gr.Blocks() as demo:
    gr.HTML(
        """
    <style>
        :root{
            --azul-claro: #1A73E8;
            --azul-medio: #145dbd;
            --azul-escuro: #163C70;
            --branco-fundo-1: #f6f6f6;
            --branco-fundo-2: #f8f9fa;
            --branco-fundo-3: #ccc;
            --cinza-claro: #e9ecef;
            --cinza-medio: #9e9c9c;
            --cinza-escuro: #616161;
        }
        .cabecalho{
            width: 100%;
            background-color: var(--azul-escuro);
        }
        .titulo{
            display: flex;
            flex-direction: row;
            justify-content: center;
            color: white;
            font-family: 'Montserrat', sans-serif;
            min-height: 80px;
        }
        .logo{
            padding-right: 15px;
            align-self: center;
        }
        .logo img{
            height: 40px;
        }
    </style>

    <div class="cabecalho">
        <div class="titulo">
            <div class="logo"><img src="/web/img/logo_al.png"></div>
            <div class="rotulo-titulo"><h1 style="color:white; margin-top:20px;">Daphane</h1></div>
        </div>
    </div>
        """
    )
    state = gr.State()
    chatbot = gr.Chatbot()

    with gr.Row():
        question = gr.Textbox(
            placeholder="Digite sua pergunta",
            label="",
            show_label=False
        )
        openai_api_key = gr.Textbox(
            type="password",
            label="Enter your OpenAI API key here",
            visible=False
        )
        botao = gr.Button("Enviar")
        
    question.submit(
        Chat(
            config=default_config,
        ),
        [question, state, openai_api_key],
        [chatbot, state],
    )
    botao.click(
        Chat(
            config=default_config,
        ),
        [question, state, openai_api_key],
        [chatbot, state],
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=False, server_name="0.0.0.0", server_port=8884, show_error=True
    )

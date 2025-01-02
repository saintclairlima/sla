# Bibliotecas pra rodar a api
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware

# Importa sa coisas que você já está usando
from app import Chat
from config import default_config

# Colocando o controller para esperar os requests...
controller = FastAPI()
controller.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allow all methods
    allow_headers=['*'],  # Allow all headers
)

# Instancia o Chat lá do seu código
chat = Chat(config=default_config)

TAGS_SUBSTITUICAO_HTML= {
    'TAG_INSERCAO_URL_HOST': 'http://localhost:8000', # aqui tem que ser o IP:porta que a api tá rodando
                                                      # Pra não precisar mexer no html cada vez que mudar de IP
                                                      # em produção. Pode colocar lá no config.py, se preferir
}

@controller.post('/chat/enviar_pergunta/')
async def gerar_resposta(dadosRecebidos: dict):
    pergunta = dadosRecebidos['pergunta']
    historico = dadosRecebidos['contexto']
    chave_openai = ''
    
    resultado = chat(question=pergunta, history=historico, openai_api_key=chave_openai)
    #resultado = chat(question=pergunta, history=historico, openai_api_key=chave_openai)
    
    # Aqui pode formatar o resultado para ficar de acordo com o que a página html for esperar pra imprimir a resposta
    
    return resultado


@controller.get('/chat/')
async def pagina_chat(url_redirec: str = Query(None)):
    with open('web/chat.html', 'r', encoding='utf-8') as arquivo: conteudo_html = arquivo.read()
        
    # substituindo as tags dentro do HTML, para maior controle
    for tag, valor in TAGS_SUBSTITUICAO_HTML.items():
        conteudo_html = conteudo_html.replace(tag, valor)
    return HTMLResponse(content=conteudo_html, status_code=200)

@controller.get('/web/img/favicon/favicon.ico')
async def favicon(): return FileResponse('web/img/favicon/favicon.ico')

@controller.get('/web/img/favicon/favicon.svg')
async def favicon(): return FileResponse('web/img/favicon/favicon.svg')

@controller.get('/web/img/favicon/favicon-48x48.png')
async def favicon(): return FileResponse('web/img/favicon/favicon-48x48.png')

@controller.get('/web/img/favicon/apple-touch-icon.png')
async def favicon(): return FileResponse('web/img/favicon/apple-touch-icon.png')

@controller.get('/web/img/favicon/site.webmanifest')
async def favicon(): return FileResponse('web/img/favicon/site.webmanifest')

@controller.get('/web/img/Assistente.png')
async def legisberto(): return FileResponse('web/img/Assistente.png')

@controller.get('/web/img/logo_al.png')
async def legisberto(): return FileResponse('web/img/logo_al.png')

print('API inicializada')
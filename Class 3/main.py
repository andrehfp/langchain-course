from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate


from dotenv import load_dotenv
import os


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_organization = os.getenv("OPENAI_ORGANIZATION")

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key, openai_organization=openai_organization
)


def create_vector_from_yt_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, language="pt")
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key,
        openai_organization=openai_organization,
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """Você é um assistente que responde perguntas sobre vídeos do youtube baseado
        na transcrição do vídeo.

        Responda a seguinte pergunta: {pergunta}
        Procurando nas seguintes transcrições: {docs}

        Use somente informação da transcrição para responder a pergunta. Se você não sabe, responda
        com "Eu não sei".

        Suas respostas devem ser bem detalhadas e verbosas.
        """,
            )
        ]
    )

    chain = LLMChain(llm=llm, prompt=chat_template, output_key="answer")

    response = chain({"pergunta": query, "docs": docs_page_content})

    return response, docs


if __name__ == "__main__":
    db = create_vector_from_yt_url("https://www.youtube.com/watch?v=Nu5gyrpaivE")
    response, docs = get_response_from_query(
        db, "O que é falado sobre controle de preços?"
    )
    print(response)

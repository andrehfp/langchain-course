from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_organization = os.getenv("OPENAI_ORGANIZATION")


def generate_company_name():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key,
        openai_organization=openai_organization,
    )

    company_name = llm(
        [
            SystemMessage(
                content="Você é um assistente IA que sempre responde em Português do Brasil"
            ),
            HumanMessage(
                content="Gere 5 ideias de nomes para empresas no segmento Pets"
            ),
        ]
    )

    return company_name


if __name__ == "__main__":
    print(generate_company_name())

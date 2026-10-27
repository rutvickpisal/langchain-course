from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Func call began.")
    information = """
        • The Rebellion: Inspired by the wise boar Old Major, the mistreated animals of Manor Farm rise up and chase away the cruel farmer, Mr. Jones. They rename it Animal Farm and establish the Seven Commandments of Animalism, anchored by the core rule that "All animals are equal".
        • Rise of the Pigs: The intelligent pigs—principally Napoleon and Snowball—take charge. Soon after, Napoleon forces Snowball off the farm using a pack of vicious, trained dogs and claims absolute control.
        • Corruption of Ideals: Aided by a persuasive pig named Squealer, Napoleon alters the commandments and rewrites history to grant the pigs special privileges, such as sleeping in beds and drinking alcohol.
        • The Final Transformation: The hardworking animals face intense labor and terror, but are conditioned to obey. By the end of the story, the pigs walk on two legs, wear human clothes, and become completely indistinguishable from the oppressive humans they originally overthrew.
    """

    summary_template = """
    Given ONLY the information below about a book:
    {information}

    Write:
    1. A short summary (2-3 sentences)
    2. Two interesting facts about it
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOllama(model="llama3.2:latest", temperature=0)

    chain = prompt | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
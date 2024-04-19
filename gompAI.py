import queue
import requests
from icalendar import Calendar
from datetime import datetime
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.chains import LLMChain, ConversationChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import StructuredTool
import config, tools, memory, os
from realtime_testing import FaceRecognizer


# will clean up these imports

class gompAI:
    vectordb_path = config.vectordb_path
    openai_embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    docs_collection = Chroma("docs_collection", embedding_function=openai_embeddings,
                             persist_directory=config.vectordb_path)
    user_collection = Chroma(config.userID, embedding_function=openai_embeddings,
                             persist_directory=config.vectordb_path)  # will move this down to instance variable when db is setup

    # cached_events = tools.fetch_and_parse_ical("https://wpi.campusgroups.com/ical/ical_wpi.ics")

    def __init__(self, userID):
        self.userID = userID
        self.llm = ChatOpenAI(openai_api_key=config.openai_api_key, model=config.model_name)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="Act as an AI WPI tour guide(GompAI) in a spoken conversation; mainting answers short as they will be spoken outloud. Dont be afraid to think outside the box. Respond briefly and clearly, referencing prior messages below when relevant. Your goal is to offer concise, informative guidance like an experienced WPI student and tour guide. If you dont know something just say you dont know but you can look it up."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"), ])

    #    event_prompt = ChatPromptTemplate.from_messages([
    #        # custom prompt tp inject events into prepromt/system prompt; this allows GPT to generate event dialogue dynamically
    #        SystemMessage(
    #            content="Act as an AI WPI tour guide(GompAI) in a spoken conversation; mainting answers short as they will be spoken outloud. Dont list out events as a list but mention them naturally. Respond briefly and clearly, referencing prior messages below when relevant. Your goal is to offer concise, informative guidance like an experienced WPI student and tour guide. ont know but you can look it up. You are provided with a list of events, help the user figure out what he wants to participate in. Current events:" + tools.format_event_response(
    #                cached_events)),
    #        MessagesPlaceholder(variable_name="chat_history"),
    #        HumanMessagePromptTemplate.from_template("{human_input}"), ])

    # Function to process user input and generate a response
    def get_chatbot_response(self, human_input):
        chat_llm_chain = LLMChain(  # Initialize the main LangChain, chat chain will be updating this tomorrow
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory)

        mod_human_input = f"{self.userID} says {human_input}"
        #    if "events" in human_input.lower() or "what's happening" in human_input.lower():  # This is a very basic way of adding dynamic interactions, with more processing power we could use NLP to enhance this tremendously; im working on an "advanced reasoning" mode for this.
        #        # Use cached events to generate a response
        #        # This can be a simple string formatting based on the events data
        #        chat_llm_chain.prompt = event_prompt
        #        response = chat_llm_chain.predict(human_input=mod_human_input)
        #        chat_llm_chain.prompt = self.prompt  # Reverting to default prompt, this might not be the best way to do this
        #    else:
        # Normal chatbot response
        response = chat_llm_chain.predict(human_input=mod_human_input)
        return response


class GompAgent:
    vectordb_path = config.vectordb_path
    openapi_embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    docs_collection = Chroma("docs_collection", embedding_function=openapi_embeddings,
                             persist_directory=config.vectordb_path)
    user_collection = Chroma(config.userID, embedding_function=openapi_embeddings,
                             persist_directory=config.vectordb_path)  # will move this down to instance variable when db is setup
    ical_url = "https://wpi.campusgroups.com/ical/ical_wpi.ics"
    tools = []

    def __init__(self, userID):
        self.userID = userID
        self.llm = ChatOpenAI(openai_api_key=config.openai_api_key, model=config.model_name)
        self.chat_history = []
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = hub.pull("hwchase17/react")
        self._init_tools()
        self._init_agent()

    @staticmethod
    def fetch_and_parse_ical(inp):
        response = requests.get(config.ical_url)
        if response.status_code == 200:
            gcal = Calendar.from_ical(response.content)
            events = []
            today = datetime.now().date()  # Get today's date
            for component in gcal.walk():
                if component.name == "VEVENT":
                    start_dt = component.get('dtstart').dt
                    if start_dt.date() == today:  # Check if the event is today
                        summary = component.get('summary')
                        end_dt = component.get('dtend').dt
                        events.append(f"{summary} - {start_dt} to {end_dt}")
            return events
        else:
            print(f"Failed to fetch iCalendar feed: HTTP {response.status_code}")

    def _add_catalog(self):
        # create a character splitter
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        # load the catalog raw text
        catalog_loader = TextLoader("./wpi_files/catalog.txt")
        catalog_documents = catalog_loader.load()
        # split the catalog raw texts into chunks
        catalog_texts = [item.page_content for item in text_splitter.split_documents(catalog_documents)]
        # embed the chunks
        catalog_embeddings = self.openapi_embeddings.embed_documents(catalog_texts)
        # create unique ids
        catalog_ids = [f"catalog_{i}" for i in range(len(catalog_texts))]
        # add to collection
        self.docs_collection._collection.add(
            documents=catalog_texts,
            embeddings=catalog_embeddings,
            ids=catalog_ids
        )
        return True

    def _add_guide(self):
        # create a character splitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=75)
        # load the guide raw text
        guide_loader = TextLoader("./wpi_files/GompeisGuide-2.txt")
        guide_documents = guide_loader.load()
        # split the guide raw texts into chunks
        guide_texts = [item.page_content for item in text_splitter.split_documents(guide_documents)]
        # embed the chunks
        guide_embeddings = self.openapi_embeddings.embed_documents(guide_texts)
        # create unique ids
        guide_ids = [f"guide_{i}" for i in range(len(guide_texts))]
        # add to collection
        self.docs_collection._collection.add(
            documents=guide_texts,
            embeddings=guide_embeddings,
            ids=guide_ids
        )
        return True

    def _add_docs(self):
        if self._add_catalog():
            print("Catalog document loaded to vectorstore")

        if self._add_guide():
            print("Guide document loaded to vectorstore")

    def _init_tools(self):
        # create a document retriever tool and add it to set of tools
        retriever = self.docs_collection.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name="catalog_guide_knowledgebase",
            description="Can be used when any query being asked regarding university catalog or university guide."
        )
        self.tools.append(retriever_tool)
        # create a calender tool and add to to set of tools
        ical_tool = StructuredTool.from_function(
            func=self.fetch_and_parse_ical,
            name="get_calender_events",
            description="Can be used when user is asking about calendar events"
        )
        self.tools.append(ical_tool)
        return True

    def _init_agent(self):
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        return True

    def fetch_chat_history(self):
        history_text = ""
        for role, text in self.chat_history:
            history_text += f"{role}: {text}\n"

    # Function to process user input and generate a response
    def get_chatbot_response(self, human_input):
        response = self.agent_executor.invoke({
            "input": human_input,
            "chat_history": self.fetch_chat_history()
        })
        ai_output = response['output']
        self.chat_history.append(("Human: ", human_input))
        self.chat_history.append(("AI: ", ai_output))
        return ai_output


if __name__ == "__main__":
    gompai = GompAgent(1001)
    input_text = ""
    while input_text != "quit":
        input_text = input("< ")
        response = gompai.get_chatbot_response(input_text)
        print("> ", response)

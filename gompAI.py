import queue

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.chains import LLMChain, ConversationChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import config, tools, memory, os
from realtime_testing import FaceRecognizer


class gompAI:
   vectordb_path = config.vectordb_path
   openai_embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
   docs_collection = Chroma("docs_collection", embedding_function=openai_embeddings,
                         persist_directory=config.vectordb_path)
   user_collection = Chroma(config.userID, embedding_function=openai_embeddings, persist_directory=config.vectordb_path) #will move this down to instance variable when db is setup
   
   def __init__(self, userID):
        self.userID = userID
        self.llm = ChatOpenAI(openai_api_key=config.openai_api_key, model=config.model_name)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = ChatPromptTemplate.from_messages([SystemMessage(content="Act as an AI WPI tour guide(GompAI) in a spoken conversation; mainting answers short as they will be spoken outloud. Dont be afraid to think outside the box. Respond briefly and clearly, referencing prior messages below when relevant. Your goal is to offer concise, informative guidance like an experienced WPI student and tour guide. If you dont know something just say you dont know but you can look it up."),MessagesPlaceholder(variable_name="chat_history"),HumanMessagePromptTemplate.from_template("{human_input}"),])
   
   # Cache events
   ical_url = "https://wpi.campusgroups.com/ical/ical_wpi.ics"
   cached_events = tools.fetch_and_parse_ical(ical_url)
   
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

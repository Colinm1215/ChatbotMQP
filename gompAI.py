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

# Initialize OpenAI chat model 
llm = ChatOpenAI(openai_api_key=config.openai_api_key, model=config.model_name)

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)

docs_collection = Chroma("docs_collection", embedding_function=openai_embeddings,
                         persist_directory=config.vectordb_path)
user_collection = Chroma(config.userID, embedding_function=openai_embeddings, persist_directory=config.vectordb_path)
# Constructor for default prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="Act as an AI WPI tour guide(GompAI) in a spoken conversation; mainting answers short as they will be spoken outloud. Dont be afraid to think outside the box. Respond briefly and clearly, referencing prior messages below when relevant. Your goal is to offer concise, informative guidance like an experienced WPI student and tour guide. If you dont know something just say you dont know but you can look it up."
        ),  # The persistent system prompt, need to add "self-learning" system prompt for each user
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memories will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human speech input will injected
    ]
)

# Creating the retriever and memory (still working on vector memory, but this abstraction might be uneeded)
# retriever = chroma_store.as_retriever(search_kwargs=dict(k=3))
# chroma_memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key='chat_history', return_messages=True)
# Uncomment for populating memories
# chroma_memory.save_context({"input": "Hey my name is Eduardo"}, {"output": "that's good to know, im GompAI how are you doing Eduardo?"})
# chroma_memory.save_context({"input": "I am a 4th year cs student working on my MQP, which is you GompAI"}, {"output": "Wow, pretty good project!"})
# chroma_memory.save_context({"input": "Oh I also really enjoy going to music events"}, {"output": "that's good to know, want to hear about music events today?"})


# Initialize the conversation memory for short term and intra conversation memory
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)  # return_messages is just testing purposes

# Cache events
ical_url = "https://wpi.campusgroups.com/ical/ical_wpi.ics"
cached_events = tools.fetch_and_parse_ical(ical_url)

# Initialize the main LangChain chat chain
chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,  # verbose is just for testing, prints entire prompt and chain
    memory=memory
    # still testing how to handle different memory objects with realtime processing, vector memory is brittle right now; but can still be used to provide context in preprompt
)

event_prompt = ChatPromptTemplate.from_messages([
    # custom prompt tp inject events into prepromt/system prompt; this allows GPT to generate event dialogue dynamically
    SystemMessage(
        content="Act as an AI WPI tour guide(GompAI) in a spoken conversation; mainting answers short as they will be spoken outloud. Dont list out events as a list but mention them naturally. Respond briefly and clearly, referencing prior messages below when relevant. Your goal is to offer concise, informative guidance like an experienced WPI student and tour guide. ont know but you can look it up. You are provided with a list of events, help the user figure out what he wants to participate in. Current events:" + tools.format_event_response(
            cached_events)),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}"), ])


# Function to process user input and generate a response
def get_chatbot_response(human_input):
    if "events" in human_input.lower() or "what's happening" in human_input.lower():  # This is a very basic way of adding dynamic interactions, with more processing power we could use NLP to enhance this tremendously; im working on an "advanced reasoning" mode for this.
        # Use cached events to generate a response
        # This can be a simple string formatting based on the events data
        chat_llm_chain.prompt = event_prompt
        response = chat_llm_chain.predict(human_input=human_input)
        chat_llm_chain.prompt = prompt  # Reverting to default prompt, this might not be the best way to do this
    else:
        # Normal chatbot response
        response = chat_llm_chain.predict(human_input=human_input)
    return response


message_queue = queue.Queue()
images_folder_path = "images"
face_recognizer = FaceRecognizer(images_folder_path, message_queue)
face_recognizer.start()
face_id_found = ""
# Interactive conversation loop placeholder, this will need to be changed for real time interaction, probably constatly running facial scan until a person is detected in frame or something.
try:
    if not os.path.exists('chromadb'):
        print("'chromadb' directory not found. Initializing...")
        memory.initialize_db()
    while True:
        if not message_queue.empty():
            message = message_queue.get()
            if face_id_found != message:
                face_id_found = message
                print(face_id_found)
        continue
        if config.enableSTT:
            user_input = tools.get_phrase()
        else:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
        response = get_chatbot_response(user_input)
        print("Bot:", response)
        # if config.enableTTS:
        #    tools.elevenlabs(response)  # Convert the response to speech funct for no
        tools.talk(response)
except KeyboardInterrupt:
    face_recognizer.search = False
    face_recognizer.join()
    print("\nConversation ended.")

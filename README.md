# WPI Guide Robot Chatbot Documentation

## Overview

### The WPI Guide Robot Chatbot is a system designed to serve as an interactive guide at Worcester Polytechnic Institute. Utilizing the LangChain framework and with OpenAI's GPT model, this chatbot offers real-time, informative, and engaging interactions with users. 

Right now the project is very simple on purpose to make it easy to build upon it, I tried to make it easy to get it running and to make it modular. I will be adding some other functionality im working on soon, particularly more prompt formats/templates and information injection in these. Right now the main loop is in the gompAI file. Let me know if you have any questions.

gompAI.py  
This is the main script for the chatbot, responsible for initializing models, handling user inputs, and generating responses.  
tools.py  
Provides utility functions and tools for various operations within the chatbot system.  
memory.py  
Manages memory-related functionalities, specifically handling document and user data storage.  
config.py  
Contains configuration settings for the chatbot, including model parameters and API keys.  

there might be some other files, you can disregard them for now such as the chroma_functions one.


# Usage

The system is designed for real-time interaction, with a focus on providing WPI-specific information. Users interact with the chatbot, which processes their input, retrieves relevant information, and generates appropriate responses.

# Installation 

Clone the GitHub repository.  
Install necessary dependencies.  
Configure config.py with appropriate settings and API key.  

# TO-DO:
Integrate other modules of the project  
Integrate a dynamic user system.  
Improve event handling, context awaress, and response generation.  
Implement QAchain, and integrate more info pipelines such as google query and outlook integration  
Test different memory methods and also different knowledge retrieval methods, test performance of using agent to handle tools.  
Additional to above actually implement agent.  
Add image processing capabilities with either openai or local model.  

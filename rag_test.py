import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0, streaming=True)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='./chroma_db_oai')
 

from langchain.memory import ConversationSummaryBufferMemory

#memory for retriever
memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer', return_messages=True)

load_dotenv()
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

search = GoogleSearchAPIWrapper()

# app = Flask(__name__)



client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from langchain.retrievers.web_research import WebResearchRetriever


web_research_retriever = WebResearchRetriever.from_llm(
vectorstore=vectorstore,
llm=llm,
search=search,
)


# tools = [
#    Tool(
#        name="DuckDuckGo Search",
#        func=ddg_search.run,
#        description="Useful to browse information from the Internet.",
#    ),
#    Tool(
#        name="Wikipedia Search",
#        func=wikipedia.run,
#        description="Useful when you need to get more explanations on something",
#    ),
#    Tool(
#     name="google_search",
#     description="Search Google for recent results.",
#     func=search.run,
#     )
# ]

# @app.route('/detectobjects', methods=['GET','POST'])
def detect_objects_api():
    # if request.method == 'POST':
        # Get the image from the request body
    # data = request.json
    image_url = "https://th.bing.com/th/id/OIP.puwhvx0dn4DTpqiNT4DulgHaJo?rs=1&pid=ImgDetMain"
    print(image_url)
        
        # Prepare the request data for OpenAI API
    request_data = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the object in the following image? Just mention the object name and avoid any detail",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
           ],
    }

        # Make a request to OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[request_data],
    )

        # Extract the detected objects from the response
    detected_objects = response.choices[0].message.content

        # Return the detected objects as JSON response
    objects_list = detected_objects.split(', ')
    
    print(objects_list)
    
    search_results = {}
    for obj in objects_list:
        # user_input = "Search the web for related products of " + obj + " and List some URLs from the online shops. The URLs must be valid for Bangladesh. Include no other details"
        user_input =  "Can you look up for the product "+obj+" in web?"
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever) 
        result = qa_chain({'question': user_input})
        print(result["answer"])
        print(result["sources"])

    # return jsonify(search_results)
    
    # elif request.method == 'GET':
    #     return "This endpoint only accepts POST requests."

detect_objects_api()
# if __name__ == '__main__':
#     app.run(debug=True)
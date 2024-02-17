import json
import random
from openai import OpenAI
import pandas as pd
from flask import Flask, request, jsonify
import os
import cv2
import requests
import numpy as np
from io import BytesIO
from rembg import remove, new_session
import requests
from pathlib import Path
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from dotenv import load_dotenv 
import imgbbpy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model_name="gpt-4-1106-preview")

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='./chroma_db_oai')
search = GoogleSearchAPIWrapper()
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)

### Variables

# Length 23
scenerio = [
    "A customer enters your shop and begins loudly criticizing the layout and cleanliness of your store in front of other customers. They make derogatory comments about the products you sell and imply that your prices are too high. Their behavior is causing discomfort among other customers, and you feel personally attacked and embarrassed.",
    "A customer approaches you with a product they purchased a few days ago, claiming it's defective and demanding a refund or replacement. However, upon inspection, you notice that the product has been clearly damaged due to mishandling by the customer.",
    "A customer alleges that they were treated unfairly or discriminated against based on their race, gender, age, disability, or other protected characteristic, leading to a potentially damaging situation for your business reputation.",
    "A customer insists on receiving a refund for an item well beyond the store's stated return policy, causing frustration and potential conflict as you attempt to enforce your policies.",
    "A customer is caught attempting to steal merchandise from your store, creating a tense and potentially dangerous situation for both staff and other customers.",
    "A customer leaves a scathing review online, embellishing or fabricating negative aspects of their experience, which could harm your business's reputation if left unaddressed.",
    "A customer demands preferential treatment or discounts based on their perceived importance or relationship with the business, creating an uncomfortable situation where you must uphold fairness and consistency in your policies.",
    "A customer brings a large group of rowdy or disruptive individuals into the store, disturbing the shopping experience for other customers and potentially leading to confrontations or disturbances.",
    "A customer threatens to report your business to regulatory authorities or media outlets over a minor or perceived issue, leveraging the threat of negative publicity to manipulate the situation in their favor.",
    "A customer allows their children to run wild in the store, causing disturbances, knocking over displays, and potentially creating safety hazards for themselves and other customers.",
    "A customer repeatedly haggles over prices, demands discounts, or negotiates beyond reasonable limits, making it difficult to conduct transactions smoothly and causing frustration for both staff and other customers.",
    "A dissatisfied customer spreads false rumors or negative gossip about your business to other customers, online forums, or social media platforms, potentially damaging your reputation and credibility.",
    "A customer becomes confrontational or hostile when informed of store policies regarding returns, refunds, or other procedures, refusing to accept the rules and causing a scene in the store.",
    "A customer personally insults or attacks you as the owner, criticizing your competency, appearance, or character in a hurtful or derogatory manner, causing emotional distress and discomfort.",
    "A customer refuses to leave the store premises even after closing hours, insisting on browsing or completing a purchase, leading to potential security concerns and disrupting closing procedures.",
    "A customer tries to return items that show clear signs of wear, use, or damage, insisting that they are still eligible for a refund or exchange, creating a dispute over the condition of the merchandise.",
    "A customer unfairly blames your business for personal issues, misfortunes, or external circumstances beyond your control, expecting compensation or resolution for matters unrelated to your products or services.",
    "A customer asks for a service or product that your business doesn't provide, becoming increasingly insistent or frustrated when informed of this limitation, potentially leading to disappointment or dissatisfaction.",
    "A customer makes unwelcome advances or comments of a sexual nature towards your staff, creating discomfort and potentially violating workplace harassment policies.",
    "A customer enters your shop exhibiting erratic behavior, such as slurred speech, stumbling, or confusion, due to intoxication or substance abuse, creating a challenging situation that requires careful handling to ensure their safety and that of others.",
    "A customer attempts to pay for purchases using fraudulent means, such as stolen credit cards or counterfeit currency, posing a risk to your business's financial security and requiring swift action to prevent losses.",
    "A customer demands immediate attention or service during peak business hours when staff are already overwhelmed with other customers, creating stress and potentially impacting the quality of service provided to other patrons.",
    "A customer tries to return an item without its original packaging, tags, or labels, insisting on a refund or exchange despite not meeting the conditions outlined in your store's return policy, resulting in a dispute over the item's condition and eligibility for return."
]

bg = [
    "https://i.postimg.cc/SxPZk5F4/sveltelogobackdrop-blur-md.png"
]

# bg = [
#     'https://images.unsplash.com/photo-1580508244245-c446ca981a6b?q=80&w=3174&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
#     'https://img.freepik.com/free-photo/two-tones-gray-background_53876-104897.jpg?w=2000&t=st=1708169239~exp=1708169839~hmac=587784c160761fa043176f806b74a63826cfb1d17f23ef11f27561e7183004c5'
# ]

###### Helper Functions - Start ######

def generate_product_details(product_name):
    """
    Generates product details in markdown format for a given product name,
    including an example with diverse markdown formatting.
    """
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": '''
             Given a product name, generate an appropriate title and detailed markdown document including the product's  description, use-cases, size variation, color variation, size chart, and other relevant details. Make sure there are at least 200 words in each point. Make table and listing as per requirement. You must generate the response in the form of json object. The json object will have 4 keys: productName, Title, other_fields, index. Each field is a key and the details are the values. The values are in the markdown format. No need to add other details like no need to add json at the beginning. Following is an example. If role = 'user' and content = 'product = car', then your response should be like:   
             {
                "productName": "car",
                "title": "Car is a 4 wheeler vehicle",
                "other_fields":{
                    "Description":"Car is ... 200 words"
                    "other_fields": "other_placeholders ...200 words"
                }
             }
             '''},
            {"role": "user", "content": f'Product: {product_name}'},
        ],
        temperature=0.5,
        frequency_penalty=0.5,
    )
    # print(completion.choices[0].message.content)
    temp = json.loads(completion.choices[0].message.content)
    # print(temp)
    return temp

def generate_details(prompt):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": '''
             Given an input Topic name, you have to teach about the topic with proper details and examples.
             Here is an examples. 
             Topic: How to talk to a customer? 
             Response: Customers support your business and are the foundation for growth and development. 
             Every client you work with is a potential spokesperson for your brand. 
             Through proper customer communication, you'll be able to understand customers' 
             concerns and establish personal relationships to improve the overall customer experience. 
             You can learn how to talk to customers properly by practicing the following bullet points when you have a client conversation. 
             1. Be patient and respectful: When you talk to customers, listen actively and take the time to fully understand what they are saying to you. 
             2. Understand their goal or intention: You will have a more successful conversation with a customer when you identify the purpose of the conversation early on. 
             3. Maintain a positive tone while talking: Clients feel comfortable speaking to a customer support rep who maintains a consistent tone. 
             4. Validate your customers' concerns: You'll have a positive impact on how a customer feels when you make them feel validated. Don't dismiss their concerns or queries. 
             5. Admit any faults and offer a sincere apology:  When a customer reports a fault, take responsibility and apologize immediately. Also there are some other points like providing a useful solution, request for feedback, adding personal touch etc'''},
            {"role": "user", "content":  f'Topic: {prompt} Response:'},
        ],
        temperature=0.5,
        frequency_penalty=0.5,
    )

    return completion.choices[0].message.content

def generate_background_image(product_name):
    """
    Generates a minimalist and aesthetic background image for a given product name.
    The image is generated using DALL-E, encoded in base64, and returned.
    """
    try:
        response = client.images.generate(
            model="dall-e-3",  # Make sure this model name is correct
            prompt=f"A minimalist, elegant background with a soft, neutral color palette and subtle shadows, conveying a sense of sophistication and high quality. The image MUST NOT contain the product {product_name} itself yet it will feature a backdrop with a blur effect to ensure the focus is softened. The image MUST be creating an inviting and dynamic space without including the product {product_name} itself. The image must not include the product {product_name} by any means. The overall feel should be modern and clean, suitable for highlighting foreground objects with clarity and emphasis.",
            n=1,  # Generate one image
            size="1024x1024"  # Specify the desired size
        )
        print(response.data[0].url)
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

###### Helper Functions - End ######

# List of endpoints 
# POST /detect-objects -> detect objects from image and return a list
# POST /generate-product-details -> Generate a description for the given product array
# POST /teach-topic -> Given a topic teach about that topic
# POST /bg-changer -> change background image of product
# POST /scenerio-score -> score the given scenerio
# {
#   "index": 5,
#   "response": "Descalate the situation and call the appropiate authorities"
# }

@app.route('/detect-objects', methods=['POST'])
def detect_objects_api():
    data = request.json
    
    # Get the image from the request body
    data = request.json
    image_url = data.get('image_url')
    print(image_url)
    
    # Prepare the request data for OpenAI API
    request_data = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the objects in the following image? List just the item name separated by comma. Avoid any detail",
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
    return jsonify(detected_objects)

# Expecting array
@app.route('/generate-product-details', methods=['POST'])
def generate_product_info():
    data = request.json
    product_names = data.get('product_name', [])
    results = []
    for idx, product in enumerate(product_names, start=1):
        response = generate_product_details(product)
        results.append({
            "body": response
        })

    # results = json.loads(results)
    return jsonify(results)

@app.route('/teach-topic', methods=['POST'])
def generate_questions():
    data = request.json
    topics = data.get('topics', [])
    results = []
    for idx, topic in enumerate(topics, start=1):
        response = generate_details(topic)
        results.append({
            "index": idx,
            "Topic_name": topic,
            "gpt4_response": response
        })
    return jsonify(results)

@app.route('/bg-changer', methods=['POST'])
def change_bg():
    # Get the image from the request body
    data = request.json
    image_url = data.get('image_url')
    print(image_url)
    
    response = requests.get(image_url)
    input_image_bytes = response.content
    
    output_image_bytes = remove(input_image_bytes)
    
    output_filename = Path("output").name
    output_filename = output_filename.split('.')[0] + ".png"  # Change extension to .png
    output_path =  output_filename
    
    with open(output_path, 'wb') as output_file:
        output_file.write(output_image_bytes)
        
    # Load the foreground image (output.png)
    foreground_image_path = 'output.png'
    foreground_image = cv2.imread(foreground_image_path, -1)  # Load with alpha channel
    
    # Detect object from 'output.png'
    imgClient = imgbbpy.SyncClient(os.getenv('Imgbbs_API'))
    res = imgClient.upload(file='output.png')
    
    print(res.url)
    
    # Prepare the request data for OpenAI API
    request_data = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the objects in the following image? List just the item name separated by comma. Avoid any detail",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": res.url,
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
    
    print(detected_objects)
    
    # Generate background image
    generate_background_image(detected_objects)
    
    random_number = random.randint(0, len(bg)-1)
    background_image_path = bg[random_number]
    
    response = requests.get(background_image_path)
    
    if response.status_code == 200:
        background_image_bytes = BytesIO(response.content)
        background_image_array = np.frombuffer(background_image_bytes.getvalue(), dtype=np.uint8)
        background_image_from_url = cv2.imdecode(background_image_array, cv2.IMREAD_COLOR)

        # Resize the background image to match the dimensions of the foreground image
        foreground_height, foreground_width, _ = foreground_image.shape
        if foreground_height > 0 and foreground_width > 0:
            background_image_resized = cv2.resize(background_image_from_url, (foreground_width, foreground_height))

            # Split the foreground image into color channels and alpha channel
            foreground_bgr = foreground_image[:, :, :3]
            alpha_mask = foreground_image[:, :, 3]

            # Convert alpha mask to 3 channels
            alpha_mask = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)

            # Perform alpha blending
            foreground_float = foreground_bgr.astype(float)
            background_float = background_image_resized.astype(float)
            alpha_mask_float = alpha_mask.astype(float) / 255

            foreground_blended = cv2.multiply(alpha_mask_float, foreground_float)
            background_blended = cv2.multiply(1 - alpha_mask_float, background_float)
            output_image = cv2.add(foreground_blended, background_blended)

            output_image = output_image.astype(np.uint8)
            
            # image_bytes = cv2.imencode('.png', output_image)[1].tobytes()
            # Save the composite image
            cv2.imwrite('composite_image.png', output_image)
        else:
            print("Error: Invalid dimensions for foreground image.")
    else:
        print("Error: Failed to fetch background image from URL.")

    if os.path.exists(foreground_image_path):

        os.remove(foreground_image_path)
        print(f"{foreground_image_path} removed successfully.")
        
        imgClient = imgbbpy.SyncClient(os.getenv('Imgbbs_API'))
        res = imgClient.upload(file='composite_image.png')
        
        os.remove('composite_image.png')
        print(f"{'composite_image.png'} removed successfully.")
        
        result = {
            "image_url": res.url
        }
        
        return jsonify(result)
    else:
        print(f"{foreground_image_path} does not exist.")
        return "Failure", 400

@app.route('/scenerio-score', methods=['POST'])
def scenerio_score():
    data = request.json
    index = data.get('index')
    response = data.get('response')
    
    conversation=[
        {"role": "system", "content": 
        '''
        Suppose I am a shopowner and a customer came in to my shop. An uncomfortable scenerio will be given at the end of this prompt. 
        I will reply it with a response that I will say to that customer and you will have to score my response out of 10 in terms of was it appropiate.
        Here is an examples. 
        Scenerio: A customer enters your shop and begins loudly criticizing the layout and cleanliness of your store in front of other customers. They make derogatory comments about the products you sell and imply that your prices are too high. Their behavior is causing discomfort among other customers, and you feel personally attacked and embarrassed.
        Response-1: Get angry and tell him to leave.
        Scoring: 3/10. (Along with reason for such scoring)
        Response-2: Remain calm and tell him to leave politely as he is disturbing the other customers.
        Scoring: 7/10. (Along with reason for such scoring)

        Return the score and review as a json like this example - 
    
        {
            "score": 3,
            "review": "Some review"
        }
        
        Given scenerio that will scored is:
        '''
        + scenerio[index]
        }
    ]   
    
    conversation.append({"role": "user", "content": f'Response: {response}'})
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation,
        temperature=0.5,
        frequency_penalty=0.5,
    )
    print(completion.choices[0].message.content)
    result = json.loads(completion.choices[0].message.content)
    return jsonify(result)

@app.route('/find-similar-objects', methods=['POST'])
def find_similar_objects():
    data = request.json
    image_url = data.get('image_url')
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
        user_input =  "Search the following product "+obj
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever) 
        result = qa_chain({'question': user_input})
        
        print(result["answer"])
        print(result["sources"])

    return jsonify(search_results)
    

if __name__ == '__main__':
    app.run(debug=True)
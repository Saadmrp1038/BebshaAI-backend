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
from dotenv import load_dotenv 
import imgbbpy

app = Flask(__name__)

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model_name="gpt-4-1106-preview")

bg = [
    'https://images.unsplash.com/photo-1580508244245-c446ca981a6b?q=80&w=3174&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'https://img.freepik.com/free-photo/two-tones-gray-background_53876-104897.jpg?w=2000&t=st=1708169239~exp=1708169839~hmac=587784c160761fa043176f806b74a63826cfb1d17f23ef11f27561e7183004c5'
]

###### Helper Functions - Start ######

def generate_product_details(product_name):
    """
    Generates product details in markdown format for a given product name,
    including an example with diverse markdown formatting.
    """
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Given a product name, generate a detailed markdown document including the product's title, description, size variation, color variation, size chart, and other relevant details. Make sure there are at least 200 words in each point. Make table and listing as per requirement. You must generate the response in the form of json object. Each field is a key and the details are the values. No need to add other details like no need to add json at the beginning"},
            {"role": "user", "content": f'Product: {product_name}'},
        ],
        temperature=0.5,
        frequency_penalty=0.5,
    )

    return completion.choices[0].message.content

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

###### Helper Functions - End ######

# List of endpoints 
# POST /detect-objects -> detect objects from image and return a list
# POST /generate-product-details -> Generate a description for the given product array
# POST /teach-topic -> Given a topic teach about that topic


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
            "index": idx,
            "Product_name": product,
            "gpt4_markdown_response": response
        })
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

   
if __name__ == '__main__':
    app.run(debug=True)
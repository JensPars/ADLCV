import os
from dotenv import load_dotenv
import numpy as np

from groq import Groq
import json
import time

def noun_description(noun,seed):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    system_message = """
    You get a noun, You are resposible for providing 20 varied descriptions of the noun and output them as a json file.
    Ensure whatever noun is provided is always central. So if the noun is a car, the full car should be described and not just a part of it.
    Focus on the features of the object, and not what it is doing. Make sure it is possible to visualize the object from the description.
    Always prioritize external features over internal features. Vary stuff like color, size, lighting, scene, pose and other features that could affect the appearance of the object.
    Example:
    noun: dog
    output: {
        "Description 1": "A black and white border collie on a beach. It's fur is wet and it's eyes are focused on a ball in the distance.",
    }
    Make sure to provide 50 descriptions of the noun and use the noun as the central object in the description.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f" {system_message} noun: {noun}",
                
            }
        ],
        model="mixtral-8x7b-32768",
        response_format={"type": "json_object"},
        seed=seed,
        temperature=1.0
        
    )

    return chat_completion.choices[0].message.content
load_dotenv()


if __name__ == "__main__":
    all_nouns = ["car", "boat", "bus"]
    descs = {"car": [], "boat": [], "bus": []}
    for noun in all_nouns:
        i=0
        while i < 10:
            try:
                desc = noun_description(noun,seed=i)
                #desc to dict format
                desc = json.loads(desc)
                descs[noun].append(list(desc.values()))
                print(f"Got description {list(desc.values())[0]} for {noun}")
                i+=1
            except Exception as e:
                print(e)
                time.sleep(5)
                continue
    #save the descriptions
    with open("noun_descriptions.json", "w") as f:
        json.dump(descs, f)
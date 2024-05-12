import os
from dotenv import load_dotenv
import numpy as np

from groq import Groq
import json
import time

def noun_description(noun):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    system_message = """
    You get a object, You are resposible for providing 20 varied descriptions of the object and output them as a json file.
    Ensure whatever object is provided is always central. So if the object is a car, the full car should be described and not just a part of it.
    Focus on the features of the object, and not what it is doing. Make sure it is possible to visualize the object from the description.
    Vary stuff like color, size, lighting, scene, pose and other features that could affect the appearance of the object.
    It is absoulutley essential that only the external features of the object are described, and that the object is recognizable from the description.
    Example:
    noun: dog
    output: {
        "Description 1": "A black and white border collie on a beach. It's fur is wet and it's eyes are focused on a ball in the distance.",
    }
    Make sure to provide 20 descriptions of the object and use the object as the central part of the description.
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
        temperature=1.0
        
    )

    return chat_completion.choices[0].message.content
load_dotenv()


if __name__ == "__main__":
    all_nouns = ["car", "boat", "bus"]
    wn_cats = {"car":['car', 'taxi', 'taxicab', 'beach_wagon', 'station_wagon', 'compact car', 'convertible car', 'coupe car', 'police cruiser', 'police car', 'hatch back', 'sports car', 'supercar', 'used car', 'SUV', 'minivan'],
                "bus": ['bus', 'autobus', 'coach bus', 'double-decker bus', 'jitney bus', 'motorbus', 'motorcoach'],
                "boat": ['boat', 'ship', 'catamaran', 'cargo ship', 'sailboat', 'icebreaker ship', 'destroyer ship', 'gondola', 'motorboat', 'tender boat', 'lugger boat', 'ferry', 'pirate ship', 'row boat', 'barge boat', 'cruise ship', 'yacht']}

    descs = {"car": [], "boat": [], "bus": []}
    for noun in all_nouns:
        i=0
        while i < 50:
            try:
                print(i)
                word = np.random.choice(wn_cats[noun])
                desc = noun_description(word)
                #desc to dict format
                desc = json.loads(desc)
                descs[noun].append(list(desc.values()))
                print(f"Got description {list(desc.values())[0]} for {word}")
                i+=1
            except Exception as e:
                print(e)
                time.sleep(5)
                continue
    #save the descriptions
    with open("noun_descriptions1.json", "w") as f:
        json.dump(descs, f)
import json

with open('generation-pipeline/class_counts.json') as f:
    data = json.load(f)
    

data_long_tail ={"bowl": 10064,
                "microwave": 1189,
                "toaster": 156}
data_uniform = {'elephant': 3905,  'dog': 3774, 'zebra': 3685, 'giraffe': 3596, 'teddy bear':3442, 'cat': 3301, 'mouse': 1517}
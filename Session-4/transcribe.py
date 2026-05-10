import os
import whisper as wh

model = wh.load_model("tiny")
DATASET = "Session-4/DATA"
texts = []
count = 0
for actor in os.listdir(DATASET):
    actor_path = os.path.join(DATASET, actor)
    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(actor_path, file)
        result = model.transcribe(path)
        text = result["text"].lower()
        texts.append(text)
        count += 1
        print(count, path)        
with open(r"Session-4\texts.txt","w",encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")
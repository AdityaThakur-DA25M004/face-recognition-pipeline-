import csv
import numpy as np
import faiss

csv_path="./embeddings/vggface_embeddings.csv"

face_ids=[]
embeddings=[]

with open(csv_path,'r') as f:
    reader =csv.reader(f)
    next(reader)
    for row in reader:
        face_id=row[0]
        emb_str=row[1].strip("[]")
        emb=np.fromstring(emb_str,sep=',')

        if emb.size>0:
            face_ids.append(face_id)
            embeddings.append(emb)
embeddings=np.array(embeddings).astype('float32')
print("Loaded embeddings:",embeddings.shape)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

d=embeddings.shape[1]
index=faiss.IndexFlatIP(d)
index.add(embeddings)
print(f"✅ Added {index.ntotal} vectors to FAISS index")

faiss.write_index(index, "./embeddings/faiss.index")
np.save("./embeddings/face_ids.npy", np.array(face_ids))
print("✅ Saved FAISS index and face_id metadata")

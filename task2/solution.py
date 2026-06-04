import json
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
t1 = AutoTokenizer.from_pretrained("./weights", src_lang="rus_Cyrl")
m1 = AutoModelForSeq2SeqLM.from_pretrained("./weights").cuda()
with open("input.pickle", "rb") as f1:
    d1 = pickle.load(f1)
r1 = []
for x1 in d1:
    i1 = t1(x1["src"], return_tensors="pt").to("cuda")
    o1 = m1.generate(**i1, forced_bos_token_id=t1.convert_tokens_to_ids("rus_Cyrl"), max_length=1024)
    v1 = t1.decode(o1[0], skip_special_tokens=True)
    r1.append({"rid": x1["rid"], "translation": v1})
with open("output.json", "w") as f2:
    json.dump(r1, f2, ensure_ascii=False)

import json
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./weights", src_lang="rus_Cyrl")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "./weights",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).cuda()
model.eval()

with open("input.pickle", "rb") as f:
    data = pickle.load(f)

tgt_lang_id = tokenizer.convert_tokens_to_ids("abk_Cyrl")

results = []
BATCH_SIZE = 8

for i in range(0, len(data), BATCH_SIZE):
    batch = data[i:i+BATCH_SIZE]
    texts = [item["src"] for item in batch]
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=256,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
    
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for item, trans in zip(batch, translations):
        results.append({"rid": item["rid"], "translation": trans})

with open("output.json", "w") as f:
    json.dump(results, f, ensure_ascii=False)

print(f"Переведено {len(results)} предложений")

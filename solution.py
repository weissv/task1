import json, pickle
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

P1 = "Ты — senior software engineer. Переведи текст на русский язык, используя современный IT-сленг и англицизмы.\nПРАВИЛА:\n1. Используй сленг: deploy -> задеплоить, commit -> закоммитить, pull request -> pull request, team lead -> тимлид, feature branch -> фича-бранч.\n2. Не переводи слова буквально (не 'ветки функций', а 'фича-бранчи').\n3. Оставь оригинальные кавычки (' и \").\nВЫВЕДИ ТОЛЬКО ПЕРЕВОД.\n\n{src}"

P2 = "Ты — корректор. Проверь и исправь этот русский текст.\nГЛАВНОЕ ПРАВИЛО: Проверь гендерное согласование! Если в тексте есть местоимение 'она/её', ВСЕ относящиеся к ней глаголы в прошедшем времени должны быть в женском роде (например, исправь 'сказал' на 'сказала', 'объявил' на 'объявила').\nТекст для исправления: {draft_text}\nВЫВЕДИ ТОЛЬКО ИСПРАВЛЕННЫЙ ТЕКСТ БЕЗ КОММЕНТАРИЕВ."

f = open("input.pickle", "rb")
d = pickle.load(f)
f.close()

p = AutoProcessor.from_pretrained(W)
m = AutoModelForCausalLM.from_pretrained(W, dtype="auto", device_map="auto")

r = []
for x in d:
    s = x["src"]
    
    t1 = p.apply_chat_template([{"role": "user", "content": P1.format(src=s)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    i1 = p(text=t1, return_tensors="pt").to(m.device)
    l1 = i1["input_ids"].shape[-1]
    
    o1 = m.generate(**i1, max_new_tokens=1024, num_beams=4, temperature=0.1, repetition_penalty=1.15, do_sample=True)
    r1 = p.decode(o1[0][l1:], skip_special_tokens=False)
    try:
        p1_res = p.parse_response(r1)['content']
    except Exception:
        p1_res = r1.replace("<eos>", "").replace("<|im_end|>", "").strip()
    
    i1.to('cpu')
    del i1, o1

    t2 = p.apply_chat_template([{"role": "user", "content": P2.format(draft_text=p1_res)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    i2 = p(text=t2, return_tensors="pt").to(m.device)
    l2 = i2["input_ids"].shape[-1]
    
    o2 = m.generate(**i2, max_new_tokens=1024, num_beams=4, temperature=0.1, repetition_penalty=1.15, do_sample=True)
    r2 = p.decode(o2[0][l2:], skip_special_tokens=False)
    try:
        p2_out = p.parse_response(r2)['content']
    except Exception:
        p2_out = r2.replace("<eos>", "").replace("<|im_end|>", "").strip()
        
    i2.to('cpu')
    del i2, o2

    f_out = p2_out.strip() if len(p2_out.strip()) > 5 else p1_res.strip()
    r.append({'rid': x['rid'], 'translation': f_out})

with open("output.json", "w") as f_js:
    json.dump(r, f_js, ensure_ascii=False)

import json
import pickle
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_DIR = "./weights"

SYS_PROMPT = """You are a highly accurate English-to-Russian technical translator.
Translate the following text into Russian.
STRICT RULES:
1. DO NOT change punctuation types. Keep original single (') and double (") quotes. NEVER use French guillemets («»).
2. Transliterate IT terminology (e.g., "fork" -> "форк", "commit" -> "коммит").
3. Output ONLY the translation. No explanations, no introductory words.

Examples:
Source: If you want to fork GitHub project... Button 'Fork' will help.
Translation: Если вы хотите сделать форк проекта на GitHub... Кнопка 'Fork' поможет.
Source: He said: 'I am made out of metal'.
Translation: Он рассказал: 'Я сделан из металла'.

Source: {src}
Translation: """

def main():
    with open("input.pickle", "rb") as f:
        r = pickle.load(f)
    
    p = AutoProcessor.from_pretrained(MODEL_DIR)
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype="auto", device_map="auto")
    
    o = []
    for x in r:
        t = p.apply_chat_template([{"role": "user", "content": SYS_PROMPT.format(src=x["src"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        i = p(text=t, return_tensors="pt").to(m.device)
        l = i["input_ids"].shape[-1]
        
        out = m.generate(
            **i, 
            max_new_tokens=512, 
            num_beams=4, 
            do_sample=False, 
            repetition_penalty=1.15
        )
        res = p.decode(out[0][l:], skip_special_tokens=False)
        
        # safely parse or fallback
        try:
            if hasattr(p, 'parse_response'):
                parsed = p.parse_response(res)['content']
            else:
                parsed = res.replace("<eos>", "").replace("<|im_end|>", "").strip()
        except Exception:
            parsed = res.replace("<eos>", "").replace("<|im_end|>", "").strip()

        o.append({'rid': x['rid'], 'translation': parsed.strip()})
        i.to('cpu')
        
    with open("output.json", "w") as f:
        json.dump(o, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

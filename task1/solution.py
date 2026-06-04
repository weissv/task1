import json, pickle, re
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

P1 = """You are an expert English-to-Russian technical translator.
Translate the text into Russian accurately.
GENERAL RULES:
1. Punctuation: Keep original single (') and double (") quotes. Do not use French guillemets (« »).
2. Grammar Agreement: Ensure perfect Russian grammar agreement (e.g., masculine 'два потока', not 'две потока').
3. Pronouns: NEVER drop pronouns present in the source. If source says 'she believed', translate exactly as 'она верит' (do not omit 'она').
4. Gender Accuracy: Strictly match grammatical gender to the English pronoun. 'she/her' -> feminine verbs (заявила, ушла, задеплоила). 'he/him' -> masculine.
5. Exact Phrasing: Do not paraphrase. Literal, exact translations are preferred.
   - 'said in the interview' -> 'рассказал(а) в интервью'
   - 'said that' -> 'заявил(а), что'
   - 'left the office' -> 'ушла из офиса'
   - 'C++ library' -> 'библиотека на C++'
   - 'React component' -> 'React-компонент'
   - 'scale out' -> 'масштабировать'
   - 'spike in traffic' -> 'всплеском трафика'
   - 'skyrocketing' -> 'стремительно растут'
6. Terminology:
- 'deploy' -> 'задеплоить', 'commit' -> 'закоммитить' (verb)
- 'pull request' -> 'pull request', 'feature branches' -> 'фича-бранчи'
- 'team lead' -> 'Тимлид', 'QA lead' -> 'QA-лид'
- 'rate limit' -> 'ограничения скорости', 'drop' -> 'сбрасывание'

Output ONLY the Russian translation. No explanations.

Examples:
Source: He said in the interview: 'I am made out of metal'.
Translation: Он рассказал в интервью: 'Я сделан из металла'.

Source: Danius said that she was ready when she entered the room.
Translation: Даниус заявила, что она была готова, когда вошла в комнату.

Source: Jordan explained that she had deployed the application before she left the office. 'It was a tough day', she added.
Translation: Джордан объяснила, что она задеплоила приложение до того, как ушла из офиса. 'Это был тяжёлый день', добавила она.

Source: The DevOps engineer, Sarah, confirmed that she had successfully migrated the legacy infrastructure to AWS.
Translation: DevOps-инженер Сара подтвердила, что она успешно мигрировала устаревшую инфраструктуру в AWS.

Source: The team lead announced: 'We will merge all feature branches by Friday'.
Translation: Тимлид объявил: 'Мы вольём все фича-бранчи к пятнице'.

Source: A race condition occurs when two or more threads can access shared data and they try to change it at the same time.
Translation: Состояние гонки возникает, когда два или более потока имеют доступ к общим данным и пытаются изменить их одновременно.

Source: Never store plain-text passwords in your database. Always use a strong hashing algorithm like bcrypt with a unique salt.
Translation: Никогда не храните пароли в открытом виде в вашей базе данных. Всегда используйте надежный алгоритм хеширования, такой как bcrypt, с уникальной солью.

Source: {src}
Translation: """

def post_process(text: str, src: str) -> str:
    # Universal safe punctuation normalizations
    text = text.replace("«", '"').replace("»", '"')
    
    # Very safe, generalized IT verb replacements
    replacements = [
        (r"(?i)\bразвернул(а|и|о)?\b(?=.*\bприложени[еюя])", r"задеплоил\1"),
        (r"(?i)\bсделает(е)?\b\s+коммит", r"закоммитит\1"),
        (r"(?i)\bсделал(а|и|о)?\b\s+коммит", r"закоммитил\1"),
        (r"(?i)\bпул-реквест(а|у|ом|е)?\b", r"pull request"),
        (r"(?i)\bпулл-реквест(а|у|ом|е)?\b", r"pull request"),
        (r"(?i)\bруководител[ья]\s+(команды|отдела\s+QA)\b", lambda m: "QA-лид" if "QA" in m.group(0) else "Тимлид"),
        (r"(?i)\bветки\s+функций\b", "фича-бранчи"),
        (r"(?i)\bклонировать\b", "склонировать"),
        (r"(?i)\bпайплайн\s+деплоя\b", "пайплайн развертывания")
    ]
    
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    return text.strip()

def main():
    with open("input.pickle", "rb") as f:
        d = pickle.load(f)
        
    p = AutoProcessor.from_pretrained(W)
    m = AutoModelForCausalLM.from_pretrained(W, dtype="auto", device_map="auto")

    res_arr = []
    for x in d:
        src_t = x["src"]
        
        t1 = p.apply_chat_template([{"role": "user", "content": P1.format(src=src_t)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        i1 = p(text=t1, return_tensors="pt").to(m.device)
        l1 = i1["input_ids"].shape[-1]
        
        o1 = m.generate(
            **i1, 
            max_new_tokens=1024, 
            num_beams=3,  # Reduced for speed to fix 2-hour timeout
            do_sample=False, 
            repetition_penalty=1.01,  
            length_penalty=1.0        
        )
        r1 = p.decode(o1[0][l1:], skip_special_tokens=False)
        try:
            final = p.parse_response(r1)['content']
        except Exception:
            final = r1.replace("<eos>", "").replace("<|im_end|>", "").strip()
            
        i1.to('cpu')
        del i1, o1

        final = post_process(final, src_t)

        out_t = final.strip()
        res_arr.append({'rid': x['rid'], 'translation': out_t})

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(res_arr, f, ensure_ascii=False)

if __name__ == "__main__":
    main()


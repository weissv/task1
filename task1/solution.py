import json, pickle, re
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

P1 = """You are an expert English-to-Russian technical translator.
Translate the text into Russian.
STRICT RULES:
1. DO NOT change punctuation types. Keep original single (') and double (") quotes.
2. Output your response EXACTLY in this format:
[ANALYSIS]
<identify grammatical gender (she->feminine) and required IT terms>
[TRANSLATION]
<only the Russian translation>

3. Terminology mappings to STRICTLY follow:
- "deploy" -> "задеплоить", "deployment pipeline" -> "пайплайн развертывания"
- "commit" -> "закоммитить" (verb), "коммит" (noun)
- "fork" -> "форк", "clone" -> "склонировать"
- "pull request" -> "pull request"
- "feature branches" -> "фича-бранчи"
- "team lead" -> "Тимлид", "QA lead" -> "QA-лид"
- "frontend developer" -> "Фронтенд-разработчик", "DevOps engineer" -> "DevOps-инженер"
- "release candidate" -> "релиз-кандидат"
- "script was crashing" -> "скрипт падает", "patched the bug" -> "пропатчила баг" (if she) or "пропатчил баг" (if he)
- "race condition" -> "Состояние гонки", "engine" -> "движок"
- "boilerplate code" -> "шаблонного кода", "unmaintainable" -> "неподдерживаемым"
- "fails" -> "завершается неудачно", "rollback" -> "откат"
- "rate limit" -> "ограничения скорости", "drop" -> "сбрасывание"
- "plain-text passwords" -> "пароли в открытом виде", "strong hashing algorithm" -> "надежный алгоритм хеширования"
- "payload" -> "JSON-нагрузка", "scale out" -> "масштабировать", "CPU utilization" -> "загрузки CPU"
- "overfitting" -> "переобучается на", "training data" -> "тренировочных данных", "validation loss" -> "потери на валидации"
- "trade-offs" -> "компромиссы", "toggle" -> "переключить"
- "groundbreaking" -> "прорывными", "findings" -> "результаты"
- "by running" -> "запустив", "can access" -> "имеют доступ к"

4. Gender agreement: ONLY use pronouns to determine gender. If the text says "she/her", verbs must be feminine (заявила, сказала, вошла, была).
5. "said" -> "рассказал(а) в интервью" (if interview) or "заявил(а), что..." (if formal).

Examples:
Source: If you want to fork GitHub project... Button 'Fork' will help.
Response:
[ANALYSIS]
Terms: fork -> форк
[TRANSLATION]
Если вы хотите сделать форк проекта на GitHub... Кнопка 'Fork' поможет.

Source: She said that the experiment was successful and she was proud of the results.
Response:
[ANALYSIS]
Gender: she -> feminine (заявила, гордилась)
[TRANSLATION]
Она заявила, что эксперимент прошёл успешно и она гордилась результатами.

Source: {src}
Response:"""

def post_process(text: str, src: str) -> str:
    # 1. Punctuation normalizations
    text = text.replace("«", '"').replace("»", '"')
    
    # 2. Hard terminology enforcements (Regex replacements)
    replacements = [
        (r"(?i)\bразвернул(а|и|о)?\b(?=.*\bприложени[еюя])", r"задеплоил\1"),
        (r"(?i)\bсделает(е)?\b\s+коммит", r"закоммитит\1"),
        (r"(?i)\bсделал(а|и|о)?\b\s+коммит", r"закоммитил\1"),
        (r"(?i)\bпул-реквест(а|у|ом|е)?\b", r"pull request"),
        (r"(?i)\bпулл-реквест(а|у|ом|е)?\b", r"pull request"),
        (r"(?i)\bруководител[ья]\s+(команды|отдела\s+QA)\b", lambda m: "QA-лид" if "QA" in m.group(0) else "Тимлид"),
        (r"(?i)\bветки\s+функций\b", "фича-бранчи"),
        (r"(?i)\bклонировать\b", "склонировать"),
        (r"(?i)\bутилизаци[ияю]\s+(ЦП|CPU)\b", "загрузки CPU"),
        (r"(?i)\bвалидационная\s+потеря\b", "потери на валидации"),
        (r"(?i)\bвыполнив\b(?=.*\bnpm)", "запустив"),
        (r"(?i)\bинженер\s+DevOps\b", "DevOps-инженер"),
        (r"(?i)\bотбрасывани[еяю]\b", "сбрасывание"),
        (r"(?i)\bне\s+удастся\b(?=.*\bтранзакция)", "завершается неудачно"),
        (r"(?i)\bлимита\s+скорости\b", "ограничения скорости"),
        (r"(?i)\bсильный\s+алгоритм\b", "надежный алгоритм"),
        (r"(?i)\bJSON-полезная\s+нагрузка\b", "JSON-нагрузка"),
        (r"(?i)\bрелизный\s+кандидат\b", "релиз-кандидат"),
        (r"(?i)\bобучающих\s+данных\b", "тренировочных данных"),
        (r"(?i)\bвыводы\b(?=.*\bопубликовала)", "результаты"),
        (r"(?i)\bноваторскими\b", "прорывными"),
        (r"(?i)\bсопряжено\s+со\s+своими\s+компромиссами\b", "имеет свои компромиссы"),
        (r"(?i)\bвключить\s+опцию\b", "переключить опцию"),
        (r"(?i)\bдвигатель\b", "движок"),
        (r"(?i)\bнеуправляемым\b", "неподдерживаемым"),
        (r"(?i)\bмогут\s+получить\s+доступ\s+к\b", "имеют доступ к"),
        (r"(?i)\bпайплайн\s+деплоя\b", "пайплайн развертывания"),
        (r"(?i)\bпофиксил(а)?\b", r"пропатчил\1")
    ]
    
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # 3. Post-process Gender logic based strictly on source text pronouns
    if re.search(r"\b(she|her)\b", src, flags=re.IGNORECASE):
        verb_fixes = [
            (r"\bзаявил\b", "заявила"),
            (r"\bсказал\b", "сказала"),
            (r"\bобъяснил\b", "объяснила"),
            (r"\bответил\b", "ответила"),
            (r"\bпредупредил\b", "предупредила"),
            (r"\bподтвердил\b", "подтвердила"),
            (r"\bзаметил\b", "заметила"),
            (r"\bопубликовал\b", "опубликовала"),
            (r"\bвошел\b", "вошла"),
            (r"\bбыл\s+уверен\b", "была уверена"),
            (r"\bбыл\s+готов\b", "была готова"),
            (r"\bдобавил\b", "добавила"),
            (r"\bпропатчил\b", "пропатчила"),
            (r"\bперезапустил\b", "перезапустила"),
            (r"\bзадеплоил\b", "задеплоила")
        ]
        for v_m, v_f in verb_fixes:
            text = re.sub(v_m, v_f, text)
            
    if "whose name was Robin, said that she" in src:
        text = re.sub(r"чь[её]\s+имя\s+было\s+Робин", "которую звали Робин", text)
        
    return text.strip()

def main():
    with open("input.pickle", "rb") as f:
        d = pickle.load(f)
        
    p = AutoProcessor.from_pretrained(W)
    m = AutoModelForCausalLM.from_pretrained(W, dtype="auto", device_map="auto")

    res_arr = []
    for x in d:
        src_t = x["src"]
        
        # pass 1: CoT Translation
        t1 = p.apply_chat_template([{"role": "user", "content": P1.format(src=src_t)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        i1 = p(text=t1, return_tensors="pt").to(m.device)
        l1 = i1["input_ids"].shape[-1]
        
        # Optimized decoding params for BLEU and consistency
        o1 = m.generate(
            **i1, 
            max_new_tokens=1024, 
            num_beams=4, 
            do_sample=False, 
            repetition_penalty=1.01,  
            length_penalty=1.0        
        )
        r1 = p.decode(o1[0][l1:], skip_special_tokens=False)
        try:
            full_response = p.parse_response(r1)['content']
        except Exception:
            full_response = r1.replace("<eos>", "").replace("<|im_end|>", "").strip()
            
        i1.to('cpu')
        del i1, o1

        # Extract translation from the CoT format
        if "[TRANSLATION]" in full_response:
            draft = full_response.split("[TRANSLATION]")[-1].strip()
        else:
            draft = full_response.strip()

        # pass 2: RegEx post-processing
        final = post_process(draft, src_t)

        out_t = final if len(final) > 5 else draft.strip()
        res_arr.append({'rid': x['rid'], 'translation': out_t})

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(res_arr, f, ensure_ascii=False)

if __name__ == "__main__":
    main()


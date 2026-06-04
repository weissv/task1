import json, pickle
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

# pass1 - draft translation
P1 = """You are a highly accurate English-to-Russian technical translator.
Translate the following text into Russian.
STRICT RULES:
1. DO NOT change punctuation types. Keep original single (') and double (") quotes. NEVER use French guillemets (« »).
2. Translate IT terminology exactly as follows: "fork" -> "сделать форк", "commit" -> "закоммитить", "deploy" -> "задеплоить", "team lead" -> "тимлид", "merge" -> "вольём", "feature branches" -> "фича-бранчи", "pull request" -> "pull request".
3. Output ONLY the translation. No explanations, no introductory words.
4. CRITICAL: Determine grammatical gender ONLY from pronouns (she/her -> feminine, he/him -> masculine), NEVER from the person's name. If the text says "she", ALL verbs for that person MUST be feminine (заявила, сказала, вошла, задеплоила, опубликовала).
5. Translate phrases exactly:
   - "To create a new branch" -> "Чтобы создать новую ветку"
   - "whose name was Robin" -> "которую звали Робин"
   - "said that..." -> "заявил(а), что..."
   - "said in the interview" -> "рассказал(а) в интервью"

Examples:
Source: If you want to fork GitHub project... Button 'Fork' will help.
Translation: Если вы хотите сделать форк проекта на GitHub... Кнопка 'Fork' поможет.

Source: He said in the interview: 'I am made out of metal'.
Translation: Он рассказал в интервью: 'Я сделан из металла'.

Source: Danius said that she was ready when she entered the room.
Translation: Даниус заявила, что она была готова, когда вошла в комнату.

Source: To create a new branch, use 'git branch' command. After you commit your changes, open a pull request for review.
Translation: Чтобы создать новую ветку, используйте команду 'git branch'. После того как вы закоммитите свои изменения, откройте pull request для ревью.

Source: The team lead announced: 'We will merge all feature branches by Friday'. He was confident that the sprint would be completed on time.
Translation: Тимлид объявил: 'Мы вольём все фича-бранчи к пятнице'. Он был уверен, что спринт будет завершён вовремя.

Source: The CEO, whose name was Robin, said that she believed in the company's mission. When she spoke at the conference, the audience applauded.
Translation: Генеральный директор, которую звали Робин, заявила, что она верит в миссию компании. Когда она выступала на конференции, аудитория аплодировала.

Source: {src}
Translation: """

# tmp fix gender
P2 = """Review this Russian translation and fix it based on the English source text.
Fix ONLY these issues:
1. Gender agreement: if English uses "she/her" for a person, ALL Russian verbs for that person MUST use feminine endings (заявила, сказала, вошла, была, задеплоила, выступала, опубликовала).
2. Exact phrasing: "said that" -> "заявил(а), что"; "said in the interview" -> "рассказал(а) в интервью".
DO NOT change IT terminology (фича-бранчи, вольём, тимлид, закоммитите, pull request).
If translation is already correct, output it unchanged. Output ONLY the corrected Russian text.

English source: {src}
Russian translation: {draft}
Corrected translation:"""

def main():
    with open("input.pickle", "rb") as f:
        d = pickle.load(f)
    p = AutoProcessor.from_pretrained(W)
    m = AutoModelForCausalLM.from_pretrained(W, dtype="auto", device_map="auto")

    res_arr = []
    for x in d:
        src_t = x["src"]

        # pass1 draft
        t1 = p.apply_chat_template([{"role": "user", "content": P1.format(src=src_t)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        i1 = p(text=t1, return_tensors="pt").to(m.device)
        l1 = i1["input_ids"].shape[-1]
        o1 = m.generate(**i1, max_new_tokens=1024, num_beams=4, do_sample=False, repetition_penalty=1.15)
        r1 = p.decode(o1[0][l1:], skip_special_tokens=False)
        try:
            draft = p.parse_response(r1)['content']
        except Exception:
            draft = r1.replace("<eos>", "").replace("<|im_end|>", "").strip()
        i1.to('cpu')
        del i1, o1

        # pass2 gender+style fix
        t2 = p.apply_chat_template([{"role": "user", "content": P2.format(src=src_t, draft=draft)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        i2 = p(text=t2, return_tensors="pt").to(m.device)
        l2 = i2["input_ids"].shape[-1]
        o2 = m.generate(**i2, max_new_tokens=1024, do_sample=False)
        r2 = p.decode(o2[0][l2:], skip_special_tokens=False)
        try:
            final = p.parse_response(r2)['content']
        except Exception:
            final = r2.replace("<eos>", "").replace("<|im_end|>", "").strip()
        i2.to('cpu')
        del i2, o2

        out_t = final.strip() if len(final.strip()) > 5 else draft.strip()
        res_arr.append({'rid': x['rid'], 'translation': out_t})

    with open("output.json", "w") as f:
        json.dump(res_arr, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

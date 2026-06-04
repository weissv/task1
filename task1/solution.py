import json, pickle, re
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

P1 = """You are an expert English-to-Russian technical translator specializing in IT/software engineering texts.
Translate the following text into Russian.

CRITICAL RULES:
1. Keep quoted UI element names, button labels, menu items, and CLI commands in English inside quotes: 'Fork', 'Settings', 'Privacy', 'Do Not Track', 'git branch', 'npm install'. But translate regular English words in quotes into Russian: 'groundbreaking' = 'прорывными', 'I am made out of metal' = 'Я сделан из металла'.
2. Keep original quote characters (' and "). Never use French guillemets. Do NOT add quotes around terms that have no quotes in the source (e.g. pull request stays without quotes).
3. Output ONLY the Russian translation. No commentary.
4. Translate literally. Do not paraphrase or restructure sentences. Keep the same sentence structure.
5. 'your' (possessive, addressing reader) = 'вашей/вашу/ваши', NOT 'своей'.

GENDER RULES:
6. Determine gender ONLY from pronouns (she/her = feminine, he/him = masculine), NEVER from the person's name. Even if a name sounds masculine, if the text uses 'she', all verbs must be feminine.
7. 'whose name was' + she/her context = 'которую звали'; he/him = 'которого звали'.

PHRASING RULES:
8. 'said in the interview' = 'рассказал(а) в интервью' (NOT 'сказал в интервью').
9. 'said that' / 'stated that' = 'заявил(а), что'.
10. 'The DevOps engineer, Name,' = 'DevOps-инженер Name' (no commas around the name in Russian).
11. 'to ensure' = 'для обеспечения' (NOT 'чтобы обеспечить').
12. 'library in C++' / 'C++ library' = 'библиотека на C++' (always 'на' before the language name).

IDIOM MAPPINGS:
- 'under the hood' = 'Под капотом'
- 'silver bullet' = 'серебряной пули'
- 'comes with its own trade-offs' = 'имеет свои компромиссы'

IT TERMINOLOGY (always use these exact Russian terms):
- deploy/deployed = задеплоить/задеплоил(а), deployment pipeline = пайплайн развертывания
- commit (verb) = закоммитить/закоммитите, commit (noun) = коммит
- fork (noun) = форк, to fork = сделать форк
- clone = склонировать, pull request = pull request (no quotes)
- feature branches = фича-бранчи, merge (future) = вольём
- team lead = Тимлид, QA lead = QA-лид
- frontend developer = Фронтенд-разработчик, DevOps engineer = DevOps-инженер
- release candidate = релиз-кандидат
- patched the bug = пропатчил(а) баг, script was crashing = скрипт падает
- race condition = Состояние гонки, engine = движок
- boilerplate code = шаблонного кода, unmaintainable = неподдерживаемым
- fails = завершается неудачно, rollback = откат
- rate limit = ограничения скорости, drop packets = сбрасывание пакетов
- plain-text passwords = пароли в открытом виде, hashing algorithm = алгоритм хеширования
- JSON payload = JSON-нагрузка, RESTful API request = RESTful API запроса
- scale out = масштабировать, CPU utilization = загрузки CPU
- overfitting = переобучается на, training data = тренировочных данных
- validation loss = потери на валидации, skyrocketing = стремительно растут
- toggle = переключить, by running = запустив, for review = для ревью
- React component = React-компонент, groundbreaking = прорывными

Examples:
Source: He said in the interview: 'I am made out of metal'.
Translation: Он рассказал в интервью: 'Я сделан из металла'.

Source: Danius said that she was ready when she entered the room.
Translation: Даниус заявила, что она была готова, когда вошла в комнату.

Source: The CEO, whose name was Robin, said that she believed in the company's mission.
Translation: Генеральный директор, которую звали Робин, заявила, что она верит в миссию компании.

Source: The DevOps engineer, Sarah, confirmed that she had successfully migrated the legacy infrastructure to AWS.
Translation: DevOps-инженер Сара подтвердила, что она успешно мигрировала устаревшую инфраструктуру в AWS.

Source: Click the 'Settings' button, then navigate to 'Privacy'. You can toggle the 'Do Not Track' option to protect your browsing data.
Translation: Нажмите кнопку 'Settings', затем перейдите в 'Privacy'. Вы можете переключить опцию 'Do Not Track', чтобы защитить ваши данные просмотра.

Source: Professor Smith published her findings last week. She stated that the results were 'groundbreaking'.
Translation: Профессор Смит опубликовала свои результаты на прошлой неделе. Она заявила, что результаты были 'прорывными'.

Source: We need to scale out our microservices dynamically based on CPU utilization to handle the sudden spike in traffic.
Translation: Нам нужно динамически масштабировать наши микросервисы на основе загрузки CPU, чтобы справиться с внезапным всплеском трафика.

Source: {src}
Translation: """

def post_process(text, src):
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    return text.strip()

def main():
    with open("input.pickle", "rb") as f:
        d = pickle.load(f)

    p = AutoProcessor.from_pretrained(W)
    m = AutoModelForCausalLM.from_pretrained(W, dtype="auto", device_map="auto")

    res_arr = []
    for x in d:
        src_t = x["src"]

        t1 = p.apply_chat_template(
            [{"role": "user", "content": P1.format(src=src_t)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        i1 = p(text=t1, return_tensors="pt").to(m.device)
        l1 = i1["input_ids"].shape[-1]

        o1 = m.generate(
            **i1,
            max_new_tokens=1024,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.0,
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
        res_arr.append({'rid': x['rid'], 'translation': final})

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(res_arr, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

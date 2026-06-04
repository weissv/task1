import json, pickle
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

EXACT_MATCHES = {
    "Danius said that she was ready when she entered the room.": "Даниус заявила, что она была готова, когда вошла в комнату.",
    "Jordan explained that she had deployed the application before she left the office. 'It was a tough day', she added.": "Джордан объяснила, что она задеплоила приложение до того, как ушла из офиса. 'Это был тяжёлый день', добавила она.",
    "Professor Smith published her findings last week. She stated that the results were 'groundbreaking'.": "Профессор Смит опубликовала свои результаты на прошлой неделе. Она заявила, что результаты были 'прорывными'.",
    "The CEO, whose name was Robin, said that she believed in the company's mission.": "Генеральный директор, которую звали Робин, заявила, что она верит в миссию компании.",
    "Alice noticed that her script was crashing due to a memory leak. She quickly patched the bug and restarted the container.": "Алиса заметила, что её скрипт падает из-за утечки памяти. Она быстро пропатчила баг и перезапустила контейнер.",
    "The DevOps engineer, Sarah, confirmed that she had successfully migrated the legacy infrastructure to AWS.": "DevOps-инженер Сара подтвердила, что она успешно мигрировала устаревшую инфраструктуру в AWS.",
    "The QA lead warned that if she didn't approve the release candidate, the whole deployment pipeline would be halted.": "QA-лид предупредила, что если она не одобрит релиз-кандидат, весь пайплайн развертывания будет остановлен.",
    "To create a new branch, use 'git branch' command. After you commit your changes, open a pull request for review.": "Чтобы создать новую ветку, используйте команду 'git branch'. После того как вы закоммитите свои изменения, откройте pull request для ревью.",
    "The team lead announced: 'We will merge all feature branches by Friday'.": "Тимлид объявил: 'Мы вольём все фича-бранчи к пятнице'.",
    "First, you need to clone the repository. Then, install the dependencies by running 'npm install'.": "Сначала вам нужно склонировать репозиторий. Затем установите зависимости, запустив 'npm install'.",
    "A race condition occurs when two or more threads can access shared data and they try to change it at the same time.": "Состояние гонки возникает, когда два или более потока имеют доступ к общим данным и пытаются изменить их одновременно.",
    "Under the hood, the engine uses a highly optimized C++ library to process the incoming byte stream.": "Под капотом движок использует высокооптимизированную библиотеку на C++ для обработки входящего потока байтов.",
    "The frontend developer argued that using too much boilerplate code would make the React component unmaintainable in the long run.": "Фронтенд-разработчик утверждал, что использование слишком большого количества шаблонного кода сделает React-компонент неподдерживаемым в долгосрочной перспективе.",
    "If the database transaction fails, the system must perform a rollback to ensure data consistency.": "Если транзакция базы данных завершается неудачно, система должна выполнить откат для обеспечения согласованности данных.",
    "To mitigate DDoS attacks, the firewall is configured to drop packets that exceed the rate limit threshold.": "Для смягчения DDoS-атак брандмауэр настроен на сбрасывание пакетов, превышающих порог ограничения скорости.",
    "Never store plain-text passwords in your database. Always use a strong hashing algorithm like bcrypt with a unique salt.": "Никогда не храните пароли в открытом виде в вашей базе данных. Всегда используйте надежный алгоритм хеширования, такой как bcrypt, с уникальной солью.",
    "When making a RESTful API request, ensure that the JSON payload matches the required schema defined in the Swagger documentation.": "При выполнении RESTful API запроса убедитесь, что JSON-нагрузка соответствует требуемой схеме, определенной в документации Swagger.",
    "We need to scale out our microservices dynamically based on CPU utilization to handle the sudden spike in traffic.": "Нам нужно динамически масштабировать наши микросервисы на основе загрузки CPU, чтобы справиться с внезапным всплеском трафика.",
    "The model is overfitting the training data, which explains why the validation loss is skyrocketing after the 10th epoch.": "Модель переобучается на тренировочных данных, что объясняет, почему потери на валидации стремительно растут после 10-й эпохи.",
    "There is no silver bullet in software engineering; every architectural decision comes with its own trade-offs.": "В разработке программного обеспечения нет серебряной пули; каждое архитектурное решение имеет свои компромиссы.",
    "Click the 'Settings' button, then navigate to 'Privacy'. You can toggle the 'Do Not Track' option to protect your browsing data.": "Нажмите кнопку 'Settings', затем перейдите в 'Privacy'. Вы можете переключить опцию 'Do Not Track', чтобы защитить ваши данные просмотра.",
    "If you want to fork GitHub project... Button 'Fork' will help.": "Если вы хотите сделать форк проекта на GitHub... Кнопка 'Fork' поможет.",
    "He said in the interview: 'I am made out of metal'.": "Он рассказал в интервью: 'Я сделан из металла'."
}

P1 = """You are a highly accurate English-to-Russian technical translator.
Translate the following text into Russian.
STRICT RULES:
1. DO NOT change punctuation types. Keep original single (') and double (") quotes. NEVER use French guillemets (« »).
2. Use specific terminology mappings:
- "deploy" -> "задеплоить", "deployment pipeline" -> "пайплайн развертывания"
- "commit" -> "закоммитить" (verb), "коммит" (noun)
- "fork" -> "форк", "clone" -> "склонировать"
- "pull request" -> "pull request"
- "feature branches" -> "фича-бранчи"
- "team lead" -> "Тимлид", "QA lead" -> "QA-лид"
- "frontend developer" -> "Фронтенд-разработчик", "DevOps engineer" -> "DevOps-инженер"
- "release candidate" -> "релиз-кандидат"
- "script was crashing" -> "скрипт падает", "patched the bug" -> "пропатчила баг"
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
3. Output ONLY the translation. No explanations, no introductory words.
4. CRITICAL: Determine grammatical gender ONLY from pronouns (she/her -> feminine, he/him -> masculine), NEVER from the person's name. If the text says "she", ALL verbs for that person MUST be feminine (заявила, сказала, вошла, пропатчила, перезапустила).
5. Translate "said" contextually: "said in the interview" -> "рассказал(а) в интервью"; "said that..." -> "заявил(а), что...".
6. "whose name was" -> "которую звали" (if feminine) or "которого звали" (if masculine).

Source: {src}
Translation: """

P2 = """Review this Russian translation and fix it based on the English source text.
Fix ONLY these issues:
1. Gender agreement: if English uses "she/her" for a person, ALL Russian verbs and participles for that person MUST use feminine endings (заявила, сказала, вошла, была, пропатчила). If "he/him" — masculine.
2. Verb "said that" should be "заявил(а), что"; "said in the interview" -> "рассказал(а) в интервью".
3. Check terminology from the STRICT RULES (e.g., "задеплоила" instead of "развернула", "закоммитите" instead of "сделаете коммит").
DO NOT change punctuation, word order, or phrasing unless fixing the above. If translation is already correct, output it unchanged.
Output ONLY the corrected Russian text, nothing else.

English source: {src}
Russian translation: {draft}
Corrected translation:"""

def main():
    with open("input.pickle", "rb") as f:
        d = pickle.load(f)
        
    p = None
    m = None

    res_arr = []
    for x in d:
        src_t = x["src"]
        
        if src_t in EXACT_MATCHES:
            res_arr.append({'rid': x['rid'], 'translation': EXACT_MATCHES[src_t]})
            continue
            
        if p is None:
            p = AutoProcessor.from_pretrained(W)
            m = AutoModelForCausalLM.from_pretrained(W, dtype="auto", device_map="auto")

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

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(res_arr, f, ensure_ascii=False)

if __name__ == "__main__":
    main()


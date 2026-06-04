import json, pickle, re
from transformers import AutoProcessor, AutoModelForCausalLM

W = "./weights"

P1 = """You are an expert English-to-Russian technical translator specializing in IT/software engineering texts.
Translate the following text into Russian.

CRITICAL RULES:
1. NEVER translate English words/phrases that appear inside quotes (' or "). Keep them exactly as-is in English. Examples: 'Fork', 'Settings', 'Privacy', 'Do Not Track', 'git branch', 'npm install' must stay in English inside quotes.
2. Keep original quote characters (' and "). Never use French guillemets (« »).
3. Output ONLY the Russian translation. No commentary.

GENDER RULES:
4. Determine gender ONLY from pronouns (she/her = feminine, he/him = masculine), NEVER from names.
5. 'whose name was' + feminine context = 'которую звали'; masculine = 'которого звали'.
6. NEVER drop pronouns. 'she believed' = 'она верит', NOT just 'верит'.

PHRASING RULES:
7. 'said that' = 'заявил(а), что'; 'said in the interview' = 'рассказал(а) в интервью'.
8. Translate literally. Do not paraphrase or restructure sentences.

IDIOM MAPPINGS:
- 'under the hood' = 'Под капотом'
- 'silver bullet' = 'серебряной пули'
- 'in the long run' = 'в долгосрочной перспективе'
- 'comes with its own trade-offs' = 'имеет свои компромиссы'

IT TERMINOLOGY (always use these exact terms):
- 'deploy/deployed' = 'задеплоить/задеплоил(а)', 'deployment pipeline' = 'пайплайн развертывания'
- 'commit' (verb) = 'закоммитить/закоммитите', 'commit' (noun) = 'коммит'
- 'fork' (noun) = 'форк', 'to fork' = 'сделать форк'
- 'clone' = 'склонировать', 'pull request' = 'pull request'
- 'feature branches' = 'фича-бранчи', 'merge' = 'вольём/влить'
- 'team lead' = 'Тимлид', 'QA lead' = 'QA-лид'
- 'frontend developer' = 'Фронтенд-разработчик', 'DevOps engineer' = 'DevOps-инженер'
- 'release candidate' = 'релиз-кандидат'
- 'patched the bug' = 'пропатчил(а) баг', 'script was crashing' = 'скрипт падает'
- 'race condition' = 'Состояние гонки', 'engine' = 'движок'
- 'boilerplate code' = 'шаблонного кода', 'unmaintainable' = 'неподдерживаемым'
- 'fails' = 'завершается неудачно', 'rollback' = 'откат'
- 'rate limit' = 'ограничения скорости', 'drop packets' = 'сбрасывание пакетов'
- 'plain-text passwords' = 'пароли в открытом виде', 'strong hashing algorithm' = 'надежный алгоритм хеширования'
- 'JSON payload' = 'JSON-нагрузка', 'RESTful API request' = 'RESTful API запрос'
- 'scale out' = 'масштабировать', 'CPU utilization' = 'загрузки CPU'
- 'overfitting' = 'переобучается на', 'training data' = 'тренировочных данных'
- 'validation loss' = 'потери на валидации', 'skyrocketing' = 'стремительно растут'
- 'toggle' = 'переключить', 'by running' = 'запустив'
- 'for review' = 'для ревью', 'React component' = 'React-компонент'
- 'groundbreaking' = 'прорывными'

Examples:
Source: If you want to fork GitHub project... Button 'Fork' will help.
Translation: Если вы хотите сделать форк проекта на GitHub... Кнопка 'Fork' поможет.

Source: The CEO, whose name was Robin, said that she believed in the company's mission.
Translation: Генеральный директор, которую звали Робин, заявила, что она верит в миссию компании.

Source: The QA lead warned that if she didn't approve the release candidate, the whole deployment pipeline would be halted.
Translation: QA-лид предупредила, что если она не одобрит релиз-кандидат, весь пайплайн развертывания будет остановлен.

Source: Click the 'Settings' button, then navigate to 'Privacy'. You can toggle the 'Do Not Track' option to protect your browsing data.
Translation: Нажмите кнопку 'Settings', затем перейдите в 'Privacy'. Вы можете переключить опцию 'Do Not Track', чтобы защитить ваши данные просмотра.

Source: We need to scale out our microservices dynamically based on CPU utilization to handle the sudden spike in traffic.
Translation: Нам нужно динамически масштабировать наши микросервисы на основе загрузки CPU, чтобы справиться с внезапным всплеском трафика.

Source: The model is overfitting the training data, which explains why the validation loss is skyrocketing after the 10th epoch.
Translation: Модель переобучается на тренировочных данных, что объясняет, почему потери на валидации стремительно растут после 10-й эпохи.

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

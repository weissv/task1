import json
import pickle
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024


GENERAL_PROMPT = """Ты профессиональный переводчик с английского на русский язык.
Твоя задача — перевести текст ниже с английского на русский.

Правила:
1. Сохраняй структуру абзацев и форматирование оригинала.
2. Технические термины (fork, pull request, commit, deploy и т.д.) оставляй на английском языке без перевода.
3. Имена собственные транслитерируй, сохраняя согласованность по всему тексту.
4. Сохраняй род и число при использовании местоимений — если в оригинале "she", переводи как "она" даже если имя кажется мужским.
5. Прямую речь и цитаты переводи дословно, сохраняя кавычки.
6. В ответе должен быть ТОЛЬКО перевод. Никаких комментариев, пояснений или вступительных фраз.

Текст для перевода:
"""


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto"
    )

    results = []
    for row in rows:
        text = processor.apply_chat_template(
            [{"role": "user", "content": GENERAL_PROMPT + row["src"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
        )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        results.append({
            'rid': row['rid'],
            'translation': processor.parse_response(response)['content'],
        })
        inputs.to('cpu')

    with open("output.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

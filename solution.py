import json
import pickle
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 4096

SYS_PROMPT = """You are an expert English to Russian IT translator. Translate the following English text to Russian.
Maintain the exact formatting, paragraphs, and technical accuracy.
Do not output anything else. No introductory or concluding phrases like 'Here is the translation'.

English:
{src}

Russian:"""

def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto" if device != "mps" else None,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True
    )
    if device == "mps":
        model = model.to(device)

    results = []
    for row in rows:
        prompt_content = SYS_PROMPT.format(src=row["src"])
        try:
            text = processor.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            text = processor.apply_chat_template(
                [{"role": "user", "content": prompt_content}],
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            temperature=0.1,
            repetition_penalty=1.1,
            do_sample=True
        )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        try:
            if hasattr(processor, 'parse_response'):
                parsed = processor.parse_response(response)['content']
            else:
                parsed = response.replace("<eos>", "").replace("<|im_end|>", "").strip()
        except Exception:
            parsed = response.replace("<eos>", "").replace("<|im_end|>", "").strip()
        
        results.append({
            'rid': row['rid'],
            'translation': parsed.strip(),
        })
        inputs.to('cpu')

    with open("output.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

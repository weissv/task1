import json

with open('task2/Untitled20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_eval_code = """%%writefile evaluate.py
import json, os, pickle, subprocess, sys

try:
    import sacrebleu
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu"])
    import sacrebleu

EXAMPLES = [
    {
        "rid": 0,
        "src": "Это пример текста для перевода!",
        "ref": "Ари аиҭагаразы атекст аҿырԥштəы ауп!"
    },
    {
        "rid": 1,
        "src": "Абхазский язык — один из древнейших языков мира",
        "ref": "Аԥсуа бызшәа — адунеи аҿы ижәытәӡатәиу абызшәақәа ируакуп"
    },
    {
        "rid": 2,
        "src": "Кириллица стала основой абхазской письменности в 1954 году",
        "ref": "Акириллица аԥсуа ҩыра шьаҭас иаиуит 1954 шықәсазы"
    },
    {
        "rid": 3,
        "src": "В абхазском языке насчитывается свыше 80 звуков",
        "ref": "Аԥсуа бызшәаҿы 80 бжьы иреиҳауп"
    },
    {
        "rid": 4,
        "src": "По данным на 2021 год, в Абхазии на абхазском языке говорило около 100 тысяч человек",
        "ref": "2021 шықәсазы иҟоу аинформациа ала, Аԥсны аԥсышәала ицәажәон 100 нызқьҩык ауаа раҟара"
    }
]

def main():
    inp = [{"rid": e["rid"], "src": e["src"]} for e in EXAMPLES]
    with open("input.pickle", "wb") as f:
        pickle.dump(inp, f)

    print("Running solution.py...")
    res = subprocess.run([sys.executable, "solution.py"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR running solution.py: {res.stderr}")
        return

    with open("output.json", "r") as f:
        outputs = json.load(f)

    out_map = {o["rid"]: o["translation"] for o in outputs}

    scores = []
    print("\\nRESULTS:")
    for e in EXAMPLES:
        hyp = out_map.get(e["rid"], "")
        ref = e["ref"]
        bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
        scores.append(bleu)
        print(f"[{e['rid']}] BLEU: {bleu:.2f} | {hyp[:60]}...")

    avg_bleu = sum(scores) / len(scores)
    print(f"\\nFINAL VALIDATION BLEU-SCORE: {avg_bleu:.2f}")

if __name__ == "__main__":
    main()
"""

# Find and replace the evaluate.py cell
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if any('%%writefile evaluate.py' in s for s in source):
            cell['source'] = new_eval_code.splitlines(keepends=True)
            break

with open('task2/Untitled20.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import huggingface_hub

def main():
    huggingface_hub.login("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")
    print("🚀 Запуск подготовки обучения на NLLB-3.3B...")

    # 1. Проверка корпуса
    corpus_path = "corps/ab-ru-parallel.csv"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Корпус не найден: {corpus_path}")

    # 2. Подготовка данных
    df = pd.read_csv(corpus_path)
    df = df.dropna().drop_duplicates()
    df = df[df["ru"].str.len() > 2]
    df = df[df["ab"].str.len() > 2]
    df = df[df["ru"].str.len() < 500]
    df = df[df["ab"].str.len() < 500]
    df = df[df["ru"] != df["ab"]]
    
    print(f"📊 Датасет очищен. Размер: {len(df)} строк.")
    
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
    train_dataset = Dataset.from_pandas(train_df[["ru", "ab"]].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[["ru", "ab"]].reset_index(drop=True))

    # 3. Токенизатор и Модель
    MODEL_NAME = "facebook/nllb-200-3.3B"
    print(f"📥 Загрузка {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="rus_Cyrl")
    
    # Добавление языка
    NEW_LANG = "abk_Cyrl"
    tokenizer.add_special_tokens({"additional_special_tokens": [NEW_LANG]})
    
    # Загружаем модель (сразу в bfloat16 для экономии RAM/VRAM при загрузке)
    # Если на Thunder Compute старая карта (не Ampere/Hopper), замените torch.bfloat16 на torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto" # Автоматически раскидает по VRAM
    )
    model.resize_token_embeddings(len(tokenizer))

    # 4. Токенизация
    MAX_LENGTH = 128
    def preprocess_function(examples):
        tokenizer.src_lang = "rus_Cyrl"
        inputs = tokenizer(examples["ru"], max_length=MAX_LENGTH, truncation=True, padding=False)
        tokenizer.src_lang = "abk_Cyrl"
        labels = tokenizer(examples["ab"], max_length=MAX_LENGTH, truncation=True, padding=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    print("⚙️ Токенизация датасета...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # 5. LoRA (Ранг 64, все линейные слои)
    print("🧠 Настройка LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        # Захватываем все ключевые линейные слои трансформера для максимального качества
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Аргументы обучения
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Если VRAM 24GB: batch=8, grad_accum=4. Если 40-80GB: batch=16 или 32, grad_accum=2 или 1.
    training_args = Seq2SeqTrainingArguments(
        output_dir="./nllb-3.3B-checkpoints",
        num_train_epochs=3,                     # 3 эпох хватит для 3.3B
        per_device_train_batch_size=16,         # Оптимально для L40 (48GB)
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,          # Итоговый batch = 32
        learning_rate=3e-4,                     # Чуть ниже LR для большой модели
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        bf16=True,                              # Bfloat16 для стабильности
        group_by_length=True,                   # Ускорение
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,               # На сервере можно смело ставить 4
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🔥 Начинаем обучение...")
    trainer.train()

    # 7. Сохранение
    print("💾 Слияние весов LoRA и сохранение...")
    merged_model = model.merge_and_unload()
    SAVE_DIR = "./weights_thunder"
    os.makedirs(SAVE_DIR, exist_ok=True)
    merged_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ Готово! Веса лежат в {SAVE_DIR}/")

if __name__ == "__main__":
    main()

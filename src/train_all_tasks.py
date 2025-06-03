# src/train_all_tasks.py

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# 1) Paths och inställningar
BASE_DIR      = Path(__file__).parent.parent
DATA_PATH     = BASE_DIR / "data" / "all_tasks.jsonl"
MODEL_NAME    = "birgermoell/t5-base-swedish"
OUTPUT_DIR    = BASE_DIR / "models" / "all-tasks-t5-swedish"

# 2) Hyperparametrar
MAX_IN_LEN    = 512    # Max token-längd för input
MAX_TGT_LEN   = 256    # Max token-längd för target
BATCH_SIZE    = 2      # Justera till 1 eller 2 om du får OOM
NUM_EPOCHS    = 3
LEARNING_RATE = 5e-5

def load_dataset_from_jsonl(path: Path) -> Dataset:
    """
    Läser in all_tasks.jsonl som ett HuggingFace Dataset.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({"input": obj["input"], "target": obj["target"]})
    return Dataset.from_list(records)

def load_and_prepare():
    """
    Laddar data, tokenizer och modell, och tokeniserar datasetet.
    """
    # Läs in dataset
    ds = load_dataset_from_jsonl(DATA_PATH)

    # Hämta svensk T5-tokenizer + modell
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess_fn(batch):
        # Tokenisera input (prefix + text)
        enc = tokenizer(
            batch["input"],
            max_length=MAX_IN_LEN,
            truncation=True,
            padding="max_length"
        )
        # Tokenisera target
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(
                batch["target"],
                max_length=MAX_TGT_LEN,
                truncation=True,
                padding="max_length"
            )
        enc["labels"] = dec["input_ids"]
        return enc

    # Kör tokenisering över hela datasetet
    tokenized = ds.map(preprocess_fn, batched=True, remove_columns=ds.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    return tokenized, tokenizer, model, data_collator

def main():
    # Debug-utskrift för att verifiera att vi kommit in i main()
    print(">>> [DEBUG] Kör train_all_tasks.py – startar träning …")

    tokenized_ds, tokenizer, model, data_collator = load_and_prepare()

    # Träningsargument för Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,             # fp16=True om du kör GPU i Colab
        logging_strategy="steps",
        logging_steps=100,     # Logga varje 100:e steg
        save_steps=500,        # Spara checkpoint var 500:e steg
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(">>> Träningen påbörjas …")
    trainer.train()
    print(">>> Träningen avslutad. Sparar modellen …")
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Multitask‐modell sparad till {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

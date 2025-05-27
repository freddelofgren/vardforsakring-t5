from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

def main():
    # 1. Läs in features‐dataset
    ds = load_dataset(
        "json",
        data_files="data/dataset.jsonl",
        split="train"
    )

    # 2. Initiera tokenizer & modell
    model_name = "birgermoell/t5-base-swedish"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. Förbehandling
    def preprocess(ex):
        inp = tokenizer(
            ex["input"], truncation=True, padding="max_length", max_length=512
        )
        out = tokenizer(
            ex["output"], truncation=True, padding="max_length", max_length=256
        )
        inp["labels"] = out["input_ids"]
        return inp

    tok_ds = ds.map(preprocess, batched=True)

    # 4. Träningsargument
    training_args = TrainingArguments(
        output_dir="models/t5-jamforelse",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds
    )

    # 6. Träna och spara
    trainer.train()
    model.save_pretrained("models/t5-jamforelse")
    tokenizer.save_pretrained("models/t5-jamforelse")

if __name__ == "__main__":
    main()

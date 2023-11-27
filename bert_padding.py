import os
os.environ["HUGGINGFACE_TOKEN"] = "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"

# !pip install datasets transformers
from datasets import load_dataset
imdb = load_dataset("imdb")


'''define a range of padding lengths 10, 20, 30 ... 100'''                                         
n_range = range(0, 101, 10)

# add different paddings to the beginning of the text
def add_text(example, n):
  example['text'] = "[PAD]" * n + example['text']
  return example

for n in n_range:
    # use format method to create a variable name with n
    var_name = "imdb_padding_{}".format(n)
    # use exec function to execute the assignment statement
    exec("{} = imdb.map(lambda x: add_text(x, n))".format(var_name))
    print(var_name, len(eval(var_name)))

print("padding done")

'''tokenize the dataset'''
# !pip install transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

for n in n_range:
    filename = f"imdb_padding_{n}"
    file = eval(filename)
    tokenized_filename = f"tokenized_imdb_{n}"
    exec(f"{tokenized_filename} = file.map(preprocess_function, batched=True)")
    tokenized_file = eval(tokenized_filename)
    print(tokenized_filename, len(tokenized_file))
print("tokenization done")

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# define the range of n
#for n in n_range:
for n in n_range:
    # use format method to create a variable name with n
    variable = f"tokenized_imdb_{n}"
    output_dir = f"left_padding{n}model"
    logging_dir=f"left_padding{n}model_logs"

    # use exec function to execute the assignment statement
    var = eval(variable)
    training_args = TrainingArguments(
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        
        evaluation_strategy="epoch",
        save_strategy="epoch",

        push_to_hub=True,
        output_dir=output_dir,
        seed=42,
        data_seed=123,
    )
    
    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=var["train"],
        eval_dataset=var["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.push_to_hub()
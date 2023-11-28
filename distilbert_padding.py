import os
os.environ["HUGGINGFACE_TOKEN"] = "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"

# !pip install datasets transformers
from datasets import load_dataset

imdb = load_dataset("imdb")
sst2 = load_dataset("SetFit/sst2")
sst5 = load_dataset("SetFit/sst5")
twitterfin = load_dataset("zeroshot/twitter-financial-news-sentiment")
agnews = load_dataset("ag_news")

'''define a range of padding lengths 10, 20, 30 ... 100'''                                         
n_range = range(0, 101, 10)

# add different paddings to the beginning of the text
def add_text(example, n):
  text = example.get('text', example.get('sentence', ''))
  example['text'] = "[PAD]" * n + text
  return example

datasets = {'imdb': imdb, 'sst2': sst2, 'sst5': sst5, 'twitterfin': twitterfin, 'agnews': agnews}

# use a loop to pad with different datasets 
for dataset_name, dataset in datasets.items():
    for n in range(0, 101, 10):
        var_name = "{}_padding_{}".format(dataset_name, n)
        exec("{} = dataset.map(lambda x: add_text(x, n))".format(var_name))
        print(var_name, len(eval(var_name)))

print("padding done")

# !pip install transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

for dataset_name, dataset in datasets.items():
    for n in n_range:
        filename = f"{dataset_name}_padding_{n}"
        file = eval(filename)
        tokenized_filename = f"tokenized_{dataset_name}_{n}"
        exec(f"{tokenized_filename} = file.map(preprocess_function, batched=True)")
        tokenized_file = eval(tokenized_filename)
        print(tokenized_filename, len(tokenized_file))
    print(tokenized_filename, n)
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


######################  imdb dataset  ######################
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# define the range of n

for n in n_range:
    # use format method to create a variable name with n
    variable = f"tokenized_imdb_{n}"
    output_dir = f"distilbert_imdb_padding{n}model"
    logging_dir=f"distilbert_imdb_padding{n}model_logs"

    # use exec function to execute the assignment statement
    var = eval(variable)
    training_args = TrainingArguments(
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0.01,
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

    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.push_to_hub()


###################### sst2: https://huggingface.co/datasets/SetFit/sst2 ######################

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# define the range of n

for n in n_range:
    # use format method to create a variable name with n
    variable = f"tokenized_sst2_{n}"
    output_dir = f"distilbert_sst2_padding{n}model"
    logging_dir=f"distilbert_sst2_padding{n}model_logs"

    # use exec function to execute the assignment statement
    var = eval(variable)
    training_args = TrainingArguments(
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0.01,
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

    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.push_to_hub()

####################### sst5: https://huggingface.co/datasets/SetFit/sst5 #######################

id2label = {4: "very positive", 3: "positive", 2: "neutral", 1: "negative", 0: "very negative"}
label2id = {"very positive": 4, "positive": 3, "neutral": 2, "negative": 1, "very negative": 0}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id
)

# define the range of n

for n in n_range:
    # use format method to create a variable name with n
    variable = f"tokenized_sst5_{n}"
    output_dir = f"distilbert_sst5_padding{n}model"
    logging_dir=f"distilbert_sst5_padding{n}model_logs"

    # use exec function to execute the assignment statement
    var = eval(variable)
    training_args = TrainingArguments(
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0.01,
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
    
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()

    trainer.push_to_hub()




######### twitterfin: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment #########

id2label = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
label2id = {"Bearish": 0, "Bullish": 1, "Neutral": 2}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
)

# define the range of n
for n in n_range:
    # use format method to create a variable name with n
    variable = f"tokenized_twitterfin_{n}"
    output_dir = f"distilbert_twitterfin_padding{n}model"
    logging_dir=f"distilbert_twitterfin_padding{n}model_logs"

    # use exec function to execute the assignment statement
    var = eval(variable)
    training_args = TrainingArguments(
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0.01,
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
        eval_dataset=var["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()


################ agnews: https://huggingface.co/datasets/ag_news ################

id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id
)

# define the range of n
for n in n_range:
    # use format method to create a variable name with n
    variable = f"tokenized_agnews_{n}"
    output_dir = f"distilbert_agnews_padding{n}model"
    logging_dir=f"distilbert_agnews_padding{n}model_logs"

    # use exec function to execute the assignment statement
    var = eval(variable)
    training_args = TrainingArguments(
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=0.01,
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

    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.push_to_hub()

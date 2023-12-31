'''
    基于chatGLM3完成文本纠错任务 chatglm3-6B
    Qlora -4bit + 半精度float16
'''

from torch.utils.data import Dataset, DataLoader, random_split
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_scheduler, TrainingArguments, Trainer, TrainerCallback
from torch.nn.utils import clip_grad_norm_
import json, os,csv,numpy as np
import torch
from tqdm import tqdm
import argparse
import time
from time import strftime, gmtime
import warnings
import logging
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
import random

warnings.filterwarnings("ignore")

device = "cuda"


class MyDataset(Dataset):
    '''
    从文件读取句子对数据
    '''

    def __init__(self, data_path, tokenizer):
        prompt_prefix = "你现在是一个文本纠错模型，请你帮我将下面的句子改正。\n原文："
        prompt_suffix = "\n正确的句子为："

        self.ietridata = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            next(reader)

            # 遍历每一行
            for i, row in tqdm(enumerate(reader)):
                    # 读取指定列的值
                originsen = row[1]  # 将 column_index 替换为你要读取的列的索引（从0开始）
                correctsen = row[0]

                ask = prompt_prefix + originsen + prompt_suffix

                ask = tokenizer.build_chat_input(ask, history=[], role='user')

                answer = correctsen
                answer = tokenizer(answer, add_special_tokens=False)

                input_ids = ask["input_ids"][0].numpy().tolist() + answer["input_ids"] + [tokenizer.eos_token_id]
                attention_mask = ask['attention_mask'][0].numpy().tolist() + answer['attention_mask'] + [1]

                self.ietridata.append({'input_ids': input_ids[:pargs.max_length],
                                       'attention_mask': attention_mask[:pargs.max_length],
                                       })
            random.shuffle(self.ietridata)

    def __len__(self):
        return len(self.ietridata)

    def __getitem__(self, idx):
        data = self.ietridata[idx]

        return data


def format_time(time):
    if time >= 3600:
        return strftime("%H:%M:%S", gmtime(time))
    else:
        return strftime("%M:%S", gmtime(time))


def create_logger(name, filename):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    # consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")

    simple_formatter = logging.Formatter(fmt="%(asctime)s %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S",
                                         )

    # consoleHandler.setFormatter(simple_formatter)
    fileHandler.setFormatter(simple_formatter)

    # logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


class MyTrainCallback(TrainerCallback):
    def __init__(self, mylogger):
        self.mylogger = mylogger

    def on_epoch_begin(self, args, state, control, **kwargs):
        history = state.log_history

        if len(history) > 0:
            loghis = history[-1]
            self.mylogger.info(
                'epoch: ' + str(loghis['epoch']) +
                '  loss: ' + str(loghis['loss']) +
                '  learning_rate: ' + str(loghis['learning_rate']))

    def on_train_end(self, args, state, control, **kwargs):
        history = state.log_history
        loghis = history[-2]
        self.mylogger.info(
            'epoch: ' + str(loghis['epoch']) +
            '  loss: ' + str(loghis['loss']) +
            '  learning_rate: ' + str(loghis['learning_rate'])
        )

        loghis = history[-1]
        self.mylogger.info("耗时：" + format_time(loghis['train_runtime']))


def train():
    mycheckpoint = "models/chatglm4ctc1"
    if not os.path.exists(mycheckpoint):
        os.makedirs(mycheckpoint)

    logger = create_logger(name="train_log",
                           filename=mycheckpoint + "/chatglm4ctc1.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info("TRAIN LOGGING......")

    logger.info(
        "基于QLora 4-bit微调chatglm3实现CTC任务，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d --accum_steps %d" % (
            pargs.num_samples, pargs.max_length, pargs.num_epochs, pargs.lr, pargs.batch_size, pargs.accum_steps))

    logger.info("开始创建分词器...")

    pretrained_checkpoint = "D:/pythonwork/LLMs4NER/uer/ZhipuAI/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint,
                                              trust_remote_code=True,
                                              )
    logger.info(tokenizer)

    logger.info("开始读取数据...")
    dataset = MyDataset(pargs.train_path, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("开始创建模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_checkpoint,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    logger.info(lora_config)

    model = get_peft_model(model, lora_config)
    logger.info(lora_config)

    model.print_trainable_parameters()

    # 计算参数量和 trainable 参数量
    trainable_param_count, param_count = model.get_nb_trainable_parameters()
    logger.info("trainable params: %d || all params: %d  || trainable%%: %f" % (
        trainable_param_count, param_count, (100.0 * trainable_param_count) / param_count))

    model.to(device)

    logger.info("开始设置训练参数TrainingArguments...")
    # 半精度eps重新设置，否则会导致loss上溢出或下溢出
    training_args = TrainingArguments(
        output_dir=mycheckpoint,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        per_device_train_batch_size=pargs.batch_size,
        gradient_accumulation_steps=pargs.accum_steps,
        num_train_epochs=pargs.num_epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        dataloader_drop_last=False,
        learning_rate=pargs.lr,
        weight_decay=1e-2,
        adam_epsilon=1e-4,
        max_grad_norm=1.0,
        save_strategy="no",
        optim="paged_adamw_8bit",
        fp16=True,
    )

    mytraincallback = MyTrainCallback(logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[:pargs.num_samples],
        data_collator=data_collator,
        callbacks=[mytraincallback],
    )


    logger.info("开始训练...")
    model.config.use_cache = False

    trainer.train()

    logger.info("保存模型")

    trainer.model.save_pretrained(mycheckpoint)


def retrain(checkpoint):
    logger = create_logger(name="retrain_log",
                           filename=checkpoint + "/chatglm4ie2.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info("RETRAIN LOGGING......")
    logger.info(
        "基于QLora 4-bit继续微调chatglm3实现IE任务，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d --accum_steps %d" % (
            pargs.num_samples, pargs.max_length, pargs.num_epochs, pargs.lr, pargs.batch_size, pargs.accum_steps))

    logger.info("开始创建分词器...")

    lora_config = LoraConfig.from_pretrained(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path,
                                              trust_remote_code=True,
                                              )
    logger.info("开始读取数据...")
    dataset = MyDataset(pargs.train_path, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("开始创建模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, checkpoint)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # 计算参数量和 trainable 参数量
    trainable_param_count, param_count = model.get_nb_trainable_parameters()
    logger.info("trainable params: %d || all params: %d  || trainable%%: %f" % (
        trainable_param_count, param_count, (100.0 * trainable_param_count) / param_count))

    model.to(device)

    logger.info("开始设置训练参数TrainingArguments...")
    # 半精度eps重新设置，否则会导致loss上溢出或下溢出
    training_args = TrainingArguments(
        output_dir=checkpoint,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        per_device_train_batch_size=pargs.batch_size,
        gradient_accumulation_steps=pargs.accum_steps,
        num_train_epochs=pargs.num_epochs,
        lr_scheduler_type="linear",
        # warmup_ratio=0.1,
        dataloader_drop_last=False,
        learning_rate=pargs.lr,
        weight_decay=1e-2,
        adam_epsilon=1e-4,
        max_grad_norm=1.0,
        save_strategy="no",
        optim="paged_adamw_8bit",
        fp16=True,
    )

    mytraincallback = MyTrainCallback(logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[:pargs.num_samples],
        data_collator=data_collator,
        callbacks=[mytraincallback],
    )

    logger.info("开始训练...")
    model.config.use_cache = False

    trainer.train()

    logger.info("保存模型")

    trainer.model.save_pretrained(checkpoint)


def generator(checkpoint):
    logger = create_logger(name="infer_log",
                           filename=checkpoint + "/chatglm4ie2.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info("INFER LOGGING......")
    logger.info(
        "基于QLora 4-bit调用chatglm3微调模型实现IE推理任务，超参数有--max_new_tokens %d" % (pargs.max_new_tokens))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    lora_config = LoraConfig.from_pretrained(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path,
                                              trust_remote_code=True,
                                              )

    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model, checkpoint)

    model.half()
    model.eval()

    prompt_prefix = "你现在是一个信息抽取模型，请你帮我找出下文中的实体关系三元组，形式为\"主体-主体类别_关系_客体-客体类别\"，三元组之间用\\n分割。\n文本："
    prompt_suffix = "\n实体关系三元组有：\n"
    intent = True

    while intent:
        query = input("\n文本：")
        if query == '':
            intent = False
            continue

        prompt = prompt_prefix + query + prompt_suffix

        logger.info(prompt)

        # 方法一：直接使用模型的chat函数
        btime = time.time()
        output = model.chat(tokenizer,
                            prompt,
                            history=[],
                            min_length=3,
                            max_new_tokens=pargs.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            repetition_penalty=3.5,
                            length_penalty=0.5,
                            early_stopping=True,
                            num_beams=3,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            )
        etime = time.time()
        tries = output[0]
        print(tries)

        print("实体关系三元组有：")
        logger.info("实体关系三元组有：")
        res = tries
        print(res["name"])
        logger.info(res["name"])
        for tri in res["content"].strip().split("\n"):
            print(tri)
            logger.info(tri)

        logger.info("耗时：" + format_time(etime - btime))

        # 方法二：使用generate函数
        res = tokenizer.build_chat_input(prompt, history=[], role="user")

        inputs = res["input_ids"].cuda()
        attention_mask = res["attention_mask"].cuda()

        btime = time.time()
        generated_ids = model.generate(
            inputs=inputs,
            attention_mask=attention_mask,
            min_length=3,
            max_new_tokens=pargs.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=3.5,
            length_penalty=0.5,
            early_stopping=True,
            num_beams=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        etime = time.time()

        decoded_pres = tokenizer.batch_decode(generated_ids,
                                              # skip_special_tokens=True,
                                              )[0]

        print(decoded_pres)

        idx1 = decoded_pres.find("<|assistant|> ")
        idx2 = decoded_pres.find("</s>")

        tries = decoded_pres[idx1 + 14:idx2]

        print("实体关系三元组有：")
        logger.info("实体关系三元组有：")
        for entity in tries.split("\n"):
            print(entity)
            logger.info(entity)
        logger.info("耗时：" + format_time(etime - btime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train", type=str, required=True)
    parser.add_argument("--train_path", default="data/bench_271k_train.csv", type=str)
    parser.add_argument("--num_samples", default=5000, type=int)
    parser.add_argument("--max_length", default=600, type=int)
    parser.add_argument("--max_new_tokens", default=200, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--accum_steps", default=4, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)

    pargs = parser.parse_args()

    if pargs.mode == "train":
        train()
    elif pargs.mode == "retrain":
        retrain("models/chatglm4ctc1")
    elif pargs.mode == "infer":
        generator("models/chatglm4ctc1")

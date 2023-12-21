'''
    基于chatGLM3完成文本纠错任务 chatglm3-6B
    hard prompt -4bit + 半精度float16
'''

from torch.utils.data import Dataset, DataLoader, random_split
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_scheduler, TrainingArguments, Trainer, TrainerCallback
from torch.nn.utils import clip_grad_norm_
import json, os, csv, numpy as np
import torch
from tqdm import tqdm
import argparse
import time
from time import strftime, gmtime
import warnings
import logging
from peft import get_peft_model, PromptTuningConfig,PromptTuningInit, TaskType, PeftModel, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
import random
import jieba

warnings.filterwarnings("ignore")

device = "cuda"


class MyDataset(Dataset):
    '''
    从文件读取句子对数据
    '''

    def __init__(self, data_path, tokenizer, logger):
        self.ietridata = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            next(reader)

            row_count = sum(1 for _ in reader)  # 计算行数

            logger.info("一共有%d条数据，从中随机选取%d条。" % (row_count, pargs.num_samples))

            random_rows = np.random.choice(np.arange(1, row_count + 1), pargs.num_samples, False)

            f.seek(0)

            # 遍历每一行
            for i, row in tqdm(enumerate(reader)):
                if i in random_rows:
                    # 读取指定列的值
                    originsen = row[1]
                    correctsen = row[0]

                    # print(originsen,correctsen)
                    assert len(originsen) == len(correctsen)

                    ask = originsen

                    ask = tokenizer.build_chat_input(ask, history=[], role='user')

                    answer = "\n "

                    if originsen == correctsen:
                        answer += correctsen
                    else:
                        words = jieba.tokenize(correctsen)

                        for res in words:
                            originword = originsen[res[1]:res[2]]
                            correctword = res[0]
                            if originword == correctword:
                                answer += originword
                            else:
                                answer += "【" + originword + "：" + correctword + "】"

                    answer = tokenizer(answer, add_special_tokens=False)

                    input_ids = ask["input_ids"][0].numpy().tolist() + answer["input_ids"] + [tokenizer.eos_token_id]
                    attention_mask = ask['attention_mask'][0].numpy().tolist() + answer['attention_mask'] + [1]
                    labels = [-100] * len(ask['input_ids'][0]) + answer["input_ids"] + [tokenizer.eos_token_id]

                    self.ietridata.append({'input_ids': input_ids[:pargs.max_length],
                                           'attention_mask': attention_mask[:pargs.max_length],
                                           "labels": labels[:pargs.max_length],
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

    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")

    simple_formatter = logging.Formatter(fmt="%(asctime)s %(message)s",
                                         datefmt="%H:%M:%S",
                                         )
    complex_formatter = logging.Formatter(fmt="%(asctime)s %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S",
                                          )

    consoleHandler.setFormatter(simple_formatter)
    fileHandler.setFormatter(complex_formatter)

    logger.addHandler(consoleHandler)
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
    if not os.path.exists(pargs.model_path):
        os.makedirs(pargs.model_path)

    logger = create_logger(name="train_log",
                           filename=pargs.model_path + "/chatglm4ctc.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info("Train Logging......")

    logger.info(
        "基于HardPrompt 4-bit微调chatglm3实现CTC任务，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d --accum_steps %d --train_path %s --model_path %s" % (
            pargs.num_samples, pargs.max_length, pargs.num_epochs, pargs.lr, pargs.batch_size, pargs.accum_steps,
            pargs.train_path, pargs.model_path))

    logger.info("开始创建分词器...")

    pretrained_checkpoint = "D:/pythonwork/LLMs4NER/uer/ZhipuAI/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint,
                                              trust_remote_code=True,
                                              )
    logger.info(tokenizer)

    logger.info("开始读取数据...")
    dataset = MyDataset(pargs.train_path, tokenizer, logger)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
    )

    dataloder=DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=data_collator)

    data=next(iter(dataloder))

    print(data)

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

    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="请替换句子中的错别字。",
        num_virtual_tokens=len(tokenizer("请替换句子中的错别字。")['input_ids']),
        tokenizer_name_or_path=pretrained_checkpoint,
    )

    logger.info(prompt_config)

    model = get_peft_model(model, prompt_config)
    logger.info(prompt_config)

    # model.print_trainable_parameters()

    # 计算参数量和 trainable 参数量
    trainable_param_count, param_count = model.get_nb_trainable_parameters()
    logger.info("trainable params: %d || all params: %d  || trainable%%: %f" % (
        trainable_param_count, param_count, (100.0 * trainable_param_count) / param_count))

    model.to(device)

    logger.info("开始设置训练参数TrainingArguments...")
    # 半精度eps重新设置，否则会导致loss上溢出或下溢出
    training_args = TrainingArguments(
        output_dir=pargs.model_path,
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
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        fp16=True,
    )

    mytraincallback = MyTrainCallback(logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[mytraincallback],
    )

    logger.info("开始训练...")
    model.config.use_cache = False

    trainer.train()

    logger.info("保存模型")

    trainer.model.save_pretrained(pargs.model_path)


def retrain():
    logger = create_logger(name="retrain_log",
                           filename=pargs.model_path + "/chatglm4ctc.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info("Retrain Logging......")
    logger.info(
        "基于SoftPrompt 4-bit继续微调chatglm3实现CTC任务，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d --accum_steps %d --train_path %s --model_path %s --model_checkpoint %s" % (
            pargs.num_samples, pargs.max_length, pargs.num_epochs, pargs.lr, pargs.batch_size, pargs.accum_steps,
            pargs.train_path, pargs.model_path, pargs.model_checkpoint))

    logger.info("开始创建分词器...")

    prompt_config = PromptTuningConfig.from_pretrained(pargs.model_path + pargs.model_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(prompt_config.base_model_name_or_path,
                                              trust_remote_code=True,
                                              )
    logger.info("开始读取数据...")
    dataset = MyDataset(pargs.train_path, tokenizer, logger)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
    )

    logger.info("开始创建模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        prompt_config.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, pargs.model_path + pargs.model_checkpoint)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    # model.print_trainable_parameters()

    # 计算参数量和 trainable 参数量
    trainable_param_count, param_count = model.get_nb_trainable_parameters()
    logger.info("trainable params: %d || all params: %d  || trainable%%: %f" % (
        trainable_param_count, param_count, (100.0 * trainable_param_count) / param_count))

    model.to(device)

    logger.info("开始设置训练参数TrainingArguments...")
    # 半精度eps重新设置，否则会导致loss上溢出或下溢出
    training_args = TrainingArguments(
        output_dir=pargs.model_path,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        per_device_train_batch_size=pargs.batch_size,
        gradient_accumulation_steps=pargs.accum_steps,
        num_train_epochs=pargs.num_epochs,
        lr_scheduler_type="linear",
        dataloader_drop_last=False,
        learning_rate=pargs.lr,
        weight_decay=1e-2,
        adam_epsilon=1e-4,
        max_grad_norm=1.0,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        fp16=True,
    )

    mytraincallback = MyTrainCallback(logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[mytraincallback],
    )

    logger.info("开始训练...")
    model.config.use_cache = False

    trainer.train()

    logger.info("保存模型")

    trainer.model.save_pretrained(pargs.model_path)


def generator():
    logger = create_logger(name="infer_log",
                           filename=pargs.model_path + "/chatglm4ctc.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info("Infer Logging......")
    logger.info(
        "基于SoftPrompt 4-bit调用chatglm3微调模型实现CTC任务，超参数有--max_new_tokens %d --model_path %s --model_checkpoint %s" % (
            pargs.max_new_tokens, pargs.model_path, pargs.model_checkpoint))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    prompt_config = PromptTuningConfig.from_pretrained(pargs.model_path + pargs.model_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(prompt_config.base_model_name_or_path,
                                              trust_remote_code=True,
                                              )

    model = AutoModelForCausalLM.from_pretrained(
        prompt_config.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model, pargs.model_path + pargs.model_checkpoint)

    model.half()
    model.eval()

    intent = True

    while intent:
        query = input("\n句子：")
        if query == '':
            intent = False
            continue

        prompt = query

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
        logger.info(tries)

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

        logger.info(decoded_pres)

        idx1 = decoded_pres.find("<|assistant|> ")

        tries = decoded_pres[idx1 + 15:]

        logger.info(tries)

        logger.info("耗时：" + format_time(etime - btime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train", type=str, required=True)
    parser.add_argument("--train_path", default="data/bench_peoplenews.csv", type=str)
    parser.add_argument("--model_path", default="models/chatglm4ctc52/", type=str)
    parser.add_argument("--model_checkpoint", default="", type=str)
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
        retrain()
    elif pargs.mode == "infer":
        generator()

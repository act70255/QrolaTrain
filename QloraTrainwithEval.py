import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate, nltk
import numpy as np
import os

from dotenv import load_dotenv
# 載入環境變數
load_dotenv(override=True)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
nltk.download('punkt')

# 從.env文件中獲取設定
MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")  # 模型的本地路徑
LLM_NAME = os.getenv("LLM_NAME", "mistralai/mistral-7b")  # 模型的本地路徑
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "qrola_output")  # 輸出目錄

RESOURCE_FOLDER=os.getenv("RESOURCE_FOLDER", "Resource")
DATASET_NAME= os.getenv("DATASET_NAME", "dataset.json")

model_path:str = os.path.join(os.path.abspath(MODEL_FOLDER), LLM_NAME)
dataset_path = os.path.join(os.path.abspath(RESOURCE_FOLDER), DATASET_NAME)

PROMPT_KEY_QUESTION = "question" #"instruction"
PROMPT_KEY_ANSWER = "answer" #"output"

# ------------------------------------------
def print_trainable_parameters(model):
    """
    輸出模型的參數
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # print(f"[輸出模型的參數] {name}: {param.numel()}")
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"[輸出模型的可訓練參數] trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
# ------------------------------------------

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 啟用 4-bit 量化
    bnb_4bit_quant_type="nf4",      # 使用 NF4 量化
    bnb_4bit_compute_dtype=torch.float16,  # 計算使用 float16
    bnb_4bit_use_double_quant=False  # 不使用雙重量化
)

# 2. 載入模型和分詞器
model_name = model_path  # 使用 Instruct 版本
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 設置填充標記

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自動分配到 GPU
    trust_remote_code=True
)
model.config.use_cache = False  # 訓練時禁用快取

# 3. 準備 LoRA 配置
lora_config = LoraConfig(
    r=16,              # LoRA 秩
    lora_alpha=32,     # 縮放因子
    target_modules=["q_proj", "v_proj"],  # 目標模組（注意力層）
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    #     "lm_head",
    # ],
    lora_dropout=0.05, # Dropout 比率
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 啟用量化訓練並應用 LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 5. 載入數據集
print(f"載入數據集: {dataset_path}")
# 5. 載入並預處理數據集
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

def formatting_prompts_func(example):
    text = f"[INST] {example[PROMPT_KEY_QUESTION]} [/INST] {example[PROMPT_KEY_ANSWER]}"
    return text

# 數據預處理：分詞並限制長度
def tokenize_function(example):
    # print(f"tokenize_function {str(example)}")
    return tokenizer(
        formatting_prompts_func(example),
        truncation=True,
        max_length=512,  # 在這裡控制最大序列長度
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=False)

# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=False)
# tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=False)

# 6. 設置訓練參數
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
    optim="paged_adamw_8bit",
    # 驗證參數
    per_device_eval_batch_size=8,  # 驗證批次大小
    eval_steps=100,  # 每 100 步進行一次驗證
    evaluation_strategy="steps",  # 按步驟進行評估 epoch / steps
    save_strategy="epoch",  # 每個 epoch 保存一次
    # load_best_model_at_end=True,  # 訓練結束時載入最佳模型
    metric_for_best_model="loss",  # loss以損失為指標選擇最佳模型 | eval_loss以評估損失為指標選擇最佳模型
    greater_is_better=False,  # 損失越低越好
    # predict_with_generate=True,  # 啟用生成模式進行評估
    # generation_max_length=128,   # 生成的最大長度
    # generation_num_beams=4,      # 光束搜索數量
    # 早停機制
    # early_stopping_patience=3,  # 如果連續 3 次評估沒有改善則停止
    # early_stopping_threshold=0.01,  # 至少改善 0.01 才算有效提升
    # 日誌參數
    logging_dir="logs",  # 日誌存放目錄
    logging_strategy="steps",  # 控制日誌記錄頻率
    save_total_limit=3,  # 僅保留最新的 3 個檢查點，節省空間
)

# 計算評估指標
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # 忽略填充標記（通常是-100）
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    
    # 計算基本指標
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # 返回多個評估指標
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 7. 初始化 SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    # eval_dataset=tokenized_eval_dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    # compute_metrics=compute_metrics,
    # formatting_func=formatting_prompts_func,  # 已移到 tokenize_function 中處理
)


# 8. 開始訓練
print_trainable_parameters(model)
trainer.train()

# 9. 保存模型
trainer.model.save_pretrained(OUTPUT_DIR + "_save")
tokenizer.save_pretrained(OUTPUT_DIR + "_save")

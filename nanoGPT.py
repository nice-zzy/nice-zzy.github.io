import torch
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import evaluate  # 假设这是一个用于评估的自定义模块或库

# 设置随机种子以确保结果的可重复性
torch.manual_seed(1337)
np.random.seed(1337)


# 配置参数类，用于存储实验配置
class Config:
    dataset = 'wikitext'  # 数据集名称
    dataset_config = 'wikitext-2-raw-v1'  # 数据集的特定配置
    model_name = 'gpt2'  # 使用的预训练模型名称
    max_length = 64  # 输入文本的最大长度
    batch_size = 16  # 批处理大小
    epochs = 1  # 训练轮数
    learning_rate = 5e-5  # 学习率


# 实例化配置类
config = Config()


# 准备数据函数
def prepare_data():
    # 加载数据集，这里只加载训练集的前1000个样本作为示例
    dataset = load_dataset(config.dataset, config.dataset_config, split='train[:1000]')

    # 加载GPT-2的tokenizer，并设置不清理空格
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=False)

    # 设置padding token为EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # 定义tokenization函数
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=config.max_length, padding='max_length')

        # 对数据集进行tokenization

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])

    # 将tokenized数据转换为TensorDataset，并创建DataLoader
    input_ids = torch.tensor(tokenized_dataset['input_ids'])
    attention_mask = torch.tensor(tokenized_dataset['attention_mask'])
    return DataLoader(TensorDataset(input_ids, attention_mask), batch_size=config.batch_size, shuffle=True)


# 训练模型函数
def train_model(model, dataloader, optimizer):
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 初始化总损失
    for batch in tqdm(dataloader, desc="Training"):  # 使用tqdm显示进度条
        input_ids, attention_mask = batch  # 解包batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)  # 前向传播
        loss = outputs.loss  # 计算损失
        total_loss += loss.item()  # 累加损失

        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    return total_loss / len(dataloader)  # 返回平均损失


# 获取隐藏状态函数
def get_hidden_states(model, input_ids):
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(input_ids, output_hidden_states=True)  # 前向传播并获取隐藏状态
    return outputs.hidden_states  # 返回隐藏状态


# 在GLUE benchmark上评估模型函数
def evaluate_on_glue(model, tokenizer):
    task = "mrpc"  # 选择GLUE中的一个任务
    metric = evaluate.load("glue", task)  # 加载评估指标
    dataset = load_dataset("glue", task, split='validation[:100]')  # 加载验证集的前100个样本

    # 定义编码函数
    def encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True,
                         max_length=config.max_length)

    encoded_dataset = dataset.map(encode, batched=True)  # 对数据集进行编码

    model.eval()  # 设置模型为评估模式
    predictions = []  # 初始化预测列表
    for batch in DataLoader(encoded_dataset, batch_size=config.batch_size):  # 遍历batch
        inputs = {k: torch.tensor(v) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}  # 准备输入
        with torch.no_grad():  # 禁用梯度计算
            outputs = model(**inputs)  # 前向传播
        logits = outputs.logits  # 获取logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())  # 获取预测结果

    results = metric.compute(predictions=predictions, references=encoded_dataset['label'])  # 计算评估结果
    return results  # 返回评估结果


# 主函数
def main():
    # 准备数据
    dataloader = prepare_data()

    # 初始化模型和优化器
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 训练模型
    for epoch in range(config.epochs):
        loss = train_model(model, dataloader, optimizer)  # 训练并获取损失
        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {loss:.4f}")  # 打印损失

    # 获取隐藏状态
    sample_input = next(iter(dataloader))[0][:1]  # 获取一个batch的输入作为示例
    hidden_states = get_hidden_states(model, sample_input)  # 获取隐藏状态
    print(f"Hidden states shape: {[hs.shape for hs in hidden_states]}")  # 打印隐藏状态的形状

    # 在GLUE benchmark上评估
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=False)  # 重新加载tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # 重新设置padding token
    results = evaluate_on_glue(model, tokenizer)  # 评估模型
    print("GLUE results:", results)  # 打印评估结果


# 程序入口
if __name__ == '__main__':
    main()  # 调用主函数
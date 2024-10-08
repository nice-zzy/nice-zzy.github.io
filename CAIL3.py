import json
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pylab import mpl

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# 数据加载和采样
def load_and_sample_data(file_path, sample_size):
    # 打开指定路径的文件，以只读模式和utf-8编码读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        # 逐行读取文件内容，将每行文本转换为JSON对象并存储在列表中
        data = [json.loads(line) for line in f]
    # 从数据列表中随机抽取指定数量的样本并返回
    return random.sample(data, sample_size)


train_data = load_and_sample_data('E:/pycharm/CAIL/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_train.json',
                                  20000)  
test_data = load_and_sample_data('E:/pycharm/CAIL/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_valid.json',
                                 8000)  


# 数据分析函数
def analyze_data(data):
    # 计算每个数据项中'fact'字段的长度，并存储在列表sentence_lengths中
    sentence_lengths = [len(item['fact']) for item in data]
    # 提取每个数据项的'meta'字典中的'accusation'列表的第一个元素，并存储在列表labels中
    labels = [item['meta']['accusation'][0] for item in data]

    # 打印平均句子长度，保留两位小数
    print(f"平均句子长度： {np.mean(sentence_lengths):.2f}")
    # 打印最长的句子长度
    print(f"最大句子长度： {max(sentence_lengths)}")
    # 打印最短的句子长度
    print(f"最小句子长度： {min(sentence_lengths)}")

    # 使用Counter统计各个标签的出现次数
    label_counts = Counter(labels)
    # 打印出现次数最多的前10个标签及其数量
    print(f"标签分布：")
    for label, count in label_counts.most_common(10):
        print(f"{label}: {count}")

    # 创建一个新的图形窗口，设置大小为10x5英寸
    plt.figure(figsize=(10, 5))
    # 绘制句子长度的直方图，分成50个区间
    plt.hist(sentence_lengths, bins=50)
    # 设置图形标题
    plt.title("句子长度分布")
    # 设置x轴标签
    plt.xlabel("长度")
    # 设置y轴标签
    plt.ylabel("频率")
    # 显示图形
    plt.show()


print("训练数据分析:")
analyze_data(train_data)
print("\n测试数据分析:")
analyze_data(test_data)

# 准备数据集（保持不变）
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')


class CriminalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        # 初始化方法，用于设置实例的属性
        # data: 输入的数据，通常是一个列表或数组，包含要处理的文本数据
        self.data = data
        # tokenizer: 分词器对象，用于将文本转换为模型可以理解的数字表示形式
        self.tokenizer = tokenizer
        # max_length: 序列的最大长度，用于限制输入文本的长度，避免过长的输入导致内存溢出或其他问题
        self.max_length = max_length

    def __len__(self):
        # 返回对象中存储的数据的长度
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引的数据项
        item = self.data[idx]
        
        # 使用tokenizer对数据项中的'fact'字段进行编码，并添加特殊标记
        encoding = self.tokenizer.encode_plus(
            item['fact'],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
    
        # 获取数据项的标签（第一个指控）
        label = item['meta']['accusation'][0]
        
        # 返回包含输入ID、注意力掩码和标签的字典
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


# 准备标签映射
all_labels = list(set([item['meta']['accusation'][0] for item in train_data + test_data]))
label_to_id = {label: i for i, label in enumerate(all_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

train_dataset = CriminalDataset(train_data, tokenizer)
test_dataset = CriminalDataset(test_data, tokenizer)

# 调整批量大小
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

#模型初始化
model = BertForSequenceClassification.from_pretrained('hfl/chinese-macbert-base', num_labels=len(all_labels))
device = torch.device('cpu')  # 使用CPU
model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)


# 训练函数
def train(model, dataloader, optimizer, device):
    # 设置模型为训练模式
    model.train()
    # 初始化总损失为0
    total_loss = 0
    # 遍历数据加载器中的每个批次
    for batch in dataloader:
        # 清空优化器的梯度缓存
        optimizer.zero_grad()
        # 将输入ID和注意力掩码转移到指定的设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # 将标签转换为对应的ID并转移到指定设备上
        labels = torch.tensor([label_to_id[label] for label in batch['labels']]).to(device)

        # 使用模型进行前向传播，得到输出结果
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # 从输出中获取损失值
        loss = outputs.loss
        # 累加当前批次的损失到总损失
        total_loss += loss.item()

        # 反向传播计算梯度
        loss.backward()
        # 根据梯度更新模型参数
        optimizer.step()

    # 返回平均损失值
    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, device):
    # 设置模型为评估模式，关闭梯度计算和dropout等训练特性
    model.eval()
    # 初始化预测结果和真实标签列表
    predictions = []
    true_labels = []

    # 使用torch.no_grad()上下文管理器，确保在此上下文中的所有计算不会跟踪梯度
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for batch in dataloader:
            # 将输入数据和注意力掩码转移到指定的设备（CPU或GPU）上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 通过模型进行前向传播，得到输出结果
            outputs = model(input_ids, attention_mask=attention_mask)
            # 获取预测的类别索引，即概率最大的类别
            _, preds = torch.max(outputs.logits, dim=1)

            # 将预测结果从GPU转移到CPU并转换为列表形式，添加到预测列表中
            predictions.extend(preds.cpu().tolist())
            # 将真实标签转换为对应的ID并添加到真实标签列表中
            true_labels.extend([label_to_id[label] for label in batch['labels']])

    # 计算准确率和分类报告，返回这两个指标
    return accuracy_score(true_labels, predictions), classification_report(true_labels, predictions,
                                                                           target_names=all_labels)


# 训练循环
num_epochs = 3  # 减少轮次以加快训练速度
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    accuracy, report = evaluate(model, test_loader, device)
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

# 保存模型
torch.save(model.state_dict(), 'criminal_accusation_model.pth')


# 推理函数
def predict(text, model, tokenizer, device):
    # 设置模型为评估模式，关闭梯度计算和dropout等训练特性
    model.eval()
    
    # 使用tokenizer对输入文本进行编码，包括添加特殊标记、截断或填充至最大长度512，并转换为PyTorch张量
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # 将编码后的input_ids和attention_mask转移到指定的设备（CPU或GPU）上
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 在不计算梯度的情况下运行模型前向传播，得到输出logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # 获取logits中预测概率最高的类别索引
        _, preds = torch.max(outputs.logits, dim=1)

    # 返回预测结果对应的标签
    return id_to_label[preds.item()]


# 测试推理
test_text = "2016年4月16日22时许，被告人李某在湖南省吉首市乾州新区某酒吧内，与被害人史某因琐事发生争执，继而厮打在一起。其间，被告人李某用随身携带的折叠刀朝被害人史某的左腹部捅刺一刀。经鉴定，被害人史某的损伤程度为轻伤一级。"
predicted_accusation = predict(test_text, model, tokenizer, device)
print(f"预测的罪名: {predicted_accusation}")
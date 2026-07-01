
import torch
import os
import datetime
from transformers.models import BertModel,BertTokenizer,BertConfig
current_date=datetime.datetime.now().date().strftime("%Y%m%d")

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert" # 模型名称
        self.data_path = "../data"  #数据集的根路径
        self.train_path = self.data_path + "\\train.txt"  # 训练集
        # self.train_path = self.data_path + "\\csv.txt"  # 训练集
        self.dev_path = self.data_path + "\\dev.txt"  # 少量验证集，快速验证
        self.test_path = self.data_path + "\\test.txt"  # 测试集

        #aspect_label_id 8个维度 [1,2,4]
        self.aspect_label_id_class_list_path=self.data_path + "\\dimension_labels.txt"
        #senti_label 正负性评价   [0,1,0,1,0,1,0,1]
        self.senti_label_class_list_path=self.data_path + "\\sentiment_labels.txt"
        #risk_label_id 评价类别 0-4
        self.risk_label_id_class_path=self.data_path + "\\risk_labels.txt"

        self.aspect_label_id_class_list = [line.strip() for line in open(self.aspect_label_id_class_list_path, encoding="utf-8")]  # 类别名单
        self.senti_label_class_list = [line.strip() for line in open(self.senti_label_class_list_path, encoding="utf-8")]  # 类别名单
        self.risk_label_id_class = [line.strip() for line in open(self.risk_label_id_class_path, encoding="utf-8")]  # 类别名单

        self.model_save_path = "../save_models/test_bertclassifer_model.pt"  #模型训练结果保存路径

        # 模型训练+预测的时候
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu

        # self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "../bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)#BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)# BERT模型的配置
        self.hidden_size = 768 # BERT模型的隐藏层大小

        self.embed_size = 256   # 嵌入层大小
        self.num_layers = 4 # LSTM的层数
        self.hidden_size_lstm = 256 # LSTM的隐藏层大小
        self.distil_model_save_dir = "../save_models"
        self.dropout = 0.2  # dropout概率
        self.num_dimensions = 8 # 评价维度数
        self.num_risk_levels = 4 # 评价等级数

if __name__ == '__main__':
    conf = Config()
    print(conf.bert_config)
    input_size=conf.tokenizer.convert_tokens_to_ids(["你","好","中国","人"])
    tokens = conf.tokenizer.convert_ids_to_tokens(input_size)
    print(input_size)
    print(tokens)
    # print(conf.class_list)
    print(conf.aspect_label_id_class_list)
    print(conf.senti_label_class_list)
    print(conf.risk_label_id_class)
    print(conf.risk_label_id_class[0])
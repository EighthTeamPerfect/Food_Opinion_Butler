import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from data_preprocess_addpath import load_and_preprocess_data, tokenize_text, TEXT_COL, ASPECT_NUM, RISK_NUM

# 全局配置
MODEL_SAVE_PATH = "../models/rf_model.pkl"
VECTORIZER_SAVE_PATH = "../models/tfidf_vectorizer.pkl"


# 训练RF模型
def train_rf_model(train_df, test_df):
    # 文本分词+TF-IDF向量化
    train_text = train_df[TEXT_COL].apply(tokenize_text)
    test_text = test_df[TEXT_COL].apply(tokenize_text)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    # 多标签维度分类模型
    aspect_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    y_train_aspect = train_df['aspect_tensor'].apply(lambda x: x.numpy()).tolist()
    y_test_aspect = test_df['aspect_tensor'].apply(lambda x: x.numpy()).tolist()
    aspect_model.fit(X_train, y_train_aspect)

    # 风险等级分类模型
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    y_train_risk = train_df['risk_label']
    y_test_risk = test_df['risk_label']
    risk_model.fit(X_train, y_train_risk)

    # 模型评估
    y_pred_aspect = aspect_model.predict(X_test)
    y_pred_risk = risk_model.predict(X_test)
    print("===== RF模型评估结果 =====")
    print("维度分类F1分数:", f1_score(y_test_aspect, y_pred_aspect, average='micro'))
    print("风险分类分类报告:\n", classification_report(y_test_risk, y_pred_risk))

    # 保存模型
    joblib.dump({
        "vectorizer": vectorizer,
        "aspect_model": aspect_model,
        "risk_model": risk_model
    }, MODEL_SAVE_PATH)
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    print(f"RF模型已保存至: {MODEL_SAVE_PATH}")
    return vectorizer, aspect_model, risk_model


# 预测函数
def predict_rf(text, vectorizer, aspect_model, risk_model):
    """单文本预测，返回维度标签、风险等级"""
    tokenized_text = tokenize_text(text)
    X = vectorizer.transform([tokenized_text])
    # 维度预测
    aspect_pred = aspect_model.predict(X)[0]
    aspect_str = ",".join([str(i) for i in range(ASPECT_NUM) if aspect_pred[i] == 1])
    # 风险预测
    risk_pred = risk_model.predict(X)[0]
    return {
        "aspect_label_id": aspect_str,
        "risk_label_id": int(risk_pred)
    }


# 主训练入口
if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data("../data/合并标注_SOP格式_长度_500.csv")
    train_rf_model(train_df, test_df)
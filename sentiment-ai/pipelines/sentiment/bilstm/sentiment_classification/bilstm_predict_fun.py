"""
目标：封装fasttext预测能力

步骤：
1. 导包
2. 模型加载
3. 模型预测
4. 返回结果
"""
# 步骤：
# 1. 导包
from bilstm_config import Config
from bilstm_model import BiLSTM_MultiHead
import time
import time
import torch
import json
import os
from bilstm_data_process import tokenize_data
from bilstm_data_build_batch import text_to_ids


previous_time = time.time()

def predict(input_data):
    previous_time = time.time()
    en2zh = {
        '0': "差评",
        '1': "好评",
    }
    # 2. 模型加载
    config = Config()

    vocab_path = config.save_vocab_path + "/vocab.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = json.load(f)
    else:
        # 兜底：如果没有 vocab.pkl，请务必在训练时保存它，这里会报错
        raise FileNotFoundError("未找到词表文件 vocab.pkl，无法进行预测！")

    # 2. 定义 JSON 文件路径（请确保这个路径和您保存的位置一致）
    param_json_path = os.path.join(config.save_best_param_path, 'best_params.json')

    # 3. 从 JSON 加载并注入配置
    if os.path.exists(param_json_path):
        with open(param_json_path, 'r', encoding='utf-8') as f:
            saved_params = json.load(f)

        # 循环注入：只注入 JSON 里存在的键，不存在的键保持 Config 默认值
        for k, v in saved_params.items():
            if hasattr(config, k):
                setattr(config, k, v)
                print(f"✅ 更新参数: {k} = {v}")
            else:
                print(f"⚠️ JSON 中存在未知参数: {k}，已跳过")
    else:
        print("⚠️ 未找到 JSON 配置文件，使用 Config 默认参数。")

    model = BiLSTM_MultiHead(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=vocab.get('<PAD>', 0)
    )
    model.load_state_dict(torch.load(config.save_model_path + '/bilstm_sentiment_best.pt'))
    model.to(config.device)

    model.eval()

    # 3. 模型预测
    input_text = [input_data['text']]
    input_new_text = tokenize_data(input_text, config.zh_model)
    ids, seq_len = text_to_ids(" ".join(input_new_text), vocab, config.max_len)
    with torch.no_grad():
        logits,_ = model(torch.tensor([ids], dtype=torch.long).to(config.device))
        preds = torch.argmax(logits, dim=1)

    result_label = en2zh.get(str(preds.cpu().numpy().item()), "未知")

    # 打印日志 (可选)
    print(f"输入文本: {input_text} | 预测结果: {result_label}")

    input_data['pre_result'] = result_label
    input_data['duration'] = time.time() - previous_time
    return input_data


if __name__ == '__main__':
    test_cases = [
        "从下单到收货，用了整整 7 天。",
        "收到货了，但盒子是破的。",
        "这个价格买了三个，值吗？",
        "这是我这辈子买过最差的手机了。",
        "Oh, great! 这包装真精美，就是这质量太 terrible 了。",
        "Very nice！结果用了一天就坏了。",
        "这质量确实无敌了，佩服佩服。",
        "Good luck with this product.",
        "好！",
        "退！",
        "还行吧。",
        "有点贵。",
        "Very bad product.",
        "This is great!",
        "I hate this item.",
        "Not worth the money."
    ]
    for text in test_cases:
        input_data = {'text': text}
        result = predict(input_data)
        print(result)

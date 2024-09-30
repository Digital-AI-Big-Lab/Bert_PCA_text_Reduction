#Install Package
pip install transformers scikit-learn numpy
from transformers import BertModel, BertTokenizer
import torch
from sklearn.decomposition import PCA
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
# # 分割句子的函數
text =["在現代社會中，科技的快速發展正在以前所未有的速度改變著我們的生活方式。從智能手機到人工智慧，從大數據到物聯網，每一項技術的突破都深刻影響著人類的生活、工作和社交方式。首先，通訊技術的進步使得人們無論身處何地，都能夠即時聯繫。這不僅縮短了地理距離，也使得全球化進程得以加速。無論是在商業會議中使用的視頻通話技術，還是日常生活中依賴的即時訊息應用，都讓我們感受到科技給社會帶來的便捷與效率提升。然而，科技的飛速發展同時也帶來了許多新的挑戰和問題。首先，信息過載成為一個普遍現象。每天，互聯網上都有數以億計的新資訊誕生，無論是新聞報導、社交媒體動態，還是科學研究成果，這些信息都淹沒在我們的生活中，讓人無所適從。人們越來越難以分辨資訊的真偽，甚至會因為過量的信息而感到焦慮。這種現象在年輕一代中尤為明顯，他們習慣於通過社交媒體來獲取資訊，但也更容易受到假新聞和虛假訊息的影響。此外，科技的進步也對傳統產業造成了衝擊。自動化技術和人工智慧的應用使得許多傳統職業面臨被取代的風險。工廠中的機械臂、無人駕駛汽車、智能客服機器人，這些新技術在提高生產效率的同時，也帶來了就業市場的動盪。一些技能單一的勞動者可能會因為找不到合適的工作而陷入困境。這使得社會必須更加關注教育與職業培訓，幫助人們適應不斷變化的職場環境。面對科技發展帶來的挑戰，我們必須學會如何有效地管理和利用科技資源。一方面，政府和企業應該加強對新興科技的監管，避免技術被濫用或者造成不良影響；另一方面，我們每個人也應該提升自身的科技素養，學會批判性地看待信息，避免成為科技發展的受害者。只有這樣，我們才能在快速變遷的科技時代中，保持清醒的頭腦，真正享受科技進步帶來的紅利。總而言之，科技是一把雙刃劍，它在改善人類生活的同時，也帶來了諸多挑戰。我們需要以開放的態度迎接變革，同時以理性的思維應對問題，這樣才能在這個變幻莫測的時代中立於不敗之地。",
    "在這個快速變遷的數位時代，每一個組織都在尋找如何在市場上取得競爭優勢的秘訣。而能否成功的關鍵，在於企業能否有效運用四大核心策略：TWQR。這四個字母分別代表了 Transformation（轉型）、Work-life balance（工作生活平衡）、Quality（品質）、和 Resilience（韌性）。它們不僅影響企業的發展方向，更是現代社會中每個個人和團隊應當牢牢掌握的理念。我是廢話。我是測試集。"]
#Bert
# 取得 BERT 嵌入向量的函數
def get_bert_embeddings(sentences, model, tokenizer):
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用平均池化策略
        mean_embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
        sentence_embeddings.append(mean_embedding)
    return np.array(sentence_embeddings)

# 載入預訓練的BERT模型和分詞器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

for i in range(len(text)):
    print("original length:", len(text[i]))
    text_1 = text[i].strip('。')
    sentences = text_1.split('。')
    # print(sentences)
    sentence_embeddings = get_bert_embeddings(sentences, model, tokenizer)
    # 使用 PCA 進行降維
    pca = PCA()
    sentence_embeddings_reduced = pca.fit_transform(sentence_embeddings)
    eigenvalues = pca.explained_variance_
    eigenvalues_sum = eigenvalues.sum()
    # print(eigenvalues/eigenvalues_sum)
    # 計算每個句子的PCA score，根據解釋變異量進行加權
    explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)
    sentence_scores = np.dot(sentence_embeddings_reduced, explained_variance)
    # print(sentence_scores)
    # 根據PCA score選擇前三個句子
    top_sentence_indices = np.argsort(sentence_scores)[-3:]
    summary = "".join([sentences[j] for j in top_sentence_indices]) + "。"
    print("Summary:", summary)
    print("after summary length:", len(summary))

#Result
#original length: 796
#Summary: 在現代社會中，科技的快速發展正在以前所未有的速度改變著我們的生活方式只有這樣，我們才能在快速變遷的科技時代中，保持清醒的頭腦，真正享受科技進步帶來的紅利面對科技發展帶來的挑戰，我們必須學會如何有效地管理和利用科技資源。
#after summary length: 109
#original length: 203
#Summary: 而能否成功的關鍵，在於企業能否有效運用四大核心策略：TWQR在這個快速變遷的數位時代，每一個組織都在尋找如何在市場上取得競爭優勢的秘訣它們不僅影響企業的發展方向，更是現代社會中每個個人和團隊應當牢牢掌握的理念。
#after summary length: 105




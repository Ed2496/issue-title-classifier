import joblib
import pandas as pd
import numpy as np

# --- 設定模型和相關檔案的路徑 ---
# 這些路徑應指向您先前收到的檔案
# 確保這些檔案與此腳本在同一目錄，或者提供正確的絕對/相對路徑
SUBSYSTEM_MODEL_PATH = "./subsystem_classification_model.joblib"
SUBSYSTEM_VECTORIZER_PATH = "./subsystem_tfidf_vectorizer.joblib"
SUBSYSTEM_LABEL_ENCODER_PATH = "./subsystem_label_encoder.joblib"

ROOT_CAUSE_MODEL_PATH = "./root_cause_prediction_model.joblib"
ROOT_CAUSE_VECTORIZER_PATH = "./root_cause_tfidf_vectorizer.joblib"
ROOT_CAUSE_LABEL_ENCODER_PATH = "./root_cause_label_encoder.joblib"

TOP_N_ROOT_CAUSES = 5 # 您希望看到的 Top N 根本原因數量

def load_model_components(model_path, vectorizer_path, label_encoder_path):
    """載入模型、向量化器和標籤編碼器"""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(label_encoder_path)
        print(f"成功載入: {model_path}, {vectorizer_path}, {label_encoder_path}")
        return model, vectorizer, label_encoder
    except FileNotFoundError as e:
        print(f"錯誤: 找不到必要的模型檔案 - {e}。請確保檔案路徑正確且檔案存在。")
        return None, None, None
    except Exception as e:
        print(f"載入模型組件時發生錯誤: {e}")
        return None, None, None

def predict_subsystem(title_text, model, vectorizer, label_encoder):
    """預測單個標題的子系統"""
    if not all([model, vectorizer, label_encoder]):
        return "模型組件載入失敗"
    try:
        title_tfidf = vectorizer.transform([title_text])
        prediction_encoded = model.predict(title_tfidf)
        predicted_subsystem = label_encoder.inverse_transform(prediction_encoded)
        return predicted_subsystem[0]
    except Exception as e:
        return f"預測子系統時發生錯誤: {e}"

def predict_root_causes(title_text, model, vectorizer, label_encoder, top_n=TOP_N_ROOT_CAUSES):
    """預測單個標題的 Top N 技術根本原因及其機率"""
    if not all([model, vectorizer, label_encoder]):
        return ["模型組件載入失敗"]
    try:
        title_tfidf = vectorizer.transform([title_text])
        probabilities = model.predict_proba(title_tfidf)[0] # [0] 因為我們只有一個輸入
        
        # 獲取 Top N 機率及其對應的類別索引
        # np.argsort 返回的是從小到大排序的索引，所以取最後 N 個，然後反轉
        top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        results = []
        for i in top_n_indices:
            class_label = label_encoder.classes_[i]
            probability = probabilities[i]
            results.append(f"{class_label} (機率: {probability:.4f})")
        return results
    except Exception as e:
        return [f"預測根本原因時發生錯誤: {e}"]

if __name__ == "__main__":
    print("正在載入子系統分類模型組件...")
    sub_model, sub_vectorizer, sub_label_encoder = load_model_components(
        SUBSYSTEM_MODEL_PATH, SUBSYSTEM_VECTORIZER_PATH, SUBSYSTEM_LABEL_ENCODER_PATH
    )

    print("\n正在載入技術根本原因預測模型組件...")
    rc_model, rc_vectorizer, rc_label_encoder = load_model_components(
        ROOT_CAUSE_MODEL_PATH, ROOT_CAUSE_VECTORIZER_PATH, ROOT_CAUSE_LABEL_ENCODER_PATH
    )

    if not all([sub_model, rc_model]):
        print("\n由於模型載入失敗，無法繼續進行預測。請檢查檔案路徑和完整性。")
    else:
        # --- 在此處輸入您想要預測的標題 --- 
        # 您可以修改這個列表來測試不同的標題
        sample_titles = [
            "System cannot wake up from sleep mode after installing new graphics driver",
            "OS installation fails with blue screen error 0x0000007B",
            "Touchpad gestures not working on new laptop model",
            "Application crashes frequently after recent update",
            "Battery drains quickly even when idle"
        ]
        
        print("\n--- 開始預測 --- (請將模型檔案放置於同目錄下或修改路徑)")
        for title in sample_titles:
            print(f"\n標題: \"{title}\"")
            
            # 預測子系統
            predicted_subsystem = predict_subsystem(title, sub_model, sub_vectorizer, sub_label_encoder)
            print(f"  預測子系統: {predicted_subsystem}")
            
            # 預測技術根本原因
            predicted_root_causes = predict_root_causes(title, rc_model, rc_vectorizer, rc_label_encoder)
            print(f"  預測的 Top {TOP_N_ROOT_CAUSES} 技術根本原因:")
            for rc in predicted_root_causes:
                print(f"    - {rc}")


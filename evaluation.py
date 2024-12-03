import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

def evaluate_predictions(y_pred):
   
    y_true=pd.read_csv('evaluation_label.csv').to_numpy().flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 평가 지표 계산
    f1 = f1_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    
    # 결과 출력
    print("========EVALUATION RESULTS========")
    print(f"F1 :{f1:.3f}", f"BAC : {balanced_acc:.3f}",f"PPV (Precision): {ppv:.3f}")
    if f1 >= 0.8:
        print("🎉 비극적인 사고를 예방했습니다! 병원은 안전합니다.")
    elif f1 <= 0.5:
        print("💔 병원이 망했습니다. 더 나은 모델이 필요합니다.")
    else:
        print("⚠️ 모델 성능이 보통 수준입니다. 추가 개선이 필요합니다.")
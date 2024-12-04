# AMC-DLab Tutorial AI

(version 1.0) 2024.12.04

Contributors: [석현석](https://github.com/HYEONSEOKROCK), [유지원](https://github.com/Altaaaaaaa), [이건](https://github.com/az8602), [이재형](https://github.com/zeus9656)


# 진단검사 결과 검체뒤바뀜 오류 감지

## **프로젝트 목표**
1. **Clinical Tabular DB 다루기**
2. **진단검사 내 오류 감지 모델 개발**
3. **하이퍼파라미터 튜닝**

---

## **목차**

1. **Introductions**
   - [진단검사란](#진단검사란)
   - 튜토리얼 목차
   - 문제 상황

2. **Data**
   - 데이터 확인 결과

3. **Input Data Generation**

   3.1. 입력 데이터 구성을 위한 피벗팅  
   3.2. 데이터 인코딩

4. **Model Development**
   - 검체 뒤바뀜 오류 시뮬레이션

   4.1. Data Splitting  
   - 조건을 고려하지 않은 일반적인 방법

        4.1.1. XGBoost
        - 환경적 요소를 고려한 데이터 시뮬레이션 및 splitting

   4.2. DNN

   4.3. Preprocessing

5. **Hyperparameter + Architecture Tuning**

   5.1. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)  

   - 베이지안 최적화

   5.2. Optuna란?

---

## **진단검사란**

진단검사는 인체에서 유래하는 각종 검체(혈액, 소변, 체액 등)에 대한 적절한 검사로 질병 진단이나 치료효과 판정에 도움을 주는 검사입니다.

### **검사 순서**
![image.png?raw=true](https://github.com/dlab-amc/DLab-TG-AI/blob/main/images/image1.png?raw=true)

### **진단검사 오류란?**
![image.png?raw=true](https://github.com/dlab-amc/DLab-TG-AI/blob/main/images/image3.png?raw=true)

### **문제점 (기존 방법 포함)**
![image.png?raw=true](https://github.com/dlab-amc/DLab-TG-AI/blob/main/images/image2.png?raw=true)

### **그 중 샘플 뒤바뀜 오류**
![image-3.png?raw=true](https://github.com/dlab-amc/DLab-TG-AI/blob/main/images/image-3.png?raw=true)

---

## **문제 해결 방안**
본 프로젝트의 튜토리얼은 **제시된 상황에서 문제 해결하는 방식**으로 진행됩니다.

---

## **문제 상황**

### **배경**
서울의 한 번창하던 병원에서 갑자기 수많은 환자들이 이틀에 걸쳐 몰렸습니다. 이 과정에서 **혈액 검체가 뒤바뀌는 사고**가 발생하며, 잘못된 진단과 치료로 인해 몇몇 환자는 심각한 부작용을 겪거나 생명을 잃는 비극적인 일이 벌어졌습니다.

이 사고로 인해 병원의 신뢰도는 급격히 하락했고, 환자들은 다른 병원을 찾기 시작했습니다. 병원장은 이러한 상황을 지켜보며 극심한 스트레스와 우울증에 시달렸고, 기존 시스템으로는 문제를 해결할 수 없다는 것을 깨달았습니다. 병원은 문을 닫을 위기에 처하게 되었고, 병원장은 절망 끝에 생을 마감하려는 결심을 했습니다.

그러나, 그 순간 **신비로운 힘**에 의해 과거로 돌아갈 수 있는 기회를 얻게 되었습니다. 과거로 돌아온 병원장은 **AI(인공지능)**를 활용해 검체 뒤바뀜 문제를 근본적으로 해결하고 병원 시스템을 혁신하기로 결심했습니다.

---

### **목표**
11종의 검사항목으로 구성된 검체들의 검사 결과에서 **뒤바뀐 검체를 식별하는 AI 모델을 개발**하는 것입니다.

#### **검사항목**
- **Albumin**
- **ALP**
- **ALT**
- **AST**
- **BUN**
- **Creatinine**
- **GGT**
- **Glucose**
- **LDH**
- **Total Bilirubin**
- **Total Protein**

#### **상황**
- **기간**: 2024년 10월 29일 ~ 2024년 10월 31일
- **문제**: 검체 및 검사가 폭주하며 원인 미상으로 검체 뒤바뀜 사고 발생.
- **목표**: 뒤바뀐 검체를 정확히 찾아내는 **인공지능 모델 개발**.

---

## **데이터**

| 파일 이름              | 설명                                       |
|------------------------|--------------------------------------------|
| **development_set.csv** |  |
| **test_set.csv**        |  |
| **test_label.csv**      |  |

---

## **평가 방법**

### **기존 문제점**
- 기존 방법은 검체 뒤바뀜 오류를 충분히 탐지하지 못함.
- 탐지한 오류조차 많은 오판을 포함함.

### **해결 목표**
- **F1-score**가 가장 높은 인공지능 모델을 개발하여 오류 탐지 능력을 개선.

#### **F1-score 정의**
F1-score는 **Precision**과 **Recall**을 결합하여 계산됩니다:

$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

- **Precision:** 탐지된 오류 중 실제 오류인 비율.
- **Recall:** 실제 오류 중 탐지된 오류의 비율.

F1-score를 통해 오류를 **더 많이, 더 정확히 탐지**할 수 있는지를 평가합니다.


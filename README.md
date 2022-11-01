# traveldata
DACON 2022 관광데이터 AI 경진대회
## 1. Project Overview
  - 목표
    - 관광지에 관한 텍스트와 이미지 데이터를 이용하여 카테고리 자동분류.
  - 모델
    - [klue/bert-base](https://github.com/KLUE-benchmark/KLUE) fine-tuned model.
  - Data
    - 관광지 정보 (train: 16986개, test: 7280개).

## 2. Code Structure
``` text
├── data
|   ├── image
|   |    ├── train
|   |    └── test
|   ├── stopword.txt
|   ├── train.csv (not in repo)
|   └── test.csv  (not in repo)
├── argument.py
├── backtranslation.ipynb
├── cat1_label.pkl
├── cat2_label.pkl
├── cat3_label.pkl
├── dataset.py
├── inference.py
├── loss.py
├── model.py
├── train.py
├── trainer.py
└── utils.py
```

## 3. Detail 
  - Preprocess 
    - 정규표현식을 이용한 특수문자 제거.
    - [Open Korean Text Processor](https://github.com/open-korean-text/open-korean-text) 를 이용한 Stopword(출처: https://www.ranks.nl/stopwords/korean) 제거.
    - 문장내에 다수 공백 제거.
  - Augmentation
    - 네이버 Papago를 이용한 데이터 수가 적은(200개 이하) label을 영어와 일어로 번역 후 한글로 재번역하는 Backtranslation을 이용한 Augmentation 진행.
  - Model
    - 전처리 된 데이터를 klue/bert-base에 fine-tunning 과정을 거침.
    - K-Fold(5 Fold)의 교차 검증과정을 통해 보다 정교한 모델 제작.
    - Focal loss를 사용, Imbalance한 데이터 분포에 보다 정교한 loss를 사용하여 모델 성능 향상.
    - Batch size: 16 / Epoch : 5
  - Inference
    - 각 Fold model의 inference 과정을 거친후 나온 softmax 확률을 soft-voting 과정을 거쳐 최종추론 함.
  - 최종성적
    - Public Score: 0.6972 (22th/549)
    - Private Score: 0.69968(24th/549)

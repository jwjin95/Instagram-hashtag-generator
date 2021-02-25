# Instagram-hashtag-generator

## 프로젝트 개요

> Reference : CNN 및 Word2Vec 기반 이미지 연관 해시태그 추진 모델

- 입력 이미지에 대하여 자동으로 해시태그를 생성해주는 해시태그 추천 시스템
    - 이미지를 입력하면 입력 이미지와 유사한 K개의 이미지의 해시태그를 바탕으로 태그 전환 및 확장을 반복하여 추천 태그 생성
- `Python` `Keras` `Deep Learning` `CV` `NLP`

1. Data Collection
    - 인기 해시태그 Top 100을 기준으로 해시태그 1개 당 포스트 100개씩 수집
    - 인스타그램 포스트로부터 이미지 url, 이미지 파일, 장소 태그, 본문 해시태그, 댓글 해시태그 수집
2. Data Preprocessing & EDA
    - 전체에서 출현 빈도가 1인 해시태그 제거
    - `customized konlpy`를 활용하여 사용자 사전을 적용하여 해시태그 토큰화
    - 토큰화 결과 중 동일한 토큰이 연속하여 나오는 경우 토큰 확장에서 성능 저하로 이어져 제거
3. Modeling
    - 토큰 간 유사도를 계산하기 위해 워드 임베딩 방법론 중 `Word2Vec` 사용
    - 이미지 간 유사도를 계산하기 위해 ImageNet이 학습된 `VGG19`를 완전분류기를 제거하여 사용
    - 단어 벡터 간 유사도를 활용하여 **토큰 전환** 알고리즘 개발
    - 출현 빈도에 기반한 조건부 확률을 활용하여 **토큰 확장** 알고리즘 개발

## 프로젝트 디렉토리 구조

    |--- crawling
    |      |--- HashtagList.xlsx
    |      |--- crawler.py
    |      |--- img
    |      |--- data.csv
    |--- data
    |      |--- img
    |      |--- dataset.csv
    |      |--- knn_img
    |      |--- knn_data.csv
    |--- preprocessing
    |      |--- UserDic.txt
    |      |--- Preprocessing.py
    |      |--- Tokenizer.py
    |--- modeling
    |      |--- UserDic.txt
    |      |--- KNN_Classifier.py
    |      |--- KNN_Classifier_data.pickle
    |      |--- KNN_Classifier_tutorial.ipynb
    |      |--- Word2Vec.py
    |      |--- Word2Vec_model.pickle
    |      |--- Word2Vec_tutorial.ipynb
    |      |--- TokenConversion.py
    |      |--- TokenConversion_tutorial.ipynb
    |      |--- TokenExpansion.py
    |      |--- TokenExpansion_tutorial.ipynb
    |--- docs
    |--- refs

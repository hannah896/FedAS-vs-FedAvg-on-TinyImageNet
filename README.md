# FedAS: 개인화 연합 학습의 불일치 문제 해결
### 사물인터넷 Lab2 논문 구현 수행_팀4
### FedAS-vs-FedAvg-on-TinyImageNet
 ⦁  [발표자료](https://github.com/hannah896/FedAS-vs-FedAvg-on-TinyImageNet/blob/main/%5B%EC%82%AC%EB%AC%BC%EC%9D%B8%ED%84%B0%EB%84%B7-%EC%98%A4%ED%9B%84%EB%B0%98%5D_%ED%8C%804_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B82_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pptx)
 
 ⦁  [FedAS 논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.pdf?utm_source=chatgpt.com)
 
------------------------------------------------------------------------------------------------------
# [CVPR24] FedAS: Bridging Inconsistency in Personalized Fedearated Learning
개인화 연합 학습(PFL)에서 발생하는 불일치 문제 해결 프레임워크

> Xiyuan Yang, Wenke Huang, Mang Ye  
> *CVPR 2024*, [Paper Link](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html)

Code Implementation and Informations about FedAS



## Citation
```
@inproceedings{cvpr24_xiyuan_fedas,
    author    = {Yang, Xiyuan and Huang, Wenke and Ye, Mang},
    title     = {FedAS: Bridging Inconsistency in Personalized Fedearated Learning},
    booktitle = {CVPR},
    year      = {2024}
}
```
------------------------------------------------------------------------------------------------------

## 핵심 문제 정의
### 1. 클라이언트 내부 불일치 (Intra-client Inconsistency)
 ⦁ 개인화 파라미터(Local)와 공유 파라미터(Global) 간 학습 방향이 충돌
 
 ⦁ 중앙에서 받은 전역 모델이 로컬 지식을 덮어쓰는 문제
 
 → 개인화 모델의 일반화 능력 저하

### 2. 클라이언트 간 불일치 (Inter-client Inconsistency)
 ⦁ 데이터가 부족하거나 참여하지 않는 낙오자(Stragglers)
 
 ⦁ 미학습 모델이 집계에 참여 → 전체 수렴 성능 저하
 
 → 전체 시스템 성능 하락 및 불안정한 학습




## 제안 기법: FedAS Framework
PA(Parameter-Alignment) + CS(Client-Synchronization)
두 가지 기법을 결합해 불일치 문제를 동시에 해결함

### 1. Parameter Alignment (PA)
 ⦁ 로컬 파라미터를 전역 파라미터에 정렬(Alignment)
 
 ⦁ 클라이언트 내부 지식의 보존과 공유를 균형 있게 유지
 
 → 일관된 데이터 처리 & 로컬 적응력 증가

### 2. Client Synchronization (CS)
 ⦁ 참여하지 않은 클라이언트 영향을 최소화
 
 ⦁ Fisher Information Matrix(FIM) 기반 중요도 추정
 
 ⦁ 중요도 높은 업데이트에 집계 가중치 부여
 
 → 낙오자 문제 해결 → 안정 수렴




## 알고리즘 개요 흐름
1. 로컬 모델 학습 진행
2. PA를 통해 전역 파라미터와 정렬
3. 각 파라미터 중요도를 FIM으로 계산
4. CS 기반 파라미터 동기화
5. α-가중치를 반영하여 서버에서 전역 모델 집계
6. 반복 수행




## 실험 환경
항목	        내용
데이터셋	    CIFAR-10, CIFAR-100, TinyCIFAR
데이터 분할 	Dirichlet 분포 기반 Non-IID
백본 모델	    CNN 기반 네트워크
비교 방식    	FedAvg, pFedMe, FedALA, FedProto 등 최신 PFL 기법




## 실험 결과

### CIFAR-100 결과
 ⦁ 모든 조건에서 기존 방법 대비 일관된 성능 향상
 
 ⦁ 데이터 이질성 강화 시에도 정확도 우수 유지
 
 ⦁ 13개 실험 조합 중 5개 이상에서 최고 성능 달성

### Ablation Study
PA 제거:	내부 불일치 증가 → 성능 하락
CS 제거:	낙오자 영향 확대 → 수렴 저하
둘 다 유지:	최고 성능




## 추가 실험
### FedAS-vs-FedAvg-on-TinyImageNet
FedAS,
FedAvg,
FedPer 의 결과물을 담고 있음.

⦁ 추가 데이터 셋으로 학습했을 때의 FedAS, FedAvg의 Global 환경에서의 결과 비교 실험으로 추가실험 진행

⦁ 32*32로 이미지 파일 변형후 실험 진행

- 글로벌 환경 및 다양한 데이터셋에서도 안정적 성능 확인
- FedAvg보다 더 높은 Robustness 확보



# 설치 및 실행 메뉴얼

---

## 1. 환경 설정

---

### 1.1 Repository 클론

```bash
git clone https://github.com/<your-id>/FedAS-vs-FedAvg-on-TinyImageNet.git
cd FedAS-vs-FedAvg-on-TinyImageNet
```

### 1.2 Python 환경 생성 (권장)

```bash
python3 -m venv fedasenv
source fedasenv/bin/activate
```

### 1.3 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

※ 필요한 기본 라이브러리 (참고)
```
numpy
torch
torchvision
tqdm
matplotlib
h5py
pillow
```

---

## 2. TinyImageNet 데이터 준비

---

### 2.1 TinyImageNet-200 다운로드

---

TinyImageNet 공식 다운로드 페이지:

https://tiny-imagenet.herokuapp.com/

압축을 풀고 아래 구조로 배치:

```
~/data/tiny-imagenet-200/
    train/
    val/
    wnids.txt
    words.txt
```

---

## 3. TinyImageNet → FedAS 학습용 데이터 변환

---

### 3.1 변환 스크립트 실행

```bash
cd dataset
python3 tiny2cifar.py
```

### 3.2 변환 후 생성되는 구조

```
dataset/tiny_processed/
   client_0.h5
   client_1.h5
   ...
   client_19.h5
   stat.json
```

---

각 client_x.h5 내부 구조

```
x : uint8 이미지 (N, 32, 32, 3)
y : 레이블 (N,)
```

총 20개의 클라이언트 데이터가 생성되며 Dirichlet(α=0.3) 기반 non-IID 분포로 나뉨.

---

## 4. FedAS / FedAvg 학습 실행

---

### 4.1 FedAS 실행

```bash
cd system
python3 main.py \
    -did 0 \
    -data TinyCifar \
    -nb 200 \
    -m cnn \
    -lbs 16 \
    -gr 40 \
    -ls 5 \
    -algo FedAS \
    -jr 0.1 \
    -nc 20
```

---

### 4.2 FedAvg 실행

```bash
python3 main.py \
    -did 0 \
    -data TinyCifar \
    -nb 200 \
    -m cnn \
    -lbs 16 \
    -gr 40 \
    -ls 5 \
    -algo FedAvg \
    -jr 0.1 \
    -nc 20
```

---

## 5. 결과 저장 위치

---

**학습이 끝나면 아래 폴더에 결과가 자동 저장됨**

```
results/
   TinyCifar_FedAS_test_0.h5
   TinyCifar_FedAvg_test_0.h5
   TinyCifar_FedPer_test_0.h5 (선택)
```

---

**각 파일에 포함되는 값**

- **round_test_acc**
- **round_train_acc**
- **round_loss**

---

## 6. 비교 그래프 생성

---

### 6.1 비교 플롯 스크립트 실행

```bash
cd results
python3 compare_TinyCifar_test.py
```

---

### 6.2 출력물

```
tinycifar_fedavg_vs_fedas.png
```

---

**그래프 내용**

- FedAS vs FedAvg accuracy 곡선 비교
- 40 rounds 기준 global test accuracy 변화

---

## 요약

---

본 구현물에서는 TinyImageNet-200 데이터를 CIFAR 형태로 변환하여 총 20개 클라이언트에 Dirichlet 분포 기반으로 분산시켰고, FedAS 및 FedAvg 알고리즘을 동일한 조건에서 학습 후 정확도를 비교하였다. 모든 실험은 재현성을 위해 데이터 변환 스크립트, 학습 스크립트, 실험 결과를 자동 처리하는 코드와 그래프 생성 코드를 함께 제공하였다.

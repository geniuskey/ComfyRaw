# PRD: ComfyRaw - OpenCV 기반 영상처리 노드 에디터

## 1. 개요

### 1.1 프로젝트명
**ComfyRaw** - CPU 기반 OpenCV 영상처리 노드 에디터

### 1.2 배경
ComfyUI v0.12.3은 Stable Diffusion 등 생성형 AI 이미지 생성을 위한 노드 기반 워크플로우 도구이다. 본 프로젝트는 ComfyUI의 강력한 노드 에디터 인프라를 활용하되, 모든 생성형 AI 기능을 제거하고 순수 OpenCV/영상처리 도구로 변환한다.

### 1.3 목표
- PyTorch/CUDA 의존성 완전 제거
- 생성형 AI (Diffusion, CLIP, VAE 등) 코드 완전 제거
- CPU 기반 OpenCV 영상처리 노드 시스템 구축
- 기존 노드 에디터 UI/UX 유지

### 1.4 비목표
- GPU 가속 (CUDA/ROCm/MPS) 지원
- 딥러닝 기반 영상처리 (ESRGAN 등)
- 실시간 비디오 스트리밍
- 모바일/웹 배포

---

## 2. 사용자 및 사용 사례

### 2.1 대상 사용자
- 이미지/영상 편집이 필요한 개발자
- 영상처리 파이프라인을 시각적으로 구성하려는 사용자
- Python 코드 없이 OpenCV 기능을 활용하려는 비개발자
- 교육 목적의 영상처리 학습자

### 2.2 주요 사용 사례

#### UC1: 이미지 일괄 처리
- 폴더 내 이미지들에 동일한 처리 파이프라인 적용
- 예: 리사이즈 → 블러 → 색상 조정 → 저장

#### UC2: 비디오 프레임 처리
- 비디오 파일에서 프레임 추출
- 프레임별 영상처리 후 재조합
- 예: 영상 안정화, 컬러 그레이딩

#### UC3: 객체 감지 및 추적 (비 딥러닝)
- 컬러 기반 객체 감지
- 윤곽선 검출
- 템플릿 매칭

#### UC4: 이미지 분석
- 히스토그램 분석
- 에지 검출
- 특징점 추출

---

## 3. 기능 요구사항

### 3.1 핵심 기능 (P0)

#### 3.1.1 노드 에디터 인프라
- [x] 기존 웹 기반 노드 에디터 UI 유지
- [x] 그래프 실행 엔진 유지
- [x] 캐싱 시스템 유지
- [x] 워크플로우 저장/로드

#### 3.1.2 기본 I/O 노드
- Load Image (단일/폴더)
- Save Image (포맷 선택: PNG, JPEG, TIFF, BMP)
- Load Video
- Save Video
- Image Preview

#### 3.1.3 기본 변환 노드
- Resize (다양한 보간법)
- Crop
- Rotate
- Flip (수평/수직)
- Color Space Conversion (BGR, RGB, HSV, LAB, Gray 등)

#### 3.1.4 필터 노드
- Blur (Gaussian, Median, Bilateral, Box)
- Sharpen
- Morphology (Erode, Dilate, Open, Close)
- Threshold (Binary, Otsu, Adaptive)

### 3.2 확장 기능 (P1)

#### 3.2.1 에지/윤곽 노드
- Canny Edge Detection
- Sobel/Scharr/Laplacian
- Find Contours
- Draw Contours

#### 3.2.2 색상 처리 노드
- Brightness/Contrast
- Hue/Saturation
- Color Balance
- Invert
- Histogram Equalization

#### 3.2.3 합성 노드
- Blend/Composite
- Alpha Composite
- Mask Apply
- Channel Split/Merge

#### 3.2.4 기하 변환 노드
- Perspective Transform
- Affine Transform
- Remap

### 3.3 고급 기능 (P2)

#### 3.3.1 특징 검출 노드
- Corner Detection (Harris, Shi-Tomasi)
- SIFT/ORB/AKAZE (비 특허)
- Feature Matching

#### 3.3.2 비디오 처리 노드
- Frame Extract
- Frame Compose
- Optical Flow
- Background Subtraction

#### 3.3.3 유틸리티 노드
- Image Info (해상도, 채널, 타입)
- Math Operations (이미지 연산)
- Text Overlay
- Shape Draw

---

## 4. 비기능 요구사항

### 4.1 성능
- 4K 이미지 처리 시 응답 시간 < 1초 (일반 필터)
- 메모리 사용량 < 2GB (4K 이미지 10장 동시 처리 기준)
- CPU 멀티코어 활용 (OpenCV parallel_for)

### 4.2 호환성
- Python 3.10+
- Windows 10/11, macOS 12+, Ubuntu 22.04+
- 주요 브라우저 (Chrome, Firefox, Edge, Safari)

### 4.3 설치 용이성
- pip install 한 줄로 설치 가능
- 외부 바이너리 의존성 최소화
- Docker 이미지 제공

---

## 5. 제거 대상

### 5.1 완전 제거 디렉토리
```
comfy/ldm/                    # Latent Diffusion 모델
comfy/text_encoders/          # 텍스트 인코더
comfy_api_nodes/              # 외부 AI API 노드
models/                       # AI 모델 저장소
```

### 5.2 완전 제거 파일
```
comfy/samplers.py             # 샘플링 알고리즘
comfy/sample.py               # 샘플링 실행
comfy/sd.py                   # Stable Diffusion 로딩
comfy/controlnet.py           # ControlNet
comfy/lora.py                 # LoRA
comfy/clip_*.py               # CLIP 관련
comfy/diffusers_*.py          # Diffusers 변환
latent_preview.py             # 잠재 공간 미리보기
```

### 5.3 대폭 수정 파일
```
nodes.py                      # AI 노드 제거 → OpenCV 노드로 대체
folder_paths.py               # 모델 경로 → 이미지/비디오 경로
requirements.txt              # torch 제거 → opencv-python 추가
comfy/model_management.py     # GPU 관리 → 단순 메모리 관리
```

### 5.4 부분 제거 디렉토리
```
comfy_extras/                 # 대부분 제거, 일부 유틸리티 유지
comfy/                        # ldm, samplers 등 제거, 핵심 유틸 유지
```

---

## 6. 성공 지표

### 6.1 기술 지표
- 설치 크기 < 100MB (기존 ~10GB 대비)
- 의존성 패키지 수 < 20개
- 시작 시간 < 3초

### 6.2 기능 지표
- OpenCV 주요 함수 80% 이상 노드화
- 기존 ComfyUI 워크플로우 형식 호환

---

## 7. 마일스톤

### Phase 1: 정리 (Week 1-2)
- AI 관련 코드 완전 제거
- 의존성 정리
- 빌드 및 실행 확인

### Phase 2: 기반 구축 (Week 3-4)
- OpenCV 통합
- 기본 I/O 노드 구현
- 기본 변환 노드 구현

### Phase 3: 확장 (Week 5-6)
- 필터 노드 구현
- 색상 처리 노드 구현
- 합성 노드 구현

### Phase 4: 고급 기능 (Week 7-8)
- 특징 검출 노드
- 비디오 처리 노드
- 문서화 및 배포

---

## 8. 리스크

### 8.1 기술 리스크
| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 노드 시스템이 AI에 강하게 결합됨 | 중 | 고 | 실행 엔진 재작성 |
| 웹 UI가 AI 기능 의존 | 저 | 중 | UI 수정 |
| 캐싱 시스템 tensor 의존 | 중 | 중 | numpy 기반 재구현 |

### 8.2 일정 리스크
| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 제거 범위 과소평가 | 중 | 고 | 상세 코드 분석 선행 |
| 숨겨진 의존성 발견 | 중 | 중 | 점진적 제거 및 테스트 |

---

## 9. 부록

### 9.1 기존 ComfyUI 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI (React)                        │
├─────────────────────────────────────────────────────────────┤
│                     server.py (aiohttp)                      │
├─────────────────────────────────────────────────────────────┤
│                   execution.py (Graph Engine)                │
├─────────────────────────────────────────────────────────────┤
│  nodes.py  │  comfy_extras/  │  comfy_api_nodes/  │ custom/ │
├────────────┴────────────────────────────────────────────────┤
│                   comfy/ (AI Core - PyTorch)                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   ldm/  │  │samplers │  │   sd    │  │  clip   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 목표 ComfyRaw 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI (React)                        │
├─────────────────────────────────────────────────────────────┤
│                     server.py (aiohttp)                      │
├─────────────────────────────────────────────────────────────┤
│                   execution.py (Graph Engine)                │
├─────────────────────────────────────────────────────────────┤
│       nodes_opencv.py       │       custom_nodes/            │
├─────────────────────────────────────────────────────────────┤
│                   comfy_core/ (Utilities)                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │  image  │  │  video  │  │  utils  │                      │
│  └─────────┘  └─────────┘  └─────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

# TRD: ComfyRaw - 기술 요구사항 문서

## 1. 시스템 아키텍처

### 1.1 현재 아키텍처 분석

```
ComfyUI v0.12.3 구조:

├── 프레젠테이션 레이어
│   └── Web UI (comfyui-frontend-package)
│   └── server.py (aiohttp WebSocket)
│
├── 비즈니스 레이어
│   └── execution.py (그래프 실행 엔진)
│   └── comfy_execution/ (캐싱, 검증, 작업 큐)
│   └── nodes.py (노드 정의)
│
├── AI 레이어 (제거 대상)
│   └── comfy/ldm/ (Diffusion 모델)
│   └── comfy/samplers.py (샘플링)
│   └── comfy/model_management.py (GPU 관리)
│
└── 인프라 레이어
    └── folder_paths.py (파일 시스템)
    └── app/ (사용자, DB 관리)
```

### 1.2 목표 아키텍처

```
ComfyRaw 구조:

├── 프레젠테이션 레이어
│   └── Web UI (수정된 프론트엔드)
│   └── server.py (간소화)
│
├── 비즈니스 레이어
│   └── execution.py (그래프 실행 엔진 - 유지)
│   └── comfy_execution/ (캐싱, 검증 - 수정)
│   └── nodes_opencv.py (OpenCV 노드)
│
├── 처리 레이어 (신규)
│   └── comfy_cv/ (OpenCV 래퍼)
│   └── comfy_cv/image.py (이미지 처리)
│   └── comfy_cv/video.py (비디오 처리)
│
└── 인프라 레이어
    └── folder_paths.py (간소화)
    └── app/ (최소화)
```

---

## 2. 컴포넌트 상세 설계

### 2.1 유지 컴포넌트

#### 2.1.1 execution.py
**역할**: 노드 그래프 실행 엔진

**수정 사항**:
- `torch.Tensor` 참조 제거 → `numpy.ndarray` 사용
- `model_management` 호출 제거
- 메모리 관리 단순화

**핵심 클래스**:
```python
class ExecutionResult:       # 유지
class IsChangedCache:        # 유지, tensor 체크 제거
class CacheEntry:            # 유지
```

**핵심 함수**:
```python
async def execute_graph()    # 유지
def recursive_execute()      # 유지, tensor 처리 제거
def map_node_over_list()     # 유지
```

#### 2.1.2 comfy_execution/
**graph.py**: 유지 (DynamicPrompt, 그래프 파싱)
**caching.py**: 수정 (tensor 해시 → numpy 해시)
**validation.py**: 수정 (AI 타입 제거)
**jobs.py**: 유지
**progress.py**: 유지

#### 2.1.3 server.py
**유지 기능**:
- WebSocket 통신
- 이미지 업로드/다운로드
- 워크플로우 저장/로드
- 실행 큐 관리

**제거 기능**:
- 모델 다운로드 엔드포인트
- Latent 미리보기
- AI 모델 관련 API

#### 2.1.4 app/
**유지**:
- `frontend_management.py`
- `user_manager.py`
- `database/`

**제거**:
- `model_manager.py`
- AI 모델 관련 기능

### 2.2 제거 컴포넌트

#### 2.2.1 comfy/ldm/ (전체 제거)
```
ldm/
├── models/           # VAE, Diffusion 모델
├── modules/          # 어텐션, UNet
├── flux/             # Flux 모델
├── genmo/            # Genmo
├── hunyuan_video/    # 훈위안
├── wan/              # WAN
├── cosmos/           # Cosmos
├── pixart/           # PixArt
└── [15+ 모델들]
```

#### 2.2.2 comfy/ 핵심 AI 파일 (제거)
```
samplers.py           # K-Diffusion 샘플러
sample.py             # 샘플링 실행
sd.py                 # Stable Diffusion 로더
controlnet.py         # ControlNet
lora.py               # LoRA
clip_model.py         # CLIP
clip_vision.py        # CLIP Vision
model_patcher.py      # 모델 패칭
hooks.py              # 모델 훅
latent_formats.py     # 잠재 포맷
diffusers_*.py        # Diffusers 통합
model_detection.py    # 모델 감지
```

#### 2.2.3 comfy_extras/ (대부분 제거)
**제거 파일 (90+ 파일)**:
```
nodes_custom_sampler.py
nodes_train.py
nodes_gits.py
nodes_lotus.py
nodes_wan.py
nodes_hunyuan*.py
nodes_hooks.py
nodes_lora*.py
nodes_model*.py
[대부분의 AI 관련 노드]
```

**유지 가능 파일**:
```
nodes_images.py       # 일부 이미지 처리 (검토 필요)
nodes_mask.py         # 마스크 처리 (검토 필요)
```

#### 2.2.4 comfy_api_nodes/ (전체 제거)
```
apis/                 # 외부 AI API 클라이언트
├── openai.py
├── stability.py
├── runway.py
└── [25+ API]
nodes_*.py            # API 노드들
```

### 2.3 신규 컴포넌트

#### 2.3.1 comfy_cv/ (신규 모듈)

```python
# comfy_cv/__init__.py
from .image import ImageProcessor
from .video import VideoProcessor
from .types import CVImage, CVVideo

# comfy_cv/image.py
class ImageProcessor:
    @staticmethod
    def load(path: str) -> np.ndarray:
        """이미지 로드 (OpenCV)"""

    @staticmethod
    def save(image: np.ndarray, path: str, params: dict):
        """이미지 저장"""

    @staticmethod
    def resize(image: np.ndarray, size: tuple, interpolation: int):
        """리사이즈"""

    @staticmethod
    def blur(image: np.ndarray, method: str, kernel: int):
        """블러 적용"""

# comfy_cv/video.py
class VideoProcessor:
    @staticmethod
    def load(path: str) -> Iterator[np.ndarray]:
        """비디오 프레임 제너레이터"""

    @staticmethod
    def save(frames: List[np.ndarray], path: str, fps: float):
        """비디오 저장"""

# comfy_cv/types.py
CVImage = np.ndarray  # (H, W, C) uint8 or float32
CVVideo = List[CVImage]
```

#### 2.3.2 nodes_opencv.py (신규 노드 정의)

```python
# 노드 카테고리 구조
NODE_CATEGORIES = {
    "io": ["LoadImage", "SaveImage", "LoadVideo", "SaveVideo"],
    "transform": ["Resize", "Crop", "Rotate", "Flip"],
    "filter": ["GaussianBlur", "MedianBlur", "BilateralFilter"],
    "edge": ["Canny", "Sobel", "Laplacian"],
    "color": ["ColorConvert", "Brightness", "Contrast"],
    "morphology": ["Erode", "Dilate", "MorphOpen", "MorphClose"],
    "threshold": ["Threshold", "AdaptiveThreshold"],
    "composite": ["Blend", "AlphaComposite", "Mask"],
    "draw": ["DrawText", "DrawShape", "DrawContours"],
    "analyze": ["Histogram", "ImageInfo", "FindContours"],
}
```

---

## 3. 데이터 흐름

### 3.1 현재 데이터 흐름 (ComfyUI)

```
[이미지 입력]
     ↓
[VAEEncode] ─────→ torch.Tensor (latent)
     ↓
[KSampler] ──────→ torch.Tensor (denoised latent)
     ↓
[VAEDecode] ─────→ torch.Tensor (image)
     ↓
[SaveImage] ─────→ PIL.Image → 파일
```

### 3.2 목표 데이터 흐름 (ComfyRaw)

```
[LoadImage]
     ↓
[np.ndarray] (H, W, C) ──→ BGR uint8 또는 RGB float32
     ↓
[Filter/Transform 노드들]
     ↓
[np.ndarray]
     ↓
[SaveImage] ──→ 파일
```

### 3.3 이미지 타입 규약

```python
# 내부 이미지 포맷
IMAGE_DTYPE = np.float32  # 0.0 ~ 1.0
IMAGE_CHANNELS = "RGB"    # OpenCV BGR → RGB 변환

# 노드 간 전달 포맷
class ImageTensor:
    data: np.ndarray      # (B, H, W, C) float32 RGB

# I/O 시 변환
# 입력: cv2.imread (BGR uint8) → RGB float32
# 출력: RGB float32 → BGR uint8 → cv2.imwrite
```

---

## 4. 의존성 변경

### 4.1 제거 의존성

```txt
# requirements.txt에서 제거
torch
torchvision
torchaudio
torchsde
transformers
accelerate
safetensors
einops
spandrel
kornia
diffusers
tokenizers
sentencepiece
huggingface_hub
```

### 4.2 유지 의존성

```txt
# 유지
aiohttp>=3.11.8        # 웹 서버
Pillow                 # 이미지 I/O
numpy                  # 수치 연산
pydantic~=2.0          # 데이터 검증
SQLAlchemy             # DB
alembic                # 마이그레이션
PyYAML                 # 설정
tqdm                   # 진행률
psutil                 # 시스템 모니터링
```

### 4.3 추가 의존성

```txt
# 신규 추가
opencv-python>=4.8.0   # OpenCV 핵심
opencv-contrib-python  # 추가 모듈 (선택)
imageio                # 이미지 I/O 확장
imageio-ffmpeg         # 비디오 I/O
scikit-image           # 추가 이미지 처리 (선택)
```

### 4.4 최종 requirements.txt

```txt
# Core
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=9.0.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.9

# Web Server
aiohttp>=3.11.8
aiohttp-cors

# Data & Config
pydantic~=2.0
PyYAML>=6.0

# Database
SQLAlchemy>=2.0
alembic>=1.13.0

# Utilities
tqdm>=4.65.0
psutil>=5.9.0
typing_extensions>=4.0
```

---

## 5. 노드 타입 시스템 수정

### 5.1 현재 타입 (comfy/comfy_types/node_typing.py)

```python
class IO(StrEnum):
    STRING = "STRING"
    IMAGE = "IMAGE"          # torch.Tensor
    LATENT = "LATENT"        # 제거
    MODEL = "MODEL"          # 제거
    CLIP = "CLIP"            # 제거
    VAE = "VAE"              # 제거
    CONDITIONING = "CONDITIONING"  # 제거
    ...
```

### 5.2 목표 타입

```python
class IO(StrEnum):
    # 기본 타입
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"

    # 이미지/비디오 타입
    IMAGE = "IMAGE"          # np.ndarray (B, H, W, C)
    MASK = "MASK"            # np.ndarray (B, H, W)
    VIDEO = "VIDEO"          # List[np.ndarray]

    # 데이터 타입
    CONTOURS = "CONTOURS"    # List[np.ndarray]
    KEYPOINTS = "KEYPOINTS"  # List[cv2.KeyPoint]
    HISTOGRAM = "HISTOGRAM"  # np.ndarray

    # 변환 타입
    MATRIX = "MATRIX"        # np.ndarray (3x3 or 2x3)
```

---

## 6. 캐싱 시스템 수정

### 6.1 현재 캐싱 (tensor 기반)

```python
# comfy_execution/caching.py
def get_hash(value):
    if isinstance(value, torch.Tensor):
        return hash(value.data_ptr())
    ...
```

### 6.2 목표 캐싱 (numpy 기반)

```python
def get_hash(value):
    if isinstance(value, np.ndarray):
        # 빠른 해시: shape + dtype + 샘플링
        return hash((
            value.shape,
            value.dtype.str,
            value.flat[::max(1, len(value.flat)//100)].tobytes()
        ))
    ...
```

---

## 7. 메모리 관리 수정

### 7.1 현재 메모리 관리 (comfy/model_management.py)

```python
# GPU VRAM 관리, 모델 오프로딩
class VRAMState(Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
```

### 7.2 목표 메모리 관리 (단순화)

```python
# comfy_cv/memory.py
class MemoryManager:
    """CPU 메모리 관리"""

    @staticmethod
    def get_available_memory() -> int:
        """사용 가능한 RAM (bytes)"""
        return psutil.virtual_memory().available

    @staticmethod
    def estimate_image_memory(shape: tuple, dtype: np.dtype) -> int:
        """이미지 메모리 추정"""
        return np.prod(shape) * dtype.itemsize

    @staticmethod
    def should_release_cache() -> bool:
        """캐시 해제 필요 여부"""
        return MemoryManager.get_available_memory() < 1024 * 1024 * 512  # 512MB
```

---

## 8. 폴더 구조 수정

### 8.1 현재 folder_paths.py

```python
folder_names_and_paths = {
    "checkpoints": (...),
    "loras": (...),
    "vae": (...),
    "clip_vision": (...),
    "diffusion_models": (...),
    # 20+ 모델 폴더
}
```

### 8.2 목표 folder_paths.py

```python
folder_names_and_paths = {
    "input": ([os.path.join(base_path, "input")], [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]),
    "output": ([os.path.join(base_path, "output")], [".png", ".jpg", ".jpeg"]),
    "temp": ([os.path.join(base_path, "temp")], None),
    "workflows": ([os.path.join(base_path, "workflows")], [".json"]),
    "video_input": ([os.path.join(base_path, "video_input")], [".mp4", ".avi", ".mov", ".mkv"]),
    "video_output": ([os.path.join(base_path, "video_output")], [".mp4", ".avi"]),
}
```

---

## 9. 테스트 전략

### 9.1 단위 테스트

```python
# tests/test_nodes_opencv.py
class TestLoadImage:
    def test_load_png(self):
        node = LoadImage()
        result = node.execute(image="test.png")
        assert isinstance(result[0], np.ndarray)
        assert result[0].ndim == 4  # (B, H, W, C)

class TestResize:
    def test_resize_nearest(self):
        ...

class TestBlur:
    def test_gaussian_blur(self):
        ...
```

### 9.2 통합 테스트

```python
# tests/test_execution.py
class TestGraphExecution:
    def test_simple_pipeline(self):
        """LoadImage → Resize → Blur → SaveImage"""

    def test_branching_pipeline(self):
        """LoadImage → (Edge + Blur) → Blend"""

    def test_caching(self):
        """동일 입력 시 캐시 히트 확인"""
```

### 9.3 성능 테스트

```python
# tests/test_performance.py
class TestPerformance:
    def test_4k_resize_time(self):
        """4K 이미지 리사이즈 < 100ms"""

    def test_memory_limit(self):
        """10개 4K 이미지 처리 시 메모리 < 2GB"""
```

---

## 10. 마이그레이션 계획

### 10.1 Phase 1: 정리
1. AI 관련 디렉토리 삭제
2. AI 관련 파일 삭제
3. requirements.txt 정리
4. 빌드 테스트

### 10.2 Phase 2: 수정
1. `execution.py` 수정 (tensor → numpy)
2. `comfy_execution/caching.py` 수정
3. `folder_paths.py` 간소화
4. `server.py` 간소화

### 10.3 Phase 3: 구현
1. `comfy_cv/` 모듈 생성
2. `nodes_opencv.py` 노드 구현
3. 테스트 작성

### 10.4 Phase 4: 검증
1. 전체 테스트
2. 성능 검증
3. 문서화

---

## 11. 위험 요소 및 대응

### 11.1 기술적 위험

| 위험 | 영향 | 대응 |
|------|------|------|
| 프론트엔드 AI 의존성 | 중 | 프론트엔드 빌드 검토 |
| 캐싱 시스템 tensor 의존 | 고 | numpy 해시 구현 |
| 실행 엔진 tensor 의존 | 고 | 단계적 리팩토링 |
| 숨겨진 import 발견 | 중 | 정적 분석 도구 사용 |

### 11.2 코드 의존성 그래프

```
execution.py
├── nodes.py
├── comfy_execution/*
├── comfy/model_management.py  ← 수정 필요
└── comfy/ops.py               ← 제거 필요

nodes.py
├── comfy/sd.py                ← 제거
├── comfy/samplers.py          ← 제거
├── comfy/controlnet.py        ← 제거
└── folder_paths.py            ← 수정 필요
```

---

## 12. 부록: 코드 변경 예시

### 12.1 execution.py 수정 예시

```python
# Before
def get_output_data(obj, input_data_all):
    ...
    if isinstance(r, torch.Tensor):
        r = r.cpu()
    ...

# After
def get_output_data(obj, input_data_all):
    ...
    if isinstance(r, np.ndarray):
        r = np.ascontiguousarray(r)
    ...
```

### 12.2 노드 정의 예시

```python
# nodes_opencv.py
class GaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_blur"
    CATEGORY = "opencv/filter"

    def apply_blur(self, image: np.ndarray, kernel_size: int, sigma: float):
        # image: (B, H, W, C)
        result = []
        for img in image:
            blurred = cv2.GaussianBlur(
                (img * 255).astype(np.uint8),
                (kernel_size, kernel_size),
                sigma
            )
            result.append(blurred.astype(np.float32) / 255.0)
        return (np.stack(result),)
```

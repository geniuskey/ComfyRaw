# TASKS: ComfyRaw 변환 작업 목록

## 개요
ComfyUI v0.12.3 → ComfyRaw (OpenCV 노드 에디터) 변환 작업

---

## Phase 1: 정리 (AI 코드 제거)

### 1.1 디렉토리 삭제

- [ ] **TASK-001**: `comfy/ldm/` 디렉토리 전체 삭제
  - 경로: `comfy/ldm/`
  - 내용: Latent Diffusion 모델 구현체 전체
  - 예상 파일 수: 100+

- [ ] **TASK-002**: `comfy/text_encoders/` 디렉토리 전체 삭제
  - 경로: `comfy/text_encoders/`
  - 내용: CLIP, T5 등 텍스트 인코더

- [ ] **TASK-003**: `comfy_api_nodes/` 디렉토리 전체 삭제
  - 경로: `comfy_api_nodes/`
  - 내용: OpenAI, Stability 등 외부 API 노드

- [ ] **TASK-004**: `models/` 디렉토리 구조 정리
  - 기존: checkpoints, loras, vae, clip_vision, diffusion_models 등 20+ 폴더
  - 목표: input, output, temp, video_input, video_output만 유지
  - 기존 폴더 삭제 (빈 폴더만)

### 1.2 파일 삭제 (comfy/ 핵심)

- [ ] **TASK-005**: AI 샘플링 관련 파일 삭제
  ```
  comfy/samplers.py
  comfy/sample.py
  comfy/k_diffusion/
  ```

- [ ] **TASK-006**: 모델 로딩/변환 파일 삭제
  ```
  comfy/sd.py
  comfy/diffusers_load.py
  comfy/diffusers_convert.py
  comfy/model_detection.py
  comfy/supported_models.py
  comfy/supported_models_base.py
  ```

- [ ] **TASK-007**: CLIP/ControlNet/LoRA 파일 삭제
  ```
  comfy/clip_model.py
  comfy/clip_vision.py
  comfy/controlnet.py
  comfy/lora.py
  comfy/lora_convert.py
  ```

- [ ] **TASK-008**: 기타 AI 관련 파일 삭제
  ```
  comfy/model_patcher.py
  comfy/hooks.py
  comfy/latent_formats.py
  comfy/extra_samplers/
  comfy/gligen.py
  comfy/t2i_adapter/
  comfy/inpaint/
  comfy/upscale/
  ```

- [ ] **TASK-009**: 루트 AI 관련 파일 삭제
  ```
  latent_preview.py
  ```

### 1.3 comfy_extras/ 정리

- [ ] **TASK-010**: comfy_extras/ AI 노드 파일 삭제
  - 삭제 대상 (예시, 전체 검토 필요):
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
  nodes_flux.py
  nodes_sd3.py
  nodes_sdxl*.py
  nodes_stable_cascade.py
  nodes_stable3d.py
  nodes_audio.py
  nodes_*_video*.py
  [90+ 파일 검토 필요]
  ```

- [ ] **TASK-011**: comfy_extras/ 유지 가능 파일 검토
  - 검토 대상:
  ```
  nodes_images.py       # 기본 이미지 처리 - 검토 후 수정
  nodes_mask.py         # 마스크 처리 - 검토 후 수정
  nodes_compositing.py  # 합성 - 검토 후 수정
  ```

### 1.4 nodes.py 정리

- [ ] **TASK-012**: nodes.py에서 AI 노드 제거
  - 제거 대상 노드:
  ```python
  CLIPTextEncode
  CLIPTextEncodeSDXL
  CLIPVisionEncode
  CLIPVisionLoader
  CLIPLoader
  CLIPSetLastLayer
  ConditioningCombine
  ConditioningConcat
  ConditioningSetArea
  ConditioningSetMask
  KSampler
  KSamplerAdvanced
  VAEEncode
  VAEEncodeTiled
  VAEDecode
  VAEDecodeTiled
  VAELoader
  CheckpointLoader
  CheckpointLoaderSimple
  DiffusersLoader
  unCLIPCheckpointLoader
  LoraLoader
  HypernetworkLoader
  ControlNetLoader
  ControlNetApply
  [모든 AI 관련 노드]
  ```

- [ ] **TASK-013**: nodes.py 유지 노드 목록
  - 유지 대상 (수정 필요):
  ```python
  LoadImage            # torch → numpy 수정
  SaveImage            # torch → numpy 수정
  LoadImageMask        # 수정
  ImageScale           # 수정
  ImageInvert          # 수정
  ImagePadForOutpaint  # 삭제 또는 수정
  EmptyImage           # 수정
  ```

---

## Phase 2: 의존성 정리

### 2.1 requirements.txt 수정

- [ ] **TASK-014**: AI 의존성 제거
  ```txt
  # 제거
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

- [ ] **TASK-015**: OpenCV 의존성 추가
  ```txt
  # 추가
  opencv-python>=4.8.0
  imageio>=2.31.0
  imageio-ffmpeg>=0.4.9
  scikit-image>=0.21.0  # 선택
  ```

- [ ] **TASK-016**: 최종 requirements.txt 작성
  ```txt
  numpy>=1.24.0
  opencv-python>=4.8.0
  Pillow>=9.0.0
  imageio>=2.31.0
  imageio-ffmpeg>=0.4.9
  aiohttp>=3.11.8
  aiohttp-cors
  pydantic~=2.0
  PyYAML>=6.0
  SQLAlchemy>=2.0
  alembic>=1.13.0
  tqdm>=4.65.0
  psutil>=5.9.0
  typing_extensions>=4.0
  ```

### 2.2 pyproject.toml 수정

- [ ] **TASK-017**: pyproject.toml 업데이트
  - 프로젝트명: ComfyUI → ComfyRaw
  - 의존성 업데이트
  - 불필요한 설정 제거

---

## Phase 3: 핵심 코드 수정

### 3.1 실행 엔진 수정

- [ ] **TASK-018**: execution.py torch 참조 제거
  - `import torch` 제거
  - `torch.Tensor` → `np.ndarray` 변경
  - `model_management` 호출 제거
  - 메모리 관리 단순화

- [ ] **TASK-019**: comfy_execution/caching.py 수정
  - tensor 해시 → numpy 해시
  - 캐시 키 생성 로직 수정

- [ ] **TASK-020**: comfy_execution/validation.py 수정
  - AI 타입 검증 제거
  - 새 타입 검증 추가

### 3.2 타입 시스템 수정

- [ ] **TASK-021**: comfy/comfy_types/node_typing.py 수정
  - AI 타입 제거 (LATENT, MODEL, CLIP, VAE, CONDITIONING 등)
  - 새 타입 추가 (CONTOURS, KEYPOINTS, HISTOGRAM, MATRIX)

- [ ] **TASK-022**: comfy_api/ 정리
  - torch_helpers/ 제거
  - numpy 기반 헬퍼 추가

### 3.3 폴더 경로 수정

- [ ] **TASK-023**: folder_paths.py 간소화
  - AI 모델 폴더 정의 제거
  - 이미지/비디오 폴더 정의만 유지
  ```python
  folder_names_and_paths = {
      "input": (...),
      "output": (...),
      "temp": (...),
      "workflows": (...),
      "video_input": (...),
      "video_output": (...),
  }
  ```

### 3.4 서버 수정

- [ ] **TASK-024**: server.py AI 관련 엔드포인트 제거
  - 모델 다운로드 API 제거
  - Latent 미리보기 제거
  - 모델 관련 API 제거

- [ ] **TASK-025**: app/model_manager.py 제거 또는 수정
  - AI 모델 관리 기능 제거

### 3.5 기타 정리

- [ ] **TASK-026**: comfy/model_management.py 단순화
  - GPU 관리 코드 제거
  - CPU 메모리 관리만 유지
  - 또는 완전 제거 후 신규 모듈로 대체

- [ ] **TASK-027**: comfy/ops.py 검토
  - torch 의존 여부 확인
  - 필요시 제거 또는 numpy 기반 재작성

- [ ] **TASK-028**: comfy/utils/ 검토
  - torch 의존 여부 확인
  - 유틸리티 함수 정리

---

## Phase 4: 새 기능 구현

### 4.1 OpenCV 래퍼 모듈

- [ ] **TASK-029**: comfy_cv/ 디렉토리 생성
  ```
  comfy_cv/
  ├── __init__.py
  ├── image.py      # 이미지 처리
  ├── video.py      # 비디오 처리
  ├── types.py      # 타입 정의
  └── memory.py     # 메모리 관리
  ```

- [ ] **TASK-030**: comfy_cv/image.py 구현
  ```python
  class ImageProcessor:
      load()
      save()
      resize()
      blur()
      threshold()
      edge_detect()
      ...
  ```

- [ ] **TASK-031**: comfy_cv/video.py 구현
  ```python
  class VideoProcessor:
      load_frames()
      save_video()
      extract_frame()
      ...
  ```

- [ ] **TASK-032**: comfy_cv/memory.py 구현
  ```python
  class MemoryManager:
      get_available_memory()
      estimate_image_memory()
      should_release_cache()
  ```

### 4.2 OpenCV 노드 구현

- [ ] **TASK-033**: nodes_opencv.py 기본 구조 생성
  - 노드 베이스 클래스
  - 카테고리 정의
  - NODE_CLASS_MAPPINGS

#### I/O 노드

- [ ] **TASK-034**: LoadImage 노드 구현
  - 단일 이미지 로드
  - 폴더 이미지 일괄 로드

- [ ] **TASK-035**: SaveImage 노드 구현
  - PNG, JPEG, TIFF, BMP 지원
  - 품질/압축 옵션

- [ ] **TASK-036**: LoadVideo 노드 구현
  - 비디오 프레임 추출

- [ ] **TASK-037**: SaveVideo 노드 구현
  - 프레임 → 비디오 인코딩

- [ ] **TASK-038**: PreviewImage 노드 구현
  - WebSocket 실시간 미리보기

#### 변환 노드

- [ ] **TASK-039**: Resize 노드 구현
  - NEAREST, LINEAR, CUBIC, LANCZOS
  - 비율 또는 절대값 지정

- [ ] **TASK-040**: Crop 노드 구현
  - 좌표 기반 크롭
  - 중앙 크롭

- [ ] **TASK-041**: Rotate 노드 구현
  - 각도 회전
  - 90/180/270 퀵 회전

- [ ] **TASK-042**: Flip 노드 구현
  - 수평/수직 플립

- [ ] **TASK-043**: ColorConvert 노드 구현
  - BGR, RGB, HSV, LAB, Gray 변환

#### 필터 노드

- [ ] **TASK-044**: GaussianBlur 노드 구현
- [ ] **TASK-045**: MedianBlur 노드 구현
- [ ] **TASK-046**: BilateralFilter 노드 구현
- [ ] **TASK-047**: BoxFilter 노드 구현
- [ ] **TASK-048**: Sharpen 노드 구현

#### 에지 검출 노드

- [ ] **TASK-049**: Canny 노드 구현
- [ ] **TASK-050**: Sobel 노드 구현
- [ ] **TASK-051**: Laplacian 노드 구현
- [ ] **TASK-052**: Scharr 노드 구현

#### 형태학 노드

- [ ] **TASK-053**: Erode 노드 구현
- [ ] **TASK-054**: Dilate 노드 구현
- [ ] **TASK-055**: MorphOpen 노드 구현
- [ ] **TASK-056**: MorphClose 노드 구현
- [ ] **TASK-057**: MorphGradient 노드 구현

#### 임계값 노드

- [ ] **TASK-058**: Threshold 노드 구현
  - Binary, Binary_Inv, Trunc, ToZero, Otsu

- [ ] **TASK-059**: AdaptiveThreshold 노드 구현
  - Mean, Gaussian

#### 색상 처리 노드

- [ ] **TASK-060**: Brightness 노드 구현
- [ ] **TASK-061**: Contrast 노드 구현
- [ ] **TASK-062**: HueSaturation 노드 구현
- [ ] **TASK-063**: ColorBalance 노드 구현
- [ ] **TASK-064**: Invert 노드 구현
- [ ] **TASK-065**: HistogramEqualize 노드 구현

#### 합성 노드

- [ ] **TASK-066**: Blend 노드 구현
  - Normal, Add, Multiply, Screen, Overlay

- [ ] **TASK-067**: AlphaComposite 노드 구현
- [ ] **TASK-068**: MaskApply 노드 구현
- [ ] **TASK-069**: ChannelSplit 노드 구현
- [ ] **TASK-070**: ChannelMerge 노드 구현

#### 기하 변환 노드

- [ ] **TASK-071**: PerspectiveTransform 노드 구현
- [ ] **TASK-072**: AffineTransform 노드 구현
- [ ] **TASK-073**: Remap 노드 구현

#### 분석 노드

- [ ] **TASK-074**: FindContours 노드 구현
- [ ] **TASK-075**: DrawContours 노드 구현
- [ ] **TASK-076**: Histogram 노드 구현
- [ ] **TASK-077**: ImageInfo 노드 구현

#### 그리기 노드

- [ ] **TASK-078**: DrawText 노드 구현
- [ ] **TASK-079**: DrawRectangle 노드 구현
- [ ] **TASK-080**: DrawCircle 노드 구현
- [ ] **TASK-081**: DrawLine 노드 구현

#### 유틸리티 노드

- [ ] **TASK-082**: ImageMath 노드 구현
  - Add, Subtract, Multiply, Divide, Max, Min

- [ ] **TASK-083**: SplitBatch 노드 구현
- [ ] **TASK-084**: MergeBatch 노드 구현
- [ ] **TASK-085**: EmptyImage 노드 구현

---

## Phase 5: 테스트 및 검증

### 5.1 테스트 환경 구축

- [ ] **TASK-086**: pytest 설정
  - tests/ 디렉토리 구조
  - conftest.py 작성
  - 테스트 픽스처

### 5.2 단위 테스트

- [ ] **TASK-087**: I/O 노드 테스트
- [ ] **TASK-088**: 변환 노드 테스트
- [ ] **TASK-089**: 필터 노드 테스트
- [ ] **TASK-090**: 에지 검출 노드 테스트
- [ ] **TASK-091**: 형태학 노드 테스트
- [ ] **TASK-092**: 임계값 노드 테스트
- [ ] **TASK-093**: 색상 처리 노드 테스트
- [ ] **TASK-094**: 합성 노드 테스트

### 5.3 통합 테스트

- [ ] **TASK-095**: 그래프 실행 테스트
- [ ] **TASK-096**: 캐싱 테스트
- [ ] **TASK-097**: 서버 API 테스트

### 5.4 성능 테스트

- [ ] **TASK-098**: 4K 이미지 처리 시간 측정
- [ ] **TASK-099**: 메모리 사용량 측정
- [ ] **TASK-100**: 대용량 배치 처리 테스트

---

## Phase 6: 마무리

### 6.1 문서화

- [ ] **TASK-101**: README.md 업데이트
- [ ] **TASK-102**: 노드 API 문서 작성
- [ ] **TASK-103**: 설치 가이드 작성
- [ ] **TASK-104**: 사용 예시 워크플로우 작성

### 6.2 배포 준비

- [ ] **TASK-105**: Docker 이미지 생성
- [ ] **TASK-106**: PyPI 패키지 준비 (선택)
- [ ] **TASK-107**: GitHub Release 생성

### 6.3 프론트엔드 검토

- [ ] **TASK-108**: 웹 UI 빌드 테스트
- [ ] **TASK-109**: AI 관련 UI 요소 제거/숨김
- [ ] **TASK-110**: 새 노드 카테고리 UI 확인

---

## 작업 우선순위

### 즉시 실행 (Critical Path)
1. TASK-001 ~ TASK-009: 디렉토리/파일 삭제
2. TASK-014 ~ TASK-016: 의존성 정리
3. TASK-018 ~ TASK-020: 실행 엔진 수정
4. TASK-029 ~ TASK-032: OpenCV 래퍼 구현
5. TASK-033 ~ TASK-038: 기본 I/O 노드

### 높은 우선순위
6. TASK-039 ~ TASK-048: 변환/필터 노드
7. TASK-021 ~ TASK-023: 타입/폴더 수정

### 중간 우선순위
8. TASK-049 ~ TASK-085: 나머지 노드 구현
9. TASK-086 ~ TASK-100: 테스트

### 낮은 우선순위
10. TASK-101 ~ TASK-110: 문서화/배포

---

## 예상 작업량

| Phase | 태스크 수 | 예상 난이도 |
|-------|----------|------------|
| Phase 1: 정리 | 13 | 낮음 |
| Phase 2: 의존성 | 4 | 낮음 |
| Phase 3: 핵심 수정 | 11 | 높음 |
| Phase 4: 새 기능 | 57 | 중간 |
| Phase 5: 테스트 | 15 | 중간 |
| Phase 6: 마무리 | 10 | 낮음 |
| **총합** | **110** | - |

---

## 체크리스트

### 삭제 완료 확인
- [ ] torch import 없음
- [ ] transformers import 없음
- [ ] diffusers import 없음
- [ ] safetensors import 없음
- [ ] 모든 AI 모델 코드 제거

### 빌드 확인
- [ ] pip install -e . 성공
- [ ] python main.py 시작 성공
- [ ] 웹 UI 로드 성공
- [ ] 노드 목록 표시 성공

### 기능 확인
- [ ] 이미지 로드 성공
- [ ] 이미지 저장 성공
- [ ] 노드 연결 성공
- [ ] 그래프 실행 성공
- [ ] 캐싱 동작 확인

# 박원우 - 포트폴리오

## 🎯 전문가 개요

수학적 신호 처리와 현대 AI 기술을 결합한 독특한 하이브리드 접근법을 가진 경험 많은 컴퓨터 비전 및 딥러닝 전문가입니다. 산업 응용 분야에서 15년 이상의 경험을 보유하고 있으며, 8년 이상 프로덕션 딥러닝 시스템에 집중해 왔습니다.

## 🚀 핵심 역량

### 딥러닝 + OpenCV 하이브리드 비전
- 수학적 영상처리(OpenCV) + 데이터 주도 비전(AI)
- 수학적 데이터 증강 기법
- 고속 결함 후보 검색 + 라벨링 시스템
- SAM 모델 + 윤곽선 분석 통합
- DefectPose - 새로운 키포인트 기반 측정
- SRSM + Deep InPaint를 통한 결함 제거

### 신호 처리 및 영상 분석
- 1D 전처리: LPF, HPF, Butterworth, Chebyshev 필터
- 1D 신호를 영상으로 변환: 스펙트로그램, 스칼로그램, 바이스펙트럼, WVD
- 비파괴 검사를 위한 펄스 위상 열화상법
- 시계열 분석: 악성코드 검출, 센서 데이터, 반도체 센서, 배터리 초음파 신호
- 윈도우 함수: Hamming, Hann 데이터 증강 기법

### 공간 vs 스펙트럼 도메인 처리
- 이상 검출을 위한 스펙트럴 잔차 현저성
- 위상 일치 엣지 검출
- 템플릿 매칭을 위한 위상 전용 상관관계
- 조명 정규화를 위한 호모모픽 필터링
- 위상 불일치 분석
- 노이즈 제거를 위한 노치 필터링
- 열 분석을 위한 위상 열화상법

## 💻 기술 스택

### 프로그래밍 및 개발
- 주요 언어: Python(10), C++(10)
- 보조 언어: C#.NET, Java (SCJP 인증)
- GUI 프레임워크: PyQt, TkInter, WinForms, VCL, OpenFrameworks
- IDE: Xcode, Visual Studio, Cursor, Jupyter, CMake
- 라이브러리: OpenCV, Pandas, PyTorch, TorchVision, Matlab, ImageMagick, FFmpeg, Sox, GnuParallel, MCP-server
- Apple App Store: 10개 이상의 OSX 앱 출시 및 배포

### 하드웨어 및 플랫폼
- GPU 컴퓨팅: CUDA (GTX750Ti, RTX4090, A5000, 멀티노드)
- 운영체제: Ubuntu (14.04LTS ~ 22.04LTS)
- 엣지 컴퓨팅: Jetson TK1, Jetson Nano, Orin
- 임베디드: RaspberryPi4, Raspbian, Arduino+Processing
- Apple Silicon: M1 Mac, MPS GPGPU 최적화
- 모바일: iOS, macOS 개발

## 🎨 혁신적인 프로젝트 및 특허

### DefectPose (2022년 특허 출원)
문제: 기존 결함 검출은 정밀한 측정 기능이 부족
해결책: 정확한 치수 분석을 위한 딥러닝 기반 키포인트 검출

주요 특징:
- 길이, 각도, 면적, 반지름, 개수 측정을 위한 영상에서 키포인트 검출
- AOI(관심 영역) 애플리케이션을 위한 안정적인 키포인트 검출
- 교차점, 모서리점, 고립점, 끝점 검출
- Keypoints R-CNN과 HR-Net (TorchVision) 통합
- 하이브리드 접근법: 데이터 주도 + 수학적 영상처리

```python
class DefectPose:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.keypoint_model = self.load_keypoint_model()
    
    def detect_keypoints(self, image):
        # 측정을 위한 키포인트 검출
        keypoints = self.keypoint_model(image)
        return self.calculate_measurements(keypoints)
```

### DefectCutout (2022년 특허 출원)
문제: 표준 Cutout 증강이 영상의 작은 결함을 파괴함
해결책: 지능형 결함 인식 cutout 증강

주요 특징:
- 소형 결함 영상을 위한 결함 회피 cutout
- 객체 검출 데이터셋 기반 결함 인식 cutout
- 분류 데이터셋을 위한 수학적 결함 인식 cutout
- 하이브리드 접근법: 데이터 주도 + 수학적 영상처리

```python
class DefectCutout:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def smart_cutout(self, image, defect_mask):
        # 결함 영역 잘라내기 방지
        safe_regions = self.calculate_safe_regions(defect_mask)
        return self.apply_cutout(image, safe_regions)
```

### Video4CNN - 쉬운 빅데이터 생성
문제: 산업 응용을 위한 훈련 데이터 부족
해결책: 실제 샘플에서 자동화된 데이터 생성

주요 특징:
- 제어된 조명, 초점, 3D 회전을 통한 실제 샘플 촬영
- 비디오를 프레임 영상으로 변환
- 인간 참여형 필터링
- GradCAM 기반 과적합 검출 및 일반화 성능 검증

```python
class Video4CNN:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def generate_dataset(self, video_path):
        frames = self.extract_frames(video_path)
        filtered_frames = self.human_filter(frames)
        return self.validate_with_gradcam(filtered_frames)
```

### TSNE4Labeling - 비지도 품질 관리
문제: 지도 학습 데이터셋의 수동 라벨링 오류
해결책: 인간 라벨링을 위한 비지도 검증 도구

주요 특징:
- 지도 학습 인간 라벨링 데이터의 비지도 검증
- 고객 UX 스타일 마케팅 기술 적용
- 잘못 라벨링된 데이터 검출을 위한 실시간 상호작용
- 다중 클래스 데이터셋 호환성
- [데모 비디오](https://youtu.be/5vrvsiVO00k?si=Q1dkNDK8Q8pHtI30)

```python
class TSNE4Labeling:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    def detect_mislabeled(self, features, labels):
        tsne_features = self.apply_tsne(features)
        return self.find_outliers(tsne_features, labels)
```

### SRSM_InPaint - 순차적 결함 검출
문제: 단일 영상의 다중 결함은 반복적 처리 필요
해결책: 지능형 인페인팅을 통한 스펙트럴 기반 결함 검출

주요 특징:
- SRSM: 주파수 도메인 객체성을 위한 스펙트럴 잔차 현저성 맵
- 현저성 검출을 위한 수학적 영상처리
- 유연한 인페인팅: OpenCV 기반 또는 딥러닝 기반
- 순차적 결함 처리 기능

```python
class SRSMInPaint:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def detect_and_inpaint(self, image):
        saliency_map = self.calculate_srsm(image)
        defect_regions = self.extract_defects(saliency_map)
        return self.sequential_inpaint(image, defect_regions)
```

### SAM + OpenCV 통합
Meta의 Segment Anything Model과 고전적 CV의 결합

응용 분야:
- SAM + InPainting을 통한 결함 제거
- 향상된 분할을 위한 SAM + YOLO Box
- 형상 분석을 위한 SAM + 윤곽선 특징
- 패턴 인식을 위한 SAM + 템플릿 매칭
- 개인정보 보호를 위한 SAM + 블러링
- SAM을 사용한 BBox를 마스크로 변환

```python
class SAMOpenCV:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.sam_model = self.load_sam_model()
    
    def sam_with_opencv(self, image, bbox):
        # SAM을 사용하여 bbox를 마스크로 변환
        mask = self.sam_model.predict(image, bbox)
        # OpenCV 연산 적용
        return self.opencv_processing(image, mask)
```

## 📚 출판물 및 콘텐츠 제작

### 기술서적
"딥러닝을 위한 푸리에 영상처리" (2023)
- 홍릉과학출판사 출간
- AI 애플리케이션을 위한 주파수 도메인 처리의 종합 가이드
- [알라딘에서 구매 가능](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=309060931)

### 온라인 활동
- YouTube 채널: [@jegalsheek](https://www.youtube.com/@jegalsheek) - 컴퓨터 비전 튜토리얼 및 데모
- Facebook 그룹: [딥러닝 & 푸리에 변환](https://www.facebook.com/groups/297004660778037) - 1000명 이상의 회원 커뮤니티
- Apple App Store: [C-Booth 앱](https://apps.apple.com/kr/app/c-booth/id6738316726?mt=12) 외 9개 애플리케이션
- 기술 블로그: [티스토리](https://wineskin.tistory.com/) - 정기적인 기술 글쓰기

## 🏭 산업 응용

### 반도체 산업
- PCB 결함 검출: X선 영상을 활용한 다층 기판 분석
- 칩 검사: 커스텀 증강을 통한 용접 결함 검출
- 필름 분석: Mask R-CNN을 사용한 반도체 필름 결함 검출
- 히터 검사: 포즈 추정을 통한 X선 기반 결함 검출

### 태양에너지 분야
- 태양광 패널 AOI: 품질 관리를 위한 가시광 영상 분석
- 태양전지 모듈: Faster R-CNN을 활용한 적외선 결함 검출
- 열 분석: 효율성 최적화를 위한 고급 열화상법

### 자동차 및 제조업
- 품질 관리 시스템: 실시간 결함 검출
- 공정 모니터링: 통계 분석 및 이상 검출
- 예측 정비: 장비 상태를 위한 센서 데이터 분석

## 🔬 연구 개발

### 신호처리 혁신
- 주파수 도메인 분석: 영상 향상을 위한 고급 푸리에 기법
- 위상 기반 처리: 위상 정보 활용의 새로운 접근법
- 하이브리드 필터링: 공간 및 주파수 도메인 방법 결합

### 머신러닝 연구
- 커스텀 아키텍처: 특정 산업 응용을 위한 맞춤형 CNN 설계
- 전이 학습: 전문 도메인을 위한 사전 훈련된 모델 적응
- 앙상블 방법: 강건한 예측을 위한 다중 모델 결합

### 컴퓨터 비전 발전
- 실시간 처리: 프로덕션 환경을 위한 최적화 기법
- 엣지 컴퓨팅: 자원 제약 장치를 위한 배포 전략
- 다중 모달 융합: 향상된 성능을 위한 다양한 센서 모달리티 결합

## 🎯 독특한 가치 제안

### 하이브리드 접근법
전통적인 수학적 영상처리와 현대적인 딥러닝 기법을 결합하는 독특한 강점:
- 강건한 솔루션: 훈련 데이터 양에 대한 의존성 감소
- 해석 가능한 결과: 수학적 기반이 설명 가능성 제공
- 효율적인 처리: 실시간 애플리케이션을 위한 최적화된 알고리즘
- 도메인 적응성: 다양한 산업 응용을 위한 유연한 솔루션

### 혁신 철학
"수학이 기반을 제공하고, 데이터가 지능을 제공하며, 엔지니어링이 솔루션을 제공한다."

## 📈 미래 방향

### 신기술
- 비전 트랜스포머: 산업 응용을 위한 어텐션 메커니즘 적응
- 멀티모달 AI: 비전, 텍스트, 센서 데이터 결합
- 엣지 AI: 임베디드 시스템을 위한 모델 최적화
- 연합 학습: 산업 응용을 위한 분산 훈련

### 연구 관심사
- 설명 가능한 AI: 중요한 응용에서 AI 결정의 투명성 확보
- 퓨샷 학습: 새로운 응용을 위한 데이터 요구사항 감소
- 지속적 학습: 변화하는 산업 조건에 모델 적응

## 🔧 코드 예제

### MPS 최적화 딥러닝

```python
import torch
import torch.nn as nn
import cv2
import numpy as np

class HybridVisionSystem:
    def __init__(self):
        # Apple Silicon 최적화를 위해 항상 MPS 사용
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self.build_hybrid_model()
        
    def build_hybrid_model(self):
        class HybridNet(nn.Module):
            def __init__(self):
                super(HybridNet, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU()
                )
                
            def forward(self, x):
                return self.conv_layers(x)
        
        return HybridNet().to(self.device)
```

### 푸리에 전처리 함수

```python
def fourier_preprocessing(self, image):
    # 푸리에 기반 전처리 적용
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # 스펙트럴 필터링
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)
    
    return magnitude, phase
```

### OpenCV + 딥러닝 통합

```python
class OpenCVDeepLearning:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def hybrid_defect_detection(self, image):
        # 고전적 전처리
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 딥러닝 추론
        tensor_img = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        with torch.no_grad():
            features = self.model(tensor_img.unsqueeze(0))
        
        # 결과 결합
        return self.combine_classical_and_dl(edges, features)
```

---

## 📞 연락하기

협업이나 제 작업에 대해 더 자세히 알고 싶으시다면:

- 이메일: bemore@kakao.com
- YouTube: [@sheekjegal](https://www.youtube.com/@sheekjegal)
- Facebook: [컴퓨터 비전 커뮤니티](https://www.facebook.com/groups/297004660778037)
- GitHub: [bemoregt](https://github.com/bemoregt)

"수학적 엄밀성과 실용적 AI 솔루션 사이의 다리 역할"
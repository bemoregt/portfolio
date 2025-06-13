# Wonwoo Park - Portfolio

## 🎯 Professional Overview

Experienced Computer Vision and Deep Learning specialist with a unique hybrid approach combining mathematical signal processing with modern AI techniques. My expertise spans 15+ years in industrial applications, with 8+ years focused on production deep learning systems.

## 🚀 Core Competencies

### Deep Learning + OpenCV Hybrid Vision
- **Mathematical Image Processing**(OpenCV) + **Data-Driven Vision** (AI)
- **Mathematical Augmentation** techniques
- **Fast Defect-Candidates Search** + **Labeling** systems
- **SAM Model** + **Contour Analysis** integration
- **DefectPose** - Novel keypoint-based measurement
- **SRSM** + **Deep InPaint** for defect removal

### Signal Processing & Image Analysis
- **1D Preprocessing**: LPF, HPF, Butterworth, Chebyshev filters
- **1D Signal to Image**: Spectrogram, Scalogram, Bispectrum, WVD
- **Pulse Phase Thermography** for non-destructive inspection
- **Time Series Analysis**: Malware detection, sensor data, semiconductor sensors, battery ultrasound signals
- **Window Functions**: Hamming, Hann augmentation techniques

### Spatial vs Spectral Domain Processing
- **Spectral Residual Saliency** for anomaly detection
- **Phase Congruency Edge** detection
- **Phase Only Correlation** for template matching
- **Homomorphic Filtering** for illumination normalization
- **Phase Discrepancy** analysis
- **Notch Filtering** for noise removal
- **Phase Thermography** for thermal analysis

## 💻 Technical Stack

### Programming & Development
- **Primary**: Python(10), C++(10)
- **Secondary**: C#.NET, Java (SCJP Certified)
- **GUI Frameworks**: PyQt, TkInter, WinForms, VCL, OpenFrameworks
- **IDEs**: Xcode, Visual Studio, Cursor, Jupyter, CMake
- **Libraries**: OpenCV, Pandas, PyTorch, TorchVision, Matlab, ImageMagick, FFmpeg, Sox, GnuParallel, MCP-server
- **Apple App Store**: 10+ published OSX apps & Deployed.

### Hardware & Platforms
- **GPU Computing**: CUDA (GTX750Ti, RTX4090, A5000, multi-node)
- **Operating Systems**: Ubuntu (14.04LTS ~ 22.04LTS)
- **Edge Computing**: Jetson TK1, Jetson Nano, Orin
- **Embedded**: RaspberryPi4, Raspbian, Arduino+Processing
- **Apple Silicon**: M1 Mac, MPS GPGPU optimization
- **Mobile**: iOS, macOS development

## 🎨 Innovative Projects & Patents

### DefectPose (Patent Filed 2022)
**Problem**: Traditional defect detection lacks precise measurement capabilities
**Solution**: Deep learning-based keypoint detection for accurate dimensional analysis

**Key Features**:
- Keypoint detection from images for length, angle, area, radius, and count measurements
- Stable keypoint detection for AOI (Area of Interest) applications
- Detection of intersection points, corner points, isolated points, and endpoints
- Integration of Keypoints R-CNN and HR-Net (TorchVision)
- Hybrid approach: Data-driven + Mathematical image processing

```python
class DefectPose:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.keypoint_model = self.load_keypoint_model()
    
    def detect_keypoints(self, image):
        # Keypoint detection for measurement
        keypoints = self.keypoint_model(image)
        return self.calculate_measurements(keypoints)
```

### DefectCutout (Patent Filed 2022)
**Problem**: Standard Cutout augmentation destroys small defects in images
**Solution**: Intelligent defect-aware cutout augmentation

**Key Features**:
- Defect-avoiding cutout for small defect images
- Object Detection dataset-based defect-aware cutout
- Mathematical defect-aware cutout for classification datasets
- Hybrid approach: Data-driven + Mathematical image processing

```python
class DefectCutout:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def smart_cutout(self, image, defect_mask):
        # Avoid cutting out defect regions
        safe_regions = self.calculate_safe_regions(defect_mask)
        return self.apply_cutout(image, safe_regions)
```

### Video4CNN - Easy Big Data Generation
**Problem**: Insufficient training data for industrial applications
**Solution**: Automated data generation from real samples

**Key Features**:
- Real sample imaging with controlled lighting, focus, and 3D rotation
- Video to frame image conversion
- Human-in-the-loop filtering
- GradCAM-based overfitting detection and generalization performance verification

```python
class Video4CNN:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def generate_dataset(self, video_path):
        frames = self.extract_frames(video_path)
        filtered_frames = self.human_filter(frames)
        return self.validate_with_gradcam(filtered_frames)
```

### TSNE4Labeling - Unsupervised Quality Control
**Problem**: Manual labeling errors in supervised learning datasets
**Solution**: Unsupervised verification tool for human labeling

**Key Features**:
- Unsupervised verification for supervised learning human labeling data
- Customer UX-style marketing technology application
- Real-time interaction for mis-labeled data detection
- Multi-class dataset compatibility
- [Demo Video](https://youtu.be/5vrvsiVO00k?si=Q1dkNDK8Q8pHtI30)

```python
class TSNE4Labeling:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    def detect_mislabeled(self, features, labels):
        tsne_features = self.apply_tsne(features)
        return self.find_outliers(tsne_features, labels)
```

### SRSM_InPaint - Sequential Defect Detection
**Problem**: Multiple defects in single image require iterative processing
**Solution**: Spectral-based defect detection with intelligent inpainting

**Key Features**:
- **SRSM**: Spectral Residual Saliency Map for frequency domain objectness
- Mathematical image processing for saliency detection
- Flexible inpainting: OpenCV-based or deep learning-based
- Sequential defect processing capability

```python
class SRSMInPaint:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def detect_and_inpaint(self, image):
        saliency_map = self.calculate_srsm(image)
        defect_regions = self.extract_defects(saliency_map)
        return self.sequential_inpaint(image, defect_regions)
```

### SAM + OpenCV Integration
**Meta's Segment Anything Model Enhanced with Classical CV**

**Applications**:
- SAM + InPainting for defect removal
- SAM + YOLO Box for improved segmentation
- SAM + Contour Features for shape analysis
- SAM + Template Matching for pattern recognition
- SAM + Blurring for privacy protection
- BBox to Mask conversion using SAM

```python
class SAMOpenCV:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.sam_model = self.load_sam_model()
    
    def sam_with_opencv(self, image, bbox):
        # Convert bbox to mask using SAM
        mask = self.sam_model.predict(image, bbox)
        # Apply OpenCV operations
        return self.opencv_processing(image, mask)
```

## 📚 Publications & Content Creation

### Technical Book
**"Fourier Image Processing for Deep Learning"** (2023)
- Published by Hongneung Science Publishing
- Comprehensive guide to frequency domain processing for AI applications
- [Available on Aladin](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=309060931)

### Online Presence
- **YouTube Channel**: [@seohopa](https://www.youtube.com/@seohopa) - Computer Vision tutorials and demonstrations
- **Facebook Group**: [Deep Learning & Fourier Transform](https://www.facebook.com/groups/297004660778037) - Community of 1000+ members
- **Apple App Store**: [C-Booth App](https://apps.apple.com/kr/app/c-booth/id6738316726?mt=12) and 9 other applications
- **Technical Blog**: [Brunch](https://brunch.co.kr/@f6cf51e0a3154dc) - Regular technical writing

## 🏭 Industrial Applications

### Semiconductor Industry
- **PCB Defect Detection**: Multi-layer board analysis with X-ray imaging
- **Chip Inspection**: Welding defect detection with custom augmentation
- **Film Analysis**: Semiconductor film defect detection using Mask R-CNN
- **Heater Inspection**: X-ray based defect detection with pose estimation

### Solar Energy Sector
- **Solar Panel AOI**: Visible light image analysis for quality control
- **Solar Cell Modules**: Infrared defect detection with Faster R-CNN
- **Thermal Analysis**: Advanced thermography for efficiency optimization

### Automotive & Manufacturing
- **Quality Control Systems**: Real-time defect detection
- **Process Monitoring**: Statistical analysis and anomaly detection
- **Predictive Maintenance**: Sensor data analysis for equipment health

## 🔬 Research & Development

### Signal Processing Innovation
- **Frequency Domain Analysis**: Advanced Fourier techniques for image enhancement
- **Phase-based Processing**: Novel approaches to phase information utilization
- **Hybrid Filtering**: Combining spatial and frequency domain methods

### Machine Learning Research
- **Custom Architectures**: Tailored CNN designs for specific industrial applications
- **Transfer Learning**: Adapting pre-trained models for specialized domains  
- **Ensemble Methods**: Combining multiple models for robust predictions

### Computer Vision Advancement
- **Real-time Processing**: Optimization techniques for production environments
- **Edge Computing**: Deployment strategies for resource-constrained devices
- **Multi-modal Fusion**: Combining different sensor modalities for enhanced performance

## 🎯 Unique Value Proposition

### Hybrid Approach
My unique strength lies in combining traditional mathematical image processing with modern deep learning techniques. This hybrid approach offers:
- **Robust Solutions**: Less dependent on training data quantity
- **Interpretable Results**: Mathematical foundation provides explainability
- **Efficient Processing**: Optimized algorithms for real-time applications
- **Domain Adaptability**: Flexible solutions for various industrial applications

### Innovation Philosophy
*"Mathematics provides the foundation, data provides the intelligence, and engineering provides the solution."*

## 📈 Future Directions

### Emerging Technologies
- **Vision Transformers**: Adapting attention mechanisms for industrial applications
- **Multimodal AI**: Combining vision, text, and sensor data
- **Edge AI**: Optimizing models for embedded systems
- **Federated Learning**: Distributed training for industrial applications

### Research Interests
- **Explainable AI**: Making AI decisions transparent in critical applications
- **Few-shot Learning**: Reducing data requirements for new applications
- **Continuous Learning**: Adapting models to changing industrial conditions

## 🔧 Code Examples

### MPS-Optimized Deep Learning

```python
import torch
import torch.nn as nn
import cv2
import numpy as np

class HybridVisionSystem:
    def __init__(self):
        # Always use MPS for Apple Silicon optimization
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
    
    def fourier_preprocessing(self, image):
        # Apply Fourier-based preprocessing
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Spectral filtering
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)
        
        return magnitude, phase
```

### OpenCV + Deep Learning Integration

```python
class OpenCVDeepLearning:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def hybrid_defect_detection(self, image):
        # Classical preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Deep learning inference
        tensor_img = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        with torch.no_grad():
            features = self.model(tensor_img.unsqueeze(0))
        
        # Combine results
        return self.combine_classical_and_dl(edges, features)
```

---

## 📞 Let's Connect

Interested in collaboration or learning more about my work?

- **Email**: bemore@kakao.com
- **YouTube**: [@sheekjegal](https://www.youtube.com/@sheekjegal)
- **Facebook**: [Computer Vision Community](https://www.facebook.com/groups/297004660778037)
- **GitHub**: [bemoregt](https://github.com/bemoregt)

*"Bridging the gap between mathematical rigor and practical AI solutions"*

# Credits and Acknowledgments

## Original Work

This project is based on the MIT-licensed YOLOv7n implementation:

### Primary Authors
- **Kin-Yiu Wong** - Original YOLOv7n implementation
- **Hao-Tang Tsui** - Original YOLOv7n implementation

### Original Repository
- **Source**: [WongKinYiu/YOLO](https://github.com/WongKinYiu/yolo)
- **License**: MIT License
- **Paper**: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

## MIT Edition Enhancements

### Enhanced Features Added
- **Production Pipeline**: Complete training → ONNX → deployment workflow
- **Performance Optimization**: Memory profiling, benchmarking, cross-platform validation
- **Automation Scripts**: One-click training, conversion, and analysis
- **Documentation**: Comprehensive setup and usage guides
- **Transfer Learning**: Optimized fine-tuning for custom datasets

### Technology Stack
- **PyTorch Lightning**: Training framework
- **ONNX Runtime**: Inference optimization
- **WandB**: Experiment tracking
- **uv**: Python environment management
- **Matplotlib/Seaborn**: Visualization and analysis

## Dependencies and Libraries

### Core Dependencies
- PyTorch 2.0+ - Deep learning framework
- PyTorch Lightning 2.5+ - Training orchestration
- ONNX 1.18+ - Model serialization
- ONNX Runtime 1.19+ - Inference engine
- NumPy - Numerical computing
- OpenCV - Computer vision operations
- Pillow - Image processing
- PyYAML - Configuration management
- OmegaConf - Advanced configuration
- tqdm - Progress bars
- psutil - System monitoring

### Development Dependencies
- pytest - Testing framework
- black - Code formatting
- isort - Import sorting
- mypy - Type checking
- matplotlib - Plotting and visualization
- seaborn - Statistical visualization
- wandb - Experiment tracking

## License Compliance

### MIT License Requirements Met
- ✅ **Attribution**: Original authors credited
- ✅ **License Notice**: MIT license preserved
- ✅ **Copyright Notice**: Original copyright maintained
- ✅ **Disclaimer**: Warranty disclaimers included

### Commercial Use
This MIT-licensed version is fully compatible with:
- Commercial applications
- Proprietary modifications
- Redistribution (with attribution)
- Private use and modification

## Citations

If you use this work in your research or commercial projects, please cite:

### Original YOLOv7 Paper
```bibtex
@article{wang2022yolov7,
  title={YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

### Original Repository
```bibtex
@software{wong2024yolo,
  title={YOLO: Official Implementation of YOLOv9, YOLOv7, YOLO-RD},
  author={Wong, Kin-Yiu and Tsui, Hao-Tang},
  url={https://github.com/WongKinYiu/yolo},
  year={2024}
}
```

## Contributing

### How to Contribute
1. Fork this repository
2. Create a feature branch
3. Make your changes with proper attribution
4. Ensure MIT license compatibility
5. Submit a pull request

### Contribution Guidelines
- Maintain MIT license compatibility
- Add proper attribution for borrowed code
- Include tests for new features
- Follow existing code style
- Update documentation as needed

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

---

**Last Updated**: 2025-07-02  
**License**: MIT  
**Contact**: See repository issues for support
# Speculative Decoding Distillation

A PyTorch-based implementation for training smaller student models to mimic larger teacher models for code generation tasks, specifically focused on Python code. This project implements knowledge distillation techniques to create efficient models capable of speculative decoding.

## 🎯 Project Overview

This project aims to create a lightweight student model that can perform **speculative decoding** by learning from a larger teacher model (CodeLlama). The core idea is to train a smaller, faster model that can generate draft sequences which are then validated by the larger teacher model, potentially speeding up inference.

### Key Components

- **Student Model**: Custom transformer architecture with multi-headed attention
- **Teacher Model**: CodeLlama-7b-Python-hf (quantized for efficiency)
- **Knowledge Distillation**: Custom loss combining cross-entropy and KL divergence
- **Dataset**: Python code from `DaniilOr/humanized_cleaned_code`

## 🏗️ Architecture

```
├── distilation_model/
│   └── studentmodel.py          # Custom transformer student model
├── tutor_model/
│   └── codellama.py            # CodeLlama teacher model wrapper
├── loss/
│   └── customloss.py           # Knowledge distillation loss function
├── dataset/
│   └── dataset.py              # Python code dataset handler
├── metric_manager/
│   └── metric_manager.py       # Training metrics tracking
├── research/                   # Experimental notebooks and scripts
└── training.py                 # Main training loop
```

### Student Model Architecture

The student model (`StudentModel`) implements:

- **Multi-headed attention mechanism** with parallel computation
- **Positional embeddings** using sinusoidal encoding
- **Custom transformer layers** optimized for code generation
- **Significantly smaller parameter count** compared to CodeLlama

### Teacher Model (CodeLlama)

- **Model**: `codellama/CodeLlama-7b-Python-hf`
- **Quantization**: 4-bit quantization for memory efficiency
- **Frozen parameters**: Used only for generating target logits
- **Custom generation**: Implements temperature-controlled generation

## 📊 Training Process

### Knowledge Distillation Loss

The training uses a weighted combination of:

- **Cross-entropy loss**: Traditional supervised learning on ground truth
- **KL divergence loss**: Knowledge transfer from teacher to student
- **Temperature scaling**: Softens probability distributions for better knowledge transfer

```python
loss = α × CrossEntropy(student_logits, targets) + (1-α) × KL(student_soft, teacher_soft)
```

## 📈 Monitoring & Metrics

The project tracks:

- **Total loss**: Combined distillation loss
- **Cross-entropy loss**: Task-specific loss
- **KL divergence loss**: Knowledge transfer loss
- **Perplexity**: Language modeling performance
- **Sample outputs**: Input/output comparisons via TensorBoard

## Research & Experiments

The `research/` directory contains experimental work:

## 📝 Current Status

### ✅ Completed

- [x] Student model architecture with multi-headed attention
- [x] Teacher model integration with CodeLlama
- [x] Knowledge distillation loss implementation
- [x] Dataset pipeline for Python code
- [x] Training loop with checkpointing
- [x] Metrics tracking and logging
- [x] Memory-efficient training with quantization

### 🚧 In Progress

- [ ] Hyperparameter tuning and optimization
- [ ] Speculative decoding inference pipeline

### 📋 To Do

#### High Priority

- [ ] **Implement speculative decoding inference**: Core functionality for using student model as draft generator
- [ ] **Hyperparameter optimization**: Learning rate scheduling, better batch sizes

#### Medium Priority

- [ ] **Dataset diversity**: Include more programming languages or code types
- [ ] **Inference optimization**: Model quantization for student model
- [ ] **Benchmarking suite**: Compare against other code generation models
- [ ] **Configuration management**: YAML/JSON config files instead of hardcoded values

#### Future Work

- [ ] **Distributed training**: Multi-GPU support
- [ ] **Model serving**: REST API for inference
- [ ] **Documentation**: API documentation and tutorials

## 🤔 Known Issues

1. **Memory constraints**: Current batch size is limited to 4 due to GPU memory
2. **Path dependencies**: Hardcoded paths need to be made configurable
3. **Limited metrics**: Only basic loss metrics, missing code-specific evaluations

<!-- ## 📚 References

- Knowledge Distillation: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
- Speculative Decoding: [Chen et al., 2023](https://arxiv.org/abs/2302.01318)
- CodeLlama: [Rozière et al., 2023](https://arxiv.org/abs/2308.12950) -->

### Contributors

- **Onyxia**: Providing GPU infrastructure for model training and experimentation

---

*This project implements knowledge distillation for code generation models with the goal of enabling speculative decoding for faster inference.*

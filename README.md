# Gender Bias Research Pipeline

A comprehensive, modular Python pipeline for analyzing gender bias in text embeddings after removing explicit gender cues. This research demonstrates that semantic gender information persists in embeddings even after scrubbing explicit gender markers from forum posts.

## 🔬 Research Overview

This project implements a systematic approach to study gender bias in text embeddings:

1. **Auto-labeling**: ≈1M forum posts tagged as FEMALE, MALE, or NEUTRAL using GPT-4o-mini
2. **Text Scrubbing**: Explicit gender cues replaced with neutral placeholders like [PRONOUN], [NAME], [FAMILY_MEMBER]
3. **Embedding Generation**: Multiple embedding methods (BERT CLS/Mean, SBERT, OpenAI Ada) applied to scrubbed text
4. **Evaluation**: Downstream classification performance demonstrates embedding quality at preserving semantic gender information

### Key Findings

- **OpenAI Ada embeddings**: 79% classification accuracy on scrubbed text
- **BERT-Mean embeddings**: 76% accuracy
- **SBERT embeddings**: 75% accuracy
- **Human validation**: 92% agreement with GPT-4o-mini labels on 500-post sample

## 📁 Project Structure

```
gender_bias_research/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                     # Package installation
├── config/
│   └── config.yaml              # Configuration parameters
├── src/                         # Core source code
│   ├── data_processing/         # Data cleaning and preprocessing
│   │   ├── gender_labeling.py   # GPT-4o-mini labeling pipeline
│   │   └── text_scrubbing.py    # Gender cue removal/masking
│   ├── embeddings/              # Text embedding generation
│   │   ├── bert_embeddings.py   # BERT CLS/Mean embeddings
│   │   ├── sbert_embeddings.py  # Sentence-BERT embeddings  
│   │   └── openai_embeddings.py # OpenAI Ada embeddings
│   ├── modeling/                # Machine learning models
│   │   └── classifiers.py       # Classification models & evaluation
│   └── utils/                   # Utility functions
│       ├── config_loader.py     # Configuration management
│       ├── file_io.py          # File handling utilities
│       └── logging_utils.py     # Logging configuration
├── scripts/                     # Main execution scripts
│   ├── 01_label_data.py        # Stage 1: Auto-labeling with GPT-4o-mini
│   ├── 02_scrub_text.py        # Stage 2: Gender cue removal
│   ├── 03_generate_embeddings.py # Stage 3: Embedding generation
│   └── 04_train_models.py       # Stage 4: Model training & evaluation
├── tests/                       # Unit tests
│   └── test_text_scrubbing.py  # Example unit tests
├── data/                        # Data storage (create these directories)
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned/labeled data
│   ├── embeddings/             # Generated embeddings
│   └── models/                 # Trained models
└── docs/                       # Additional documentation
```

## 🚀 Quick Start

### Installation

#### Modern Setup (Recommended - using uv)
```bash
# Clone the repository
git clone <repository-url>
cd gender_bias_research

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
uv pip install -e ".[dev,notebooks]"

# Set up pre-commit hooks
pre-commit install
```

#### Traditional Setup (alternative)
```bash
# Clone the repository
git clone <repository-url>
cd gender_bias_research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,notebooks]"
```

### Setup

1. **Configure API Keys** (required for OpenAI functionality):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Prepare Data Directory**:
   ```bash
   mkdir -p data/{raw,processed,embeddings,models}
   ```

3. **Place Your Data**: 
   - Put your raw forum posts CSV in `data/raw/`
   - Ensure it has a text column (default name: `raw_cat`)

### Running the Pipeline

The pipeline consists of 4 main stages:

#### Quick Pipeline Execution (using Makefile)
```bash
# Set up data directories
make setup-data-dirs

# Copy your data to data/raw/EJR_gender_dataset_Jan2018.csv

# Run complete pipeline
make run-pipeline

# Or run individual stages
make label-data
make scrub-text  
make generate-embeddings
make train-models
```

#### Stage 1: Gender Labeling
```bash
python scripts/01_label_data.py \
    --input data/raw/EJR_gender_dataset_Jan2018.csv \
    --output data/processed/labeled_posts.csv \
    --text-column raw_cat \
    --validate
```

**What this does**: Uses GPT-4o-mini with temperature=0 and chain-of-thought reasoning to label each post as FEMALE, MALE, or NEUTRAL.

#### Stage 2: Text Scrubbing
```bash
python scripts/02_scrub_text.py \
    --input data/processed/labeled_posts.csv \
    --output data/processed/scrubbed_posts.csv \
    --analyze --validate
```

**What this does**: Removes explicit gender cues (pronouns, names, family terms, titles) and replaces them with neutral placeholders.

#### Stage 3: Generate Embeddings
```bash
python scripts/03_generate_embeddings.py \
    --input data/processed/scrubbed_posts.csv \
    --methods bert_cls bert_mean openai_ada \
    --output-dir data/embeddings/
```

**What this does**: Generates embeddings using multiple methods (BERT CLS, BERT Mean, OpenAI Ada) on the scrubbed text.

#### Stage 4: Train and Evaluate Models
```bash
python scripts/04_train_models.py \
    --embedding-dir data/embeddings/ \
    --output-dir data/models/ \
    --compare-methods
```

**What this does**: Trains classification models on each embedding type to evaluate how well they preserve semantic gender information.

## 🔧 Configuration

The pipeline is configured via `config/config.yaml`. Key settings include:

```yaml
# API Configuration
api:
  openai:
    model_name: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
    temperature: 0

# Model Configuration  
models:
  bert:
    model_name: "bert-base-uncased"
    max_length: 512
  
# Gender placeholders for text scrubbing
placeholders:
  pronouns: ["[PRONOUN]", "[PRONOUN_POSSESSIVE]", "[PRONOUN_OBJECT]"]
  names: "[NAME]"
  family_members: "[FAMILY_MEMBER]"
```

## 📊 Understanding the Results

### Gender Labeling Quality
- The pipeline validates labeling quality by re-labeling a random sample
- Expected agreement rate: ~92% based on human validation
- Low-confidence labels are flagged for manual review

### Text Scrubbing Effectiveness
- Analysis shows average word reduction and replacement counts
- Validation checks for missed gender cues
- Examples of heavily modified texts are saved for inspection

### Embedding Comparison
- Classification accuracy indicates how well embeddings preserve semantic gender information
- Higher accuracy = better semantic preservation after scrubbing
- Comparison across multiple embedding methods provides robust evaluation

## 🧪 Development & Testing

### Code Quality
We use modern Python tooling for code quality:

```bash
# Lint code with ruff
make lint

# Format code with ruff  
make format

# Type checking with mypy
make type-check

# Run all quality checks
make lint && make type-check
```

### Testing
```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Run specific test
pytest tests/test_text_scrubbing.py -v
```

Example test for text scrubbing:
```python
def test_scrub_pronouns(self):
    """Test that pronouns are correctly replaced with placeholders."""
    test_text = "He said she would help him with his work."
    result = self.scrubber.scrub_text(test_text)
    
    self.assertNotIn('He', result)
    self.assertNotIn('she', result)
    self.assertIn('[PRONOUN]', result)
```

## 📈 Advanced Usage

### Custom Embedding Methods
Add new embedding methods by creating modules in `src/embeddings/`:

```python
class CustomEmbedder:
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        # Your embedding logic here
        pass
```

### Hyperparameter Tuning
```python
from src.modeling.classifiers import EmbeddingClassifier

classifier = EmbeddingClassifier()
tuning_results = classifier.hyperparameter_tuning(
    X_train, y_train, 
    model_name='Logistic Regression'
)
```

### Batch Processing
For large datasets, use the built-in batch processing:
```python
# Automatic progress saving every 1000 items
labeled_df = labeler.label_batch(
    texts=texts,
    batch_size=1000,
    save_progress=True
)
```

## 🛠️ Error Handling and Debugging

### Common Issues

1. **API Rate Limits**: The pipeline includes automatic rate limiting for OpenAI API calls
2. **Memory Issues**: Large datasets are processed in batches with progress saving
3. **Missing Dependencies**: Check `requirements.txt` and install missing packages

### Exception Handling
The pipeline includes comprehensive error handling:
```python
try:
    result = self.label_single_text(text)
except Exception as e:
    logger.error(f"Error labeling text: {e}")
    return {"gender": "NEUTRAL", "reasoning": f"Error: {e}", "confidence": "low"}
```

### Logging
Detailed logging helps track progress and debug issues:
```bash
# Logs show progress, errors, and performance metrics
INFO - Processed 1000/10000 texts (10.0%)
INFO - BERT CLS embedding generation complete: (10000, 768)
```

## 📚 Research Methodology

### Minimizing Hallucinations
- **Temperature=0**: Deterministic outputs from GPT-4o-mini
- **Chain-of-thought**: Requires reasoning field in JSON response
- **Explicit examples**: Provides clear labeling guidelines
- **Validation sampling**: Human verification on random subset

### Gender Cue Removal Strategy
- **Comprehensive patterns**: Covers pronouns, names, family terms, titles
- **Word boundaries**: Prevents over-replacement (e.g., "theory" → "the[PRONOUN]y")
- **Case sensitivity**: Handles various capitalizations
- **Validation**: Checks for missed gender cues

### Embedding Evaluation
- **Multiple methods**: BERT CLS, BERT Mean, SBERT, OpenAI Ada
- **Downstream task**: Gender classification on scrubbed text
- **Cross-validation**: Robust performance estimation
- **Baseline comparison**: Clear performance differences between methods

## 🤝 Contributing

1. **Code Style**: Follows PEP8 guidelines
2. **Documentation**: Comprehensive docstrings and type hints
3. **Testing**: Unit tests for key functionality
4. **Modularity**: Clean separation of concerns

Example contribution:
```python
def new_embedding_method(self, texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using new method.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        2D numpy array of embeddings
        
    Raises:
        ValueError: If texts list is empty
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    # Implementation here
    pass
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and embedding APIs
- Hugging Face for BERT models and transformers library
- The open-source community for the foundational tools

## 📧 Contact

For questions about this research or implementation:
- Create an issue in this repository
- Email: okar.yigit@gmail.com

---

**Note**: This pipeline is designed for research purposes to understand gender bias in text embeddings. Please ensure ethical use and consider privacy implications when working with user-generated content.

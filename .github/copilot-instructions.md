# Copilot Instructions for textual-bayes

## 🏗 Project Architecture

This is a research framework for Bayesian methods over LLM outputs/prompts, built on **Hydra** for configuration and **textgrad** for text-based differentiation.

### Core Components
- **Entry Point**: `main.py` is the universal dispatcher. It initializes Hydra, sets seeds/logging, loads data, and invokes methods (`method/`).
- **Configuration**: Uses **Hydra** (`conf/`). All run parameters (data, method hyperparameters, engines) are defined in `.yaml` files and overridden via CLI.
- **Methods**: Algorithms reside in `method/`.
  - `bayes_over_output.py`: Bayesian inference over output space.
  - `bayes_over_prompt.py`: Bayesian optimization over prompts ("BoP").
  - `baseline/`: Baseline methods managed via `method/baseline/runner.py`.
- **Data**: `utils/data.py` contains a `dataset_classes` registry mapping string names (e.g., "mmlu", "gsm8k") to `Dataset` classes.
- **TextGrad Extensions**: `method/tgext/` contains custom overrides of `textgrad` classes (e.g., `LLMCall` with `engine_kwargs` support).

## 🧑‍💻 Development Workflows

### Configuration & Running
- **Hydra Syntax**: Use `+group=option` to compose configs.
  ```bash
  # Example: Run BOP method on MMLU
  python main.py +method=bop +data=mmlu method.steps=10
  ```
- **New Methods**: Add a new config file in `conf/method/` and handle the string in `main.py`.

### Testing
- **Integration Tests**: Tests in `tests/` (e.g., `test_cmds.py`) primarily run end-to-end integration tests by invoking `main.py` via `subprocess`.
  - Use `HYDRA_FULL_ERROR=1` for debugging.
  - Test logic checks exit codes rather than internal state often.
- **Run Tests**: `pytest -s`

### Data Pipeline
- **Adding Datasets**:
  1. Create a class inheriting from `QaDataset` in `utils/data.py`.
  2. Implement `__getitem__` to return a dict with `question`, `answers`, `label`.
  3. Register it in `dataset_classes` dict at the bottom of the file.
  4. Create a corresponding config in `conf/data/`.

## 🧩 Key Patterns & Conventions

### TextGrad Integration
- **Custom LLM Calls**: Do NOT use `textgrad.autograd.llm_ops.LLMCall` directly. Use `method.tgext.llm_call.LLMCall`.
  - *Why*: The custom class allows passing `engine_kwargs` (e.g., `temperature`, `logprobs`) to the underlying engine, which is critical for Bayesian sampling.

### Data Loading
- **Structure**: Datasets return a dictionary.
  - `question`: The prompt content.
  - `answers`: List of choices or valid answers.
  - `label`: Index or correct text.
- **Batches**: Loaders use `collate_fn=lambda x: x` to yield lists of dicts, not stacked tensors.

### Logging & Output
- **Output Directory**: Automatically generated via `namesgenerator`. Resolves to `outputs/YYYY-MM-DD_HH-MM-SS_randomname/`.
- **System Logging**: Use `logging.getLogger("ComponentName")`. Hydra captures stdio to `log.txt` in the output dir.

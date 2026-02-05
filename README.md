# textual bayes

## Installation:
Create a conda environment using:

```
conda create -n bbt python=3.12.0
conda activate bbt
```

Install dependencies
```
pip install -r requirements.txt
```

Download the datasets:
```
cd datasets
bash prepare_datasets.sh
cd ..
```

## Run
Before you start running `bop`, make sure you have OpenAI and Together AI keys. Then set the following environment variables.
```
export OPENAI_API_KEY="your open ai key"
export TOGETHER_API_KEY="your together ai key"
```
Alternately, you can add your keys to a `.env` file in the project root:
```
OPENAI_API_KEY="your open ai key"
TOGETHER_API_KEY="your together ai key"
```

These are the commands to reproduce the experiments from the paper:

AIME:
```
# Paraphrasing
python main.py \
    +data=aime \
    +method=baselines \
    method.perturber_select=paraphrasing \
    method.aggregator_select=FrequencyUQ \
    method.engine.model_name=gpt-4o

# System-Message
python main.py \
    +data=aime \
    +method=baselines \
    method.perturber_select=sys_msg \
    method.aggregator_select=FrequencyUQ \
    method.engine.model_name=gpt-4o

# Chain-of-Thought
python main.py \
    +data=aime \
    +method=bop \
    method/mcmc=langevin \
    method.model.engine=gpt-4o \
    method.steps=0 \
    method.burn_in=0 \
    method.num_repeats=10

# TextGrad
python main.py \
    +data=aime \
    +method=bop \
    method/mcmc=langevin \
    method.model.engine=gpt-4o \
    method.steps=60 \
    method.burn_in=60 \
    method.num_repeats=10

# MHLP (Ours)
python main.py \
    +data=aime \
    +method=bop \
    method.mcmc.likelihood_beta=10 \
    method.model.engine=gpt-4o \
    method.steps=60 \
    method.burn_in=6 \
    method.thinning=6 \
    method.num_repeats=10
```

SimpleQA:
```
# Paraphrasing
python main.py \
    +data=simpleqa \
    +method=baselines \
    method.perturber_select=paraphrasing \
    method.aggregator_select=SemanticFrequencyUQ \
    method.engine.model_name=gpt-4o

# System-Message
python main.py \
    +data=simpleqa \
    +method=baselines \
    method.perturber_select=sys_msg \
    method.aggregator_select=SemanticFrequencyUQ \
    method.engine.model_name=gpt-4o

# Chain-of-Thought
python main.py \
    +data=simpleqa \
    +method=bop \
    method/mcmc=langevin \
    method.model.engine=gpt-4o \
    method.steps=0 \
    method.burn_in=0 \
    method.num_repeats=10

# TextGrad
python main.py \
    +data=simpleqa \
    +method=bop \
    method/mcmc=langevin \
    method.model.engine=gpt-4o \
    method.steps=60 \
    method.burn_in=60 \
    method.num_repeats=10

# MHLP (Ours)
python main.py \
    +data=simpleqa \
    +method=bop \
    method.mcmc.likelihood_beta=100 \
    method.model.engine=gpt-4o \
    method.steps=60 \
    method.burn_in=6 \
    method.thinning=6 \
    method.num_repeats=10
```

QASPER (no context for unanswerable prompts):
```
# Paraphrasing
python main.py \
    +data=qasper \
    +method=baselines \
    method.perturber_select=paraphrasing \
    method.aggregator_select=SemanticFrequencyUQ

# System-Message
python main.py \
    +data=qasper \
    +method=baselines \
    method.perturber_select=sys_msg \
    method.aggregator_select=SemanticFrequencyUQ

# Chain-of-Thought
python main.py \
    +data=qasper \
    +method=bop \
    method.steps=0 \
    method.burn_in=0 \
    method.thinning=1 \
    method.num_chains=1 \
    method.num_repeats=10

# TextGrad
python main.py \
    +data=qasper \
    +method=bop \
    method/mcmc=langevin \
    method.steps=20 \
    method.burn_in=20 \
    method.thinning=1 \
    method.num_chains=1 \
    method.num_repeats=10

# MHLP (Ours)
python main.py \
    +data=qasper \
    +method=bop \
    method.steps=20 \
    method.burn_in=2 \
    method.thinning=2 \
    method.num_chains=1 \
    method.num_repeats=1
```

QASPER (random context for unanswerable prompts):
```
# Paraphrasing
python main.py \
    +data=qasper-random \
    +method=baselines \
    method.perturber_select=paraphrasing \
    method.aggregator_select=SemanticFrequencyUQ

# System-Message
python main.py \
    +data=qasper-random \
    +method=baselines \
    method.perturber_select=sys_msg \
    method.aggregator_select=SemanticFrequencyUQ

# Chain-of-Thought
python main.py \
    +data=qasper-random \
    +method=bop \
    method.steps=0 \
    method.burn_in=0 \
    method.thinning=1 \
    method.num_chains=1 \
    method.num_repeats=10

# TextGrad
python main.py \
    +data=qasper-random \
    +method=bop \
    method/mcmc=langevin \
    method.steps=20 \
    method.burn_in=20 \
    method.thinning=1 \
    method.num_chains=1 \
    method.num_repeats=10

# MHLP (Ours)
python main.py \
    +data=qasper-random \
    +method=bop \
    method.steps=20 \
    method.burn_in=2 \
    method.thinning=2 \
    method.num_chains=1 \
    method.num_repeats=1
```

## Development
Please run the following before merging a PR:
```
# Code formatting:
black -l 100 .

# Import sorting:
isort . --line-length 100
```

You can run tests with `pytest`:
```
pytest -s
```

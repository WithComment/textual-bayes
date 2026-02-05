import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger("Engine")

# Default vLLM model
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-4B"


class Engine(ABC):
    """Abstract Base Class for language model engines."""

    @abstractmethod
    def generate(
        self, messages: List[Dict[str, str]], logprobs: bool = False, **kwargs
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        """
        Generates text based on input messages and optionally returns log probabilities.
        Args:
            messages: A list of message dictionaries.
            logprobs: A boolean value that indicates if engine should return logprobs.
        Returns:
            A tuple containing:
            - A list of generated text strings (one for each choice).
            - An optional list containing log probability information for each choice.
        """
        pass


class OpenAIEngine(Engine):
    def __init__(self, model_name: str):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model_name

    def generate(
        self, messages: List[Dict[str, str]], logprobs: bool = False, **kwargs
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        ret = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            logprobs=logprobs,
            **kwargs,
        )

        texts = [choice.message.content for choice in ret.choices]
        logprobs_list = None
        if logprobs:
            logprobs_list = []
            for choice in ret.choices:
                l = [a.logprob for a in choice.logprobs.content]
                logprobs_list.append(l)
        return texts, logprobs_list


class VLLMEngine(Engine):
    """
    vLLM-based engine for generation.
    
    Uses the singleton vLLM engine manager to share the same model instance
    across different parts of the codebase.
    """
    
    _shared_client = None
    _shared_model = None
    
    def __init__(self, model_name: str = DEFAULT_VLLM_MODEL, **llm_config):
        """
        Initialize the vLLM engine.
        
        Args:
            model_name: The model to load (default: Qwen/Qwen3-4B)
            **llm_config: Additional arguments passed to vLLM's LLM class
        """
        self.model = model_name
        self.llm_config = llm_config
        self._ensure_client()
    
    def _ensure_client(self):
        """Ensure the vLLM client is initialized (singleton pattern)."""
        # Use class-level sharing to avoid loading the model multiple times
        if VLLMEngine._shared_client is None or VLLMEngine._shared_model != self.model:
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError(
                    "If you'd like to use vLLM models, please install the vllm package by running "
                    "`pip install vllm` or `pip install textgrad[vllm]`."
                )
            
            logger.info(f"Initializing vLLM with model: {self.model}")
            VLLMEngine._shared_client = LLM(self.model, **self.llm_config)
            VLLMEngine._shared_model = self.model
            self.tokenizer = VLLMEngine._shared_client.get_tokenizer()
            logger.info(f"vLLM initialized successfully")
        else:
            self.tokenizer = VLLMEngine._shared_client.get_tokenizer()
    
    @property
    def client(self):
        return VLLMEngine._shared_client
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build the chat prompt using the tokenizer's chat template."""
        chat_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return chat_str
    
    def generate(
        self, messages: List[Dict[str, str]], logprobs: bool = False, **kwargs
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        """
        Generate text from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            logprobs: Whether to return logprobs
            **kwargs: Additional generation parameters (n, temperature, max_tokens, top_p)
            
        Returns:
            Tuple of (list of generated texts, optional list of logprobs per generation)
        """
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError(
                "If you'd like to use vLLM models, please install the vllm package."
            )
        
        n = kwargs.pop("n", 1)
        temperature = kwargs.pop("temperature", 1)
        max_tokens = kwargs.pop("max_tokens", 2000)
        top_p = kwargs.pop("top_p", 1.0)
        
        # Build the chat prompt
        chat_str = self._build_chat_prompt(messages)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            logprobs=1 if logprobs else None,
        )
        
        # Generate
        response = self.client.generate([chat_str], sampling_params)
        
        texts = []
        logprobs_list = None
        
        if logprobs:
            logprobs_list = []
        
        for output in response[0].outputs:
            texts.append(output.text)
            
            if logprobs and output.logprobs:
                # Extract logprobs
                token_logprobs = []
                for logprob_info in output.logprobs:
                    if logprob_info:
                        # Get the sampled token's logprob
                        for token_id, lp in logprob_info.items():
                            token_logprobs.append(lp.logprob)
                            break
                logprobs_list.append(token_logprobs)
        
        return texts, logprobs_list
    
    def __deepcopy__(self, memo):
        """
        vLLM engines contain CUDA objects that can't be deepcopied.
        Since we use singleton pattern, just return self.
        """
        return self
    
    def __getstate__(self):
        """
        Custom pickle support - only save the model name and config.
        The actual vLLM LLM object can't be pickled.
        """
        return {
            'model': self.model,
            'llm_config': self.llm_config,
        }
    
    def __setstate__(self, state):
        """
        Custom unpickle support - reinitialize from saved state.
        """
        self.model = state['model']
        self.llm_config = state['llm_config']
        self._ensure_client()

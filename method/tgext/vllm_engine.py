"""
vLLM Engine with logprob support for TextGrad.

This module provides a ChatVLLMLogProb class that extends textgrad's vLLM engine
to support logprobs, which are needed for the optimizer.

It also provides a singleton engine manager to share the same vLLM instance
across different parts of the codebase.
"""

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

import platformdirs

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "If you'd like to use vLLM models, please install the vllm package by running "
        "`pip install vllm` or `pip install textgrad[vllm]`."
    )

from textgrad.engine.base import CachedEngine, EngineLM

logger = logging.getLogger("VLLM")

# Default model to use
DEFAULT_MODEL = "Qwen/Qwen3-4B"


@dataclass
class LogProbs:
    """Container for logprob information, compatible with Together API format."""
    tokens: List[str]
    token_logprobs: List[float]
    token_ids: List[int]


class ChatVLLMLogProb(EngineLM, CachedEngine):
    """
    vLLM engine with logprob support, designed for use with textgrad.
    
    This class supports both standard generation and logprob-based generation,
    making it suitable for use with TextualGradientDescentLogProb optimizer.
    """
    
    DEFAULT_SYSTEM_PROMPT = ""
    
    def __init__(
        self,
        model_string: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **llm_config,
    ):
        """
        Initialize the vLLM engine.
        
        Args:
            model_string: The model to load (default: Qwen/Qwen3-4B)
            system_prompt: Default system prompt
            **llm_config: Additional arguments passed to vLLM's LLM class
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_vllm_{model_string.replace('/', '_')}.db")
        super().__init__(cache_path=cache_path)
        
        self.model_string = model_string
        self.system_prompt = system_prompt
        
        logger.info(f"Initializing vLLM with model: {model_string}")
        self.client = LLM(self.model_string, **llm_config)
        self.tokenizer = self.client.get_tokenizer()
        logger.info(f"vLLM initialized successfully")
    
    def _build_chat_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Build the chat prompt using the tokenizer's chat template."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        conversation = []
        if sys_prompt_arg:
            conversation.append({"role": "system", "content": sys_prompt_arg})
        conversation.append({"role": "user", "content": prompt})
        
        chat_str = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return chat_str
    
    def _build_chat_prompt_with_response(
        self, 
        prompt: str, 
        response_text: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Build a chat prompt with an assistant response included (for logprob scoring)."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        conversation = []
        if sys_prompt_arg:
            conversation.append({"role": "system", "content": sys_prompt_arg})
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": response_text})
        
        chat_str = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        return chat_str
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        response_text: Optional[str] = None,
        echo: bool = False,
        logprobs: bool = False,
    ):
        """
        Generate text from the model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (uses default if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            response_text: If provided, compute logprobs for this response instead of generating
            echo: Whether to include input tokens in the output (for logprob computation)
            logprobs: Whether to return logprobs along with the generated text
            
        Returns:
            If logprobs=False: Generated text string
            If logprobs=True: Tuple of (text, LogProbs)
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        if response_text is not None and logprobs:
            # Score an existing response
            return self._score_response(prompt, response_text, sys_prompt_arg)
        
        # Check cache for generation
        cache_key = f"{sys_prompt_arg}|{prompt}|{temperature}|{max_tokens}|{top_p}"
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None and not logprobs:
            return cache_or_none
        
        # Build the chat prompt
        chat_str = self._build_chat_prompt(prompt, sys_prompt_arg)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=1,
            logprobs=1 if logprobs else None,  # Request top-1 logprobs
        )
        
        # Generate
        response = self.client.generate([chat_str], sampling_params)
        output = response[0].outputs[0]
        generated_text = output.text
        
        logger.info(f"Call to vLLM for generation. Response: {generated_text[:100]}...")
        
        # Save to cache
        self._save_cache(cache_key, generated_text)
        
        if logprobs:
            # Extract logprobs from the output
            tokens = []
            token_logprobs = []
            token_ids = []
            
            if output.logprobs:
                for logprob_info in output.logprobs:
                    # logprob_info is a dict mapping token_id to Logprob object
                    if logprob_info:
                        # Get the sampled token's logprob
                        for token_id, lp in logprob_info.items():
                            tokens.append(lp.decoded_token if hasattr(lp, 'decoded_token') else self.tokenizer.decode([token_id]))
                            token_logprobs.append(lp.logprob)
                            token_ids.append(token_id)
                            break  # Only take the first (sampled) token
            
            returned_logprobs = LogProbs(
                tokens=tokens,
                token_logprobs=token_logprobs,
                token_ids=token_ids,
            )
            return generated_text, returned_logprobs
        
        return generated_text
    
    def _score_response(
        self,
        prompt: str,
        response_text: str,
        system_prompt: Optional[str] = None,
    ) -> tuple:
        """
        Compute logprobs for a given response.
        
        This is used by the optimizer to score proposed updates.
        """
        # Build the full prompt with response
        chat_str = self._build_chat_prompt_with_response(prompt, response_text, system_prompt)
        
        # Tokenize to get the response start position
        prompt_only = self._build_chat_prompt(prompt, system_prompt)
        prompt_tokens = self.tokenizer.encode(prompt_only)
        full_tokens = self.tokenizer.encode(chat_str)
        
        # The response tokens start after the prompt tokens
        response_start = len(prompt_tokens)
        
        # Use prompt_logprobs to get logprobs for existing tokens
        sampling_params = SamplingParams(
            max_tokens=1,  # We don't need to generate anything
            prompt_logprobs=1,  # Get logprobs for prompt tokens
        )
        
        response = self.client.generate([chat_str], sampling_params)
        
        # Extract logprobs for the response portion
        tokens = []
        token_logprobs = []
        token_ids = []
        
        if response[0].prompt_logprobs:
            all_logprobs = response[0].prompt_logprobs
            # Skip the prompt portion, only get response logprobs
            for i, logprob_info in enumerate(all_logprobs):
                if i >= response_start and logprob_info:
                    # Get the token at this position
                    token_id = full_tokens[i] if i < len(full_tokens) else None
                    if token_id is not None and token_id in logprob_info:
                        lp = logprob_info[token_id]
                        tokens.append(
                            lp.decoded_token if hasattr(lp, 'decoded_token') 
                            else self.tokenizer.decode([token_id])
                        )
                        token_logprobs.append(lp.logprob)
                        token_ids.append(token_id)
        
        returned_logprobs = LogProbs(
            tokens=tokens,
            token_logprobs=token_logprobs,
            token_ids=token_ids,
        )
        
        logger.info(f"Call to vLLM for logprob scoring. Tokens: {len(tokens)}")
        
        return response_text, returned_logprobs
    
    def logprobs(
        self, 
        prompt: str, 
        response_text: str, 
        system_prompt: Optional[str] = None
    ) -> LogProbs:
        """
        Get logprobs for a given response.
        
        This method is used by TextualGradientDescentLogProb optimizer.
        """
        _, logprobs = self._score_response(prompt, response_text, system_prompt)
        return logprobs
    
    def __call__(self, prompt: str, **kwargs):
        return self.generate(prompt, **kwargs)
    
    def __deepcopy__(self, memo):
        """
        vLLM engines contain CUDA objects that can't be deepcopied.
        Since we use singleton pattern, just return self.
        """
        return self
    
    def __getstate__(self):
        """
        Custom pickle support - only save the model string.
        The actual vLLM LLM object can't be pickled.
        """
        return {
            'model_string': self.model_string,
            'system_prompt': self.system_prompt,
        }
    
    def __setstate__(self, state):
        """
        Custom unpickle support - restore from singleton.
        """
        # Get or create the singleton instance for this model
        engine = VLLMEngineSingleton.get_engine(state['model_string'])
        # Copy attributes from the singleton
        self.__dict__.update(engine.__dict__)


class VLLMEngineSingleton:
    """
    Singleton manager for vLLM engines.
    
    Ensures that only one vLLM instance is created per model, avoiding
    the overhead of loading multiple copies of the same model.
    """
    
    _instances: dict = {}
    _lock = Lock()
    
    @classmethod
    def get_engine(
        cls,
        model_string: str = DEFAULT_MODEL,
        **llm_config,
    ) -> ChatVLLMLogProb:
        """
        Get or create a vLLM engine for the specified model.
        
        Args:
            model_string: The model to load
            **llm_config: Additional arguments passed to vLLM's LLM class
            
        Returns:
            A ChatVLLMLogProb instance
        """
        with cls._lock:
            if model_string not in cls._instances:
                logger.info(f"Creating new vLLM engine for model: {model_string}")
                cls._instances[model_string] = ChatVLLMLogProb(
                    model_string=model_string,
                    **llm_config,
                )
            else:
                logger.info(f"Reusing existing vLLM engine for model: {model_string}")
            return cls._instances[model_string]
    
    @classmethod
    def clear(cls):
        """Clear all cached engines."""
        with cls._lock:
            cls._instances.clear()


# Convenience function for getting the default engine
def get_vllm_engine(model_string: str = DEFAULT_MODEL, **llm_config) -> ChatVLLMLogProb:
    """
    Get a vLLM engine instance (singleton per model).
    
    Args:
        model_string: The model to load (default: Qwen/Qwen3-4B)
        **llm_config: Additional arguments passed to vLLM
        
    Returns:
        A ChatVLLMLogProb instance
    """
    return VLLMEngineSingleton.get_engine(model_string, **llm_config)

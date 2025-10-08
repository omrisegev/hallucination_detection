"""
Answer Collection with Caching and Async Support

This module collects LLM answers for a list of prompts, with:
- File-based caching keyed by SHA hash of prompts
- Optional async batching for efficiency
- Retry logic for robustness
- LLM-agnostic via dependency injection

The cache prevents redundant API calls during experimentation.
"""

import hashlib
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Protocol
import aiohttp


class LLMClient(Protocol):
    """Protocol for synchronous LLM client interface."""
    
    def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate completion for a prompt."""
        ...


class AsyncLLMClient(Protocol):
    """Protocol for asynchronous LLM client interface."""
    
    async def complete_async(self, prompt: str, temperature: float = 0.7) -> str:
        """Asynchronously generate completion for a prompt."""
        ...


class AnswerCollector:
    """
    Collects LLM answers with caching and optional async batching.
    
    This class manages the collection of answers from an LLM, with smart
    caching to avoid redundant API calls. Supports both sync and async modes.
    
    Attributes:
        llm_client: Injected LLM client (sync or async)
        cache_path: Directory path for cache storage
        enable_async: Whether to use async batching
        batch_size: Number of concurrent requests in async mode
        max_retries: Maximum retry attempts on failure
    
    Cache Strategy:
        - Cache key: SHA256 hash of sorted prompts list
        - Cache format: JSON with {prompts: [], answers: [], metadata: {}}
        - Cache location: {cache_path}/{hash[:8]}.json
    
    Example:
        >>> collector = AnswerCollector(llm_client=my_llm, cache_path=".cache")
        >>> prompts = ["What is 2+2?", "What is the capital of France?"]
        >>> answers = collector.collect(prompts)
        >>> len(answers) == len(prompts)
        True
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        cache_path: str = ".cache/answers",
        enable_async: bool = False,
        batch_size: int = 5,
        max_retries: int = 3,
        temperature: float = 0.7,
    ):
        """
        Initialize AnswerCollector.
        
        Args:
            llm_client: LLM client for generating answers
            cache_path: Directory for cache files
            enable_async: Enable async batching
            batch_size: Concurrent requests in async mode
            max_retries: Max retry attempts
            temperature: Default temperature for generation
        """
        self.llm_client = llm_client
        self.cache_path = Path(cache_path)
        self.enable_async = enable_async
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.temperature = temperature
        
        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def collect(self, prompts: List[str]) -> List[str]:
        """
        Collect answers for a list of prompts.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of answer strings, aligned with prompts
            
        The method first checks cache, then collects missing answers.
        """
        # Check cache first
        cache_key = self._compute_cache_key(prompts)
        cached_answers = self._load_from_cache(cache_key, prompts)
        
        if cached_answers is not None:
            return cached_answers
        
        # Collect answers
        if self.enable_async and hasattr(self.llm_client, 'complete_async'):
            answers = asyncio.run(self._collect_async(prompts))
        else:
            answers = self._collect_sync(prompts)
        
        # Cache results
        self._save_to_cache(cache_key, prompts, answers)
        
        return answers
    
    def _collect_sync(self, prompts: List[str]) -> List[str]:
        """
        Synchronously collect answers.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of answers
        """
        answers = []
        
        for prompt in prompts:
            answer = None
            for attempt in range(self.max_retries):
                try:
                    answer = self.llm_client.complete(prompt, temperature=self.temperature)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        # Final attempt failed, use error placeholder
                        answer = f"[ERROR: Failed to generate answer after {self.max_retries} attempts]"
                    else:
                        # Retry with backoff
                        import time
                        time.sleep(2 ** attempt)
            
            answers.append(answer)
        
        return answers
    
    async def _collect_async(self, prompts: List[str]) -> List[str]:
        """
        Asynchronously collect answers in batches.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of answers
        """
        answers = [None] * len(prompts)
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_indices = range(i, min(i + self.batch_size, len(prompts)))
            
            # Create tasks for this batch
            tasks = [
                self._fetch_with_retry(prompt) 
                for prompt in batch
            ]
            
            # Wait for batch to complete
            batch_answers = await asyncio.gather(*tasks)
            
            # Store answers at correct indices
            for idx, answer in zip(batch_indices, batch_answers):
                answers[idx] = answer
        
        return answers
    
    async def _fetch_with_retry(self, prompt: str) -> str:
        """
        Fetch answer with retry logic (async).
        
        Args:
            prompt: Single prompt
            
        Returns:
            Answer string
        """
        for attempt in range(self.max_retries):
            try:
                answer = await self.llm_client.complete_async(
                    prompt, 
                    temperature=self.temperature
                )
                return answer
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"[ERROR: Failed to generate answer after {self.max_retries} attempts]"
                else:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
    
    def _compute_cache_key(self, prompts: List[str]) -> str:
        """
        Compute cache key from prompts.
        
        Args:
            prompts: List of prompts
            
        Returns:
            SHA256 hash (first 16 chars)
            
        The cache key is deterministic for the same set of prompts,
        regardless of order (we sort them first).
        """
        # Sort prompts for deterministic hashing
        sorted_prompts = sorted(prompts)
        
        # Compute SHA256
        prompt_str = json.dumps(sorted_prompts, sort_keys=True)
        hash_obj = hashlib.sha256(prompt_str.encode('utf-8'))
        
        return hash_obj.hexdigest()[:16]
    
    def _load_from_cache(
        self, 
        cache_key: str, 
        prompts: List[str]
    ) -> Optional[List[str]]:
        """
        Load answers from cache if available.
        
        Args:
            cache_key: Cache key hash
            prompts: Original prompts (for validation)
            
        Returns:
            List of cached answers, or None if not found/invalid
        """
        cache_file = self.cache_path / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate cache (check prompts match)
            if sorted(data.get('prompts', [])) == sorted(prompts):
                return data['answers']
            else:
                # Cache mismatch (hash collision or corruption)
                return None
                
        except Exception as e:
            # Cache read error, ignore and regenerate
            return None
    
    def _save_to_cache(
        self, 
        cache_key: str, 
        prompts: List[str], 
        answers: List[str]
    ):
        """
        Save answers to cache.
        
        Args:
            cache_key: Cache key hash
            prompts: Original prompts
            answers: Generated answers
        """
        cache_file = self.cache_path / f"{cache_key}.json"
        
        data = {
            'prompts': prompts,
            'answers': answers,
            'metadata': {
                'temperature': self.temperature,
                'batch_size': self.batch_size,
            }
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Cache write error, log but don't fail
            print(f"Warning: Failed to write cache: {e}")
    
    def clear_cache(self):
        """Clear all cached answers."""
        for cache_file in self.cache_path.glob("*.json"):
            cache_file.unlink()

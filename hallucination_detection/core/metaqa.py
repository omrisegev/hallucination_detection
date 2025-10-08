"""
MetaQA-style Question Perturbation Generator

This module generates diverse perturbations of a given question to collect
multiple answers from an LLM. The diversity in phrasing and prompting helps
reveal consistency or hallucinations in the model's responses.

Key Features:
- Basic paraphrasing via template-based rephrasing
- Role-based prompting (expert, objective, factual perspectives)
- Synonym substitution for variation
- Temperature range sampling for stochastic diversity
- LLM-agnostic via dependency injection

References:
- MetaQA approach for question perturbation
- Temperature sampling for diverse generations
"""

from typing import List, Optional, Protocol, Any
import random


class LLMClient(Protocol):
    """Protocol for LLM client interface."""
    
    def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate completion for a prompt at given temperature."""
        ...


class MetaQAGenerator:
    """
    Generates MetaQA-style perturbations of questions.
    
    This class creates diverse versions of a question through:
    1. Template-based paraphrasing
    2. Role-based prompt variations
    3. Synonym substitution
    4. Temperature-based stochastic sampling
    
    Attributes:
        llm_client: Injected LLM client for optional LLM-based paraphrasing
        num_perturbations: Number of perturbed prompts to generate
        temperature_range: Tuple of (min_temp, max_temp) for sampling
        paraphrase_templates: List of template strings for rephrasing
        role_prompts: List of role-based prompt prefixes
    
    Example:
        >>> generator = MetaQAGenerator(llm_client=my_llm, num_perturbations=10)
        >>> perturbations = generator.generate("What is the capital of France?")
        >>> len(perturbations)
        10
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        num_perturbations: int = 12,
        temperature_range: tuple[float, float] = (0.3, 0.9),
        paraphrase_templates: Optional[List[str]] = None,
        role_prompts: Optional[List[str]] = None,
    ):
        """
        Initialize MetaQAGenerator.
        
        Args:
            llm_client: Optional LLM client for advanced paraphrasing
            num_perturbations: Number of perturbations to generate
            temperature_range: Range of temperatures for sampling
            paraphrase_templates: Custom paraphrase templates
            role_prompts: Custom role-based prompts
        """
        self.llm_client = llm_client
        self.num_perturbations = num_perturbations
        self.temperature_range = temperature_range
        
        # Default paraphrase templates
        self.paraphrase_templates = paraphrase_templates or [
            "{question}",  # Original
            "Rephrase this question: {question}",
            "Ask differently: {question}",
            "Alternative phrasing: {question}",
            "Please answer: {question}",
            "I need to know: {question}",
        ]
        
        # Default role prompts
        self.role_prompts = role_prompts or [
            "{question}",  # Direct
            "As an expert, answer: {question}",
            "From a factual perspective: {question}",
            "Objectively consider: {question}",
            "Based on verified information: {question}",
            "Using your knowledge: {question}",
        ]
        
        # Simple synonym dictionary for basic substitution
        self.synonyms = {
            "what": ["what", "which"],
            "who": ["who", "which person"],
            "when": ["when", "at what time"],
            "where": ["where", "in what place"],
            "how": ["how", "in what way"],
            "why": ["why", "for what reason"],
        }
    
    def generate(self, question: str) -> List[str]:
        """
        Generate perturbed versions of a question.
        
        Args:
            question: Original question to perturb
            
        Returns:
            List of perturbed prompt strings
            
        Strategy:
            1. Always include the original question
            2. Apply template variations
            3. Apply role-based variations
            4. Optionally use synonym substitution
            5. Ensure diversity via temperature sampling metadata
        """
        perturbations = []
        
        # 1. Original question
        perturbations.append(question)
        
        # 2. Template-based paraphrasing
        for template in self.paraphrase_templates[1:]:  # Skip first (original)
            perturbations.append(template.format(question=question))
            if len(perturbations) >= self.num_perturbations:
                break
        
        # 3. Role-based prompts
        if len(perturbations) < self.num_perturbations:
            for role_prompt in self.role_prompts[1:]:  # Skip first (direct)
                perturbations.append(role_prompt.format(question=question))
                if len(perturbations) >= self.num_perturbations:
                    break
        
        # 4. Synonym substitution for additional variety
        if len(perturbations) < self.num_perturbations:
            synonym_variants = self._apply_synonym_substitution(question)
            for variant in synonym_variants:
                perturbations.append(variant)
                if len(perturbations) >= self.num_perturbations:
                    break
        
        # 5. LLM-based paraphrasing if available and needed
        if (len(perturbations) < self.num_perturbations and 
            self.llm_client is not None):
            llm_variants = self._llm_paraphrase(question, 
                                                self.num_perturbations - len(perturbations))
            perturbations.extend(llm_variants)
        
        # Trim to exact number requested
        return perturbations[:self.num_perturbations]
    
    def _apply_synonym_substitution(self, question: str) -> List[str]:
        """
        Apply basic synonym substitution to create variants.
        
        Args:
            question: Original question
            
        Returns:
            List of questions with synonym substitutions
        """
        variants = []
        words = question.lower().split()
        
        for i, word in enumerate(words):
            # Check if word (without punctuation) is in synonyms
            clean_word = word.strip("?.,!;:")
            if clean_word in self.synonyms:
                for syn in self.synonyms[clean_word]:
                    if syn != clean_word:
                        new_words = words.copy()
                        # Preserve case
                        if word[0].isupper():
                            new_words[i] = syn.capitalize() + word[len(clean_word):]
                        else:
                            new_words[i] = syn + word[len(clean_word):]
                        variants.append(" ".join(new_words))
        
        return variants
    
    def _llm_paraphrase(self, question: str, num_variants: int) -> List[str]:
        """
        Use LLM to generate paraphrases.
        
        Args:
            question: Original question
            num_variants: Number of paraphrases to generate
            
        Returns:
            List of LLM-generated paraphrases
        """
        variants = []
        
        for i in range(num_variants):
            # Sample temperature from range
            temp = random.uniform(*self.temperature_range)
            
            prompt = (
                f"Paraphrase the following question while preserving its meaning. "
                f"Provide only the paraphrased question:\n\n{question}"
            )
            
            try:
                paraphrase = self.llm_client.complete(prompt, temperature=temp)
                # Clean up response (remove quotes, extra whitespace)
                paraphrase = paraphrase.strip().strip('"\'')
                if paraphrase and paraphrase != question:
                    variants.append(paraphrase)
            except Exception as e:
                # Silently skip on error, fallback to template-based
                continue
        
        return variants
    
    def get_temperature_schedule(self) -> List[float]:
        """
        Generate temperature schedule for LLM sampling.
        
        Returns:
            List of temperatures to use for each perturbation
        """
        return [
            random.uniform(*self.temperature_range) 
            for _ in range(self.num_perturbations)
        ]

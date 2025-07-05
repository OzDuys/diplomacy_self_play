"""
vLLM integration for efficient batch inference.

Provides async batch inference capabilities for the Diplomacy GRPO pipeline
using vLLM's OpenAI-compatible API.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import aiohttp
import openai
from asyncio_throttle import Throttler


class VLLMBatchClient:
    """
    Async vLLM client optimized for batch inference.
    
    Handles concurrent requests to vLLM server with rate limiting,
    connection pooling, and error handling for robust batch generation.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        max_concurrent: int = 35,
        requests_per_second: float = 10.0,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Rate limiting
        self.throttler = Throttler(rate_limit=requests_per_second)
        
        # OpenAI client with custom settings
        self.client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",  # vLLM doesn't require real API key
            timeout=timeout,
        )
        
        self.logger = logging.getLogger(__name__)
        
    async def generate_single(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion for a single prompt.
        
        Args:
            prompt: Input prompt (string or chat messages)
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with completion and metadata
        """
        async with self.throttler:
            try:
                # Prepare messages format
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt
                
                # Make request
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
                
                # Extract result
                completion = response.choices[0].message.content
                
                return {
                    'completion': completion,
                    'prompt': prompt,
                    'model': model,
                    'usage': response.usage.model_dump() if response.usage else {},
                    'finish_reason': response.choices[0].finish_reason,
                    'success': True,
                    'error': None,
                }
                
            except Exception as e:
                self.logger.error(f"Generation failed: {e}")
                return {
                    'completion': "",
                    'prompt': prompt,
                    'model': model,
                    'usage': {},
                    'finish_reason': 'error',
                    'success': False,
                    'error': str(e),
                }
    
    async def generate_batch(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a batch of prompts concurrently.
        
        Args:
            prompts: List of prompts to process
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature  
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            List of completion dictionaries
        """
        self.logger.info(f"Generating batch of {len(prompts)} prompts")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def generate_with_semaphore(prompt):
            async with semaphore:
                return await self.generate_single(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
        
        # Execute all requests concurrently
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Prompt {i} failed: {result}")
                processed_results.append({
                    'completion': "",
                    'prompt': prompts[i],
                    'model': model,
                    'usage': {},
                    'finish_reason': 'error',
                    'success': False,
                    'error': str(result),
                })
            else:
                processed_results.append(result)
        
        # Log statistics
        successful = sum(1 for r in processed_results if r['success'])
        self.logger.info(f"Batch completed: {successful}/{len(prompts)} successful")
        
        return processed_results
        
    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.host}:{self.port}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models from vLLM server.
        
        Returns:
            Dictionary with model information
        """
        try:
            models = await self.client.models.list()
            return {
                'available_models': [model.id for model in models.data],
                'server_url': self.base_url,
                'healthy': True,
            }
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {
                'available_models': [],
                'server_url': self.base_url,
                'healthy': False,
                'error': str(e),
            }
    
    async def close(self):
        """Close the client and clean up resources."""
        await self.client.close()


class DiplomacyBatchInferenceManager:
    """
    High-level manager for Diplomacy-specific batch inference.
    
    Handles country-specific prompt formatting, batch coordination,
    and result processing for the GRPO training pipeline.
    """
    
    def __init__(
        self,
        vllm_client: VLLMBatchClient,
        countries: Optional[List[str]] = None,
    ):
        self.client = vllm_client
        self.countries = countries or [
            'Austria', 'England', 'France', 'Germany',
            'Italy', 'Russia', 'Turkey'
        ]
        self.logger = logging.getLogger(__name__)
    
    def format_diplomacy_prompt(
        self,
        country: str,
        scenario: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Format a diplomacy prompt for the specified country.
        
        Args:
            country: The country assignment
            scenario: The game scenario description
            system_prompt: Optional system prompt override
            
        Returns:
            Formatted messages for chat completion
        """
        if system_prompt is None:
            system_prompt = f"You are playing as {country} in a game of Diplomacy. You are a skilled diplomat and strategist."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": scenario}
        ]
        
        return messages
    
    async def generate_country_actions(
        self,
        scenarios: List[Dict[str, Any]],
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate actions for multiple country scenarios.
        
        Args:
            scenarios: List of scenario dicts with 'country' and 'prompt' keys
            model: Model name for generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of results with country, prompt, completion, and metadata
        """
        # Format prompts
        formatted_prompts = []
        for scenario in scenarios:
            country = scenario['country']
            prompt = scenario['prompt']
            system_prompt = scenario.get('system_prompt')
            
            formatted_prompt = self.format_diplomacy_prompt(
                country=country,
                scenario=prompt,
                system_prompt=system_prompt
            )
            formatted_prompts.append(formatted_prompt)
        
        # Generate completions
        results = await self.client.generate_batch(
            prompts=formatted_prompts,
            model=model,
            **generation_kwargs
        )
        
        # Combine with scenario metadata
        enhanced_results = []
        for scenario, result in zip(scenarios, results):
            enhanced_result = {
                **result,
                'country': scenario['country'],
                'scenario_metadata': scenario.get('metadata', {}),
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    async def generate_balanced_batch(
        self,
        num_generations_per_country: int = 5,
        scenarios_per_country: Optional[Dict[str, List[str]]] = None,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        **generation_kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a balanced batch with equal representation per country.
        
        Args:
            num_generations_per_country: Number of generations per country
            scenarios_per_country: Dict mapping countries to their scenarios
            model: Model name for generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping countries to their results
        """
        # Use default scenarios if none provided
        if scenarios_per_country is None:
            scenarios_per_country = {
                country: [f"You are {country} in a critical game situation. What is your strategy?"]
                for country in self.countries
            }
        
        # Build scenario list
        all_scenarios = []
        for country in self.countries:
            country_scenarios = scenarios_per_country.get(country, [])
            if not country_scenarios:
                continue
                
            for i in range(num_generations_per_country):
                scenario_text = country_scenarios[i % len(country_scenarios)]
                all_scenarios.append({
                    'country': country,
                    'prompt': scenario_text,
                    'generation_id': i,
                })
        
        # Generate all completions
        results = await self.generate_country_actions(
            scenarios=all_scenarios,
            model=model,
            **generation_kwargs
        )
        
        # Group by country
        results_by_country = {country: [] for country in self.countries}
        for result in results:
            country = result['country']
            if country in results_by_country:
                results_by_country[country].append(result)
        
        return results_by_country
    
    async def close(self):
        """Close the underlying client."""
        await self.client.close()
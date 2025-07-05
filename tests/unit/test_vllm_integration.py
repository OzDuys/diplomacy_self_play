"""
Unit tests for vLLM integration components.

Tests VLLMBatchClient and DiplomacyBatchInferenceManager without requiring
actual vLLM server connections.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from diplomacy_grpo.integration.vllm_client import (
    VLLMBatchClient,
    DiplomacyBatchInferenceManager,
)


class TestVLLMBatchClient:
    """Test VLLMBatchClient without actual vLLM server."""
    
    def test_initialization(self):
        """Test VLLMBatchClient initialization."""
        client = VLLMBatchClient(
            host="localhost",
            port=8000,
            max_concurrent=10,
            requests_per_second=5.0,
            timeout=30.0,
            max_retries=2
        )
        
        assert client.host == "localhost"
        assert client.port == 8000
        assert client.base_url == "http://localhost:8000/v1"
        assert client.max_concurrent == 10
        assert client.timeout == 30.0
        assert client.max_retries == 2
    
    def test_default_initialization(self):
        """Test VLLMBatchClient with default parameters."""
        client = VLLMBatchClient()
        
        assert client.host == "localhost"
        assert client.port == 8000
        assert client.base_url == "http://localhost:8000/v1"
        assert client.max_concurrent == 35
        assert client.timeout == 60.0
        assert client.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_generate_single_success(self):
        """Test successful single generation."""
        client = VLLMBatchClient()
        
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test completion"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"total_tokens": 50}
        
        with patch.object(client.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
            result = await client.generate_single("Test prompt")
            
        assert result['success'] is True
        assert result['completion'] == "Test completion"
        assert result['prompt'] == "Test prompt"
        assert result['finish_reason'] == "stop"
        assert result['usage'] == {"total_tokens": 50}
        assert result['error'] is None
    
    @pytest.mark.asyncio
    async def test_generate_single_error(self):
        """Test single generation with error."""
        client = VLLMBatchClient()
        
        with patch.object(client.client.chat.completions, 'create', new=AsyncMock(side_effect=Exception("Connection error"))):
            result = await client.generate_single("Test prompt")
            
        assert result['success'] is False
        assert result['completion'] == ""
        assert result['prompt'] == "Test prompt"
        assert result['finish_reason'] == "error"
        assert result['error'] == "Connection error"
    
    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Test batch generation."""
        client = VLLMBatchClient(max_concurrent=2)
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test completion"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        
        with patch.object(client.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = await client.generate_batch(prompts)
            
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['success'] is True
            assert result['completion'] == "Test completion"
            assert result['prompt'] == f"Prompt {i+1}"
    
    @pytest.mark.asyncio
    async def test_generate_batch_with_errors(self):
        """Test batch generation with some errors."""
        client = VLLMBatchClient()
        
        # Mock mixed responses (success, error, success)
        def mock_create(*args, **kwargs):
            prompt = kwargs.get('messages', [{}])[0].get('content', '')
            if "error" in prompt:
                raise Exception("Mock error")
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = None
            return mock_response
        
        with patch.object(client.client.chat.completions, 'create', new=AsyncMock(side_effect=mock_create)):
            prompts = ["Good prompt", "This will error", "Another good prompt"]
            results = await client.generate_batch(prompts)
            
        assert len(results) == 3
        assert results[0]['success'] is True
        assert results[1]['success'] is False
        assert results[2]['success'] is True
        assert "Mock error" in results[1]['error']
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = VLLMBatchClient()
        
        mock_response = Mock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Create proper async context manager mock
            mock_session_instance = Mock()
            mock_get = Mock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.get.return_value = mock_get
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            is_healthy = await client.health_check()
            
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        client = VLLMBatchClient()
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Create proper async context manager mock that raises exception
            mock_session_instance = Mock()
            mock_session_instance.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            is_healthy = await client.health_check()
            
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_get_model_info_success(self):
        """Test successful model info retrieval."""
        client = VLLMBatchClient()
        
        mock_models = Mock()
        mock_models.data = [Mock(id="model1"), Mock(id="model2")]
        
        with patch.object(client.client.models, 'list', new=AsyncMock(return_value=mock_models)):
            info = await client.get_model_info()
            
        assert info['healthy'] is True
        assert info['available_models'] == ["model1", "model2"]
        assert info['server_url'] == client.base_url
    
    @pytest.mark.asyncio
    async def test_get_model_info_error(self):
        """Test model info retrieval with error."""
        client = VLLMBatchClient()
        
        with patch.object(client.client.models, 'list', new=AsyncMock(side_effect=Exception("API error"))):
            info = await client.get_model_info()
            
        assert info['healthy'] is False
        assert info['available_models'] == []
        assert "API error" in info['error']


class TestDiplomacyBatchInferenceManager:
    """Test DiplomacyBatchInferenceManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.manager = DiplomacyBatchInferenceManager(
            vllm_client=self.mock_client,
            countries=['Austria', 'England', 'France']
        )
    
    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.client == self.mock_client
        assert self.manager.countries == ['Austria', 'England', 'France']
    
    def test_initialization_with_default_countries(self):
        """Test manager initialization with default countries."""
        manager = DiplomacyBatchInferenceManager(self.mock_client)
        expected_countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        assert manager.countries == expected_countries
    
    def test_format_diplomacy_prompt_basic(self):
        """Test basic prompt formatting."""
        messages = self.manager.format_diplomacy_prompt(
            country="Austria",
            scenario="Test scenario"
        )
        
        assert len(messages) == 2
        assert messages[0]['role'] == "system"
        assert "Austria" in messages[0]['content']
        assert messages[1]['role'] == "user"
        assert messages[1]['content'] == "Test scenario"
    
    def test_format_diplomacy_prompt_custom_system(self):
        """Test prompt formatting with custom system prompt."""
        custom_system = "Custom system prompt for testing"
        messages = self.manager.format_diplomacy_prompt(
            country="England",
            scenario="Test scenario",
            system_prompt=custom_system
        )
        
        assert messages[0]['content'] == custom_system
        assert messages[1]['content'] == "Test scenario"
    
    @pytest.mark.asyncio
    async def test_generate_country_actions(self):
        """Test country action generation."""
        scenarios = [
            {'country': 'Austria', 'prompt': 'Austria scenario'},
            {'country': 'England', 'prompt': 'England scenario'},
        ]
        
        mock_results = [
            {'completion': 'Austria response', 'success': True},
            {'completion': 'England response', 'success': True},
        ]
        self.mock_client.generate_batch = AsyncMock(return_value=mock_results)
        
        results = await self.manager.generate_country_actions(scenarios)
        
        assert len(results) == 2
        assert results[0]['country'] == 'Austria'
        assert results[1]['country'] == 'England'
        assert results[0]['completion'] == 'Austria response'
        assert results[1]['completion'] == 'England response'
        
        # Verify client was called with formatted prompts
        self.mock_client.generate_batch.assert_called_once()
        call_args = self.mock_client.generate_batch.call_args[1]
        prompts = call_args['prompts']
        assert len(prompts) == 2
        assert isinstance(prompts[0], list)  # Should be formatted messages
    
    @pytest.mark.asyncio
    async def test_generate_country_actions_with_metadata(self):
        """Test country action generation with metadata."""
        scenarios = [
            {
                'country': 'France',
                'prompt': 'France scenario',
                'metadata': {'difficulty': 'hard'},
                'system_prompt': 'Custom system for France'
            }
        ]
        
        mock_results = [{'completion': 'France response', 'success': True}]
        self.mock_client.generate_batch = AsyncMock(return_value=mock_results)
        
        results = await self.manager.generate_country_actions(scenarios)
        
        assert results[0]['scenario_metadata'] == {'difficulty': 'hard'}
    
    @pytest.mark.asyncio
    async def test_generate_balanced_batch_default_scenarios(self):
        """Test balanced batch generation with default scenarios."""
        mock_results = [
            {'completion': f'Response {i}', 'success': True}
            for i in range(6)  # 3 countries × 2 generations
        ]
        self.mock_client.generate_batch = AsyncMock(return_value=mock_results)
        
        results = await self.manager.generate_balanced_batch(
            num_generations_per_country=2
        )
        
        # Should have results for each country
        assert 'Austria' in results
        assert 'England' in results
        assert 'France' in results
        
        # Each country should have 2 generations
        assert len(results['Austria']) == 2
        assert len(results['England']) == 2
        assert len(results['France']) == 2
    
    @pytest.mark.asyncio
    async def test_generate_balanced_batch_custom_scenarios(self):
        """Test balanced batch generation with custom scenarios."""
        custom_scenarios = {
            'Austria': ['Austria scenario 1', 'Austria scenario 2'],
            'England': ['England scenario 1'],
            # France omitted to test handling of missing countries
        }
        
        mock_results = [
            {'completion': f'Response {i}', 'success': True}
            for i in range(3)  # Only Austria and England
        ]
        self.mock_client.generate_batch = AsyncMock(return_value=mock_results)
        
        results = await self.manager.generate_balanced_batch(
            num_generations_per_country=1,
            scenarios_per_country=custom_scenarios
        )
        
        # Should only have Austria and England (France has no scenarios)
        assert 'Austria' in results
        assert 'England' in results
        assert len(results['France']) == 0  # Empty list for France
    
    @pytest.mark.asyncio 
    async def test_close(self):
        """Test manager close method."""
        self.mock_client.close = AsyncMock()
        
        await self.manager.close()
        
        self.mock_client.close.assert_called_once()


@pytest.mark.parametrize("country", ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"])
class TestCountrySpecificPrompts:
    """Test country-specific prompt formatting."""
    
    def test_all_countries_in_system_prompt(self, country):
        """Test that each country appears in its system prompt."""
        mock_client = Mock()
        manager = DiplomacyBatchInferenceManager(mock_client)
        
        messages = manager.format_diplomacy_prompt(
            country=country,
            scenario="Test scenario"
        )
        
        system_prompt = messages[0]['content']
        assert country in system_prompt
        assert "Diplomacy" in system_prompt
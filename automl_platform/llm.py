"""
Enhanced LLM Integration Module with RAG, Agents, and Data Cleaning
Inspired by DataRobot and Akkio's GPT-4 capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import json
import hashlib
import os
from datetime import datetime
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import redis
from pathlib import Path
import re

# LLM providers
import openai
from anthropic import Anthropic

# Vector stores
import chromadb
from chromadb.config import Settings
import faiss
import pickle

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str
    tokens_used: int
    cost: float
    cached: bool = False
    metadata: Dict[str, Any] = None


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.total_tokens = 0
        self.total_cost = 0.0
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4  # Rough estimate


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        super().__init__(api_key, model_name)
        openai.api_key = api_key
        
        # Pricing per 1K tokens (as of 2024)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    
    async def generate(self, prompt: str, temperature: float = 0.7, 
                       max_tokens: int = 1000, **kwargs) -> LLMResponse:
        """Generate response from OpenAI."""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            # Calculate cost
            model_pricing = self.pricing.get(self.model_name, self.pricing["gpt-3.5-turbo"])
            cost = (response.usage.prompt_tokens * model_pricing["input"] + 
                   response.usage.completion_tokens * model_pricing["output"]) / 1000
            
            self.total_tokens += tokens
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens,
                cost=cost,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        super().__init__(api_key, model_name)
        self.client = Anthropic(api_key=api_key)
        
        # Pricing per 1K tokens
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
    
    async def generate(self, prompt: str, temperature: float = 0.7,
                       max_tokens: int = 1000, **kwargs) -> LLMResponse:
        """Generate response from Anthropic."""
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            
            # Calculate cost
            model_pricing = self.pricing.get(self.model_name, self.pricing["claude-3-haiku-20240307"])
            cost = (response.usage.input_tokens * model_pricing["input"] + 
                   response.usage.output_tokens * model_pricing["output"]) / 1000
            
            self.total_tokens += tokens
            self.total_cost += cost
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens,
                cost=cost,
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class LLMCache:
    """Redis-based cache for LLM responses."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
    
    def _get_key(self, prompt: str, model: str) -> str:
        """Generate cache key."""
        content = f"{model}:{prompt}"
        return f"llm:cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def get(self, prompt: str, model: str) -> Optional[LLMResponse]:
        """Get cached response."""
        key = self._get_key(prompt, model)
        try:
            data = self.redis_client.get(key)
            if data:
                response_dict = json.loads(data)
                response_dict["cached"] = True
                return LLMResponse(**response_dict)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None
    
    def set(self, prompt: str, response: LLMResponse):
        """Cache response."""
        key = self._get_key(prompt, response.model)
        try:
            self.redis_client.setex(
                key, self.ttl,
                json.dumps(asdict(response), default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")


class RAGSystem:
    """Retrieval-Augmented Generation system."""
    
    def __init__(self, vector_store: str = "chromadb", embedding_model: str = "text-embedding-ada-002"):
        self.vector_store_type = vector_store
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store
        if vector_store == "chromadb":
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            self.collection = self.client.get_or_create_collection("automl_docs")
        elif vector_store == "faiss":
            self.index = None
            self.documents = []
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to vector store."""
        if self.vector_store_type == "chromadb":
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata or [{}] * len(documents),
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
        
        elif self.vector_store_type == "faiss":
            # Generate embeddings
            embeddings = np.array(self.embeddings.embed_documents(documents))
            
            # Initialize or update FAISS index
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
            
            self.index.add(embeddings)
            self.documents.extend(documents)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant documents."""
        query_embedding = self.embeddings.embed_query(query)
        
        if self.vector_store_type == "chromadb":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            if results["documents"]:
                return list(zip(results["documents"][0], results["distances"][0]))
            
        elif self.vector_store_type == "faiss" and self.index:
            query_vec = np.array([query_embedding])
            distances, indices = self.index.search(query_vec, k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(dist)))
            return results
        
        return []


class DataCleaningAgent:
    """Intelligent data cleaning agent inspired by Akkio."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.cleaning_history = []
    
    async def analyze_data_issues(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze data quality issues."""
        sample = df.head(sample_size)
        
        # Create data summary
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "sample": sample.to_dict('records')[:5]
        }
        
        prompt = f"""
        Analyze this dataset for quality issues:
        
        Shape: {summary['shape']}
        Columns: {summary['columns']}
        Data Types: {summary['dtypes']}
        Missing Values: {summary['missing']}
        Sample Data: {json.dumps(summary['sample'], indent=2)}
        
        Identify:
        1. Data quality issues (missing values, outliers, incorrect types)
        2. Suggested transformations
        3. Potential feature engineering opportunities
        4. Columns that might need special handling
        
        Provide a structured JSON response with:
        {{
            "issues": [...],
            "transformations": [...],
            "feature_suggestions": [...],
            "warnings": [...]
        }}
        """
        
        response = await self.llm.generate(prompt, temperature=0.3)
        
        try:
            analysis = json.loads(response.content)
        except:
            analysis = {
                "issues": ["Could not parse LLM response"],
                "transformations": [],
                "feature_suggestions": [],
                "warnings": []
            }
        
        return analysis
    
    async def suggest_cleaning_code(self, df: pd.DataFrame, issue: str) -> str:
        """Generate Python code to fix data issues."""
        prompt = f"""
        Generate Python code to fix this data issue:
        
        Issue: {issue}
        
        Dataset info:
        - Columns: {list(df.columns)}
        - Shape: {df.shape}
        - Data types: {df.dtypes.to_dict()}
        
        Provide clean, executable Python code using pandas.
        The dataframe variable is 'df'.
        Include comments explaining each step.
        """
        
        response = await self.llm.generate(prompt, temperature=0.2)
        
        # Extract code from response
        code = self._extract_code(response.content)
        
        # Log cleaning action
        self.cleaning_history.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "code": code
        })
        
        return code
    
    async def interactive_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interactive data cleaning conversation."""
        df_cleaned = df.copy()
        
        # Initial analysis
        analysis = await self.analyze_data_issues(df_cleaned)
        
        print("\n=== Data Quality Analysis ===")
        print(f"Found {len(analysis['issues'])} issues:")
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"{i}. {issue}")
        
        # Process each issue
        for issue in analysis['issues']:
            print(f"\n--- Addressing: {issue} ---")
            
            # Get cleaning code
            code = await self.suggest_cleaning_code(df_cleaned, issue)
            print(f"Suggested fix:\n{code}")
            
            # Execute if approved (in production, add user confirmation)
            try:
                exec(code, {"df": df_cleaned, "pd": pd, "np": np})
                print("✓ Applied successfully")
            except Exception as e:
                print(f"✗ Error applying fix: {e}")
        
        return df_cleaned
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_pattern = r"```python\n(.*?)```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: look for lines that look like code
        lines = text.split('\n')
        code_lines = [line for line in lines if line.strip() and 
                     (line.startswith('df') or line.startswith('#') or '=' in line)]
        
        return '\n'.join(code_lines)


class AutoMLLLMAssistant:
    """Main LLM assistant for AutoML platform."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize LLM provider
        if config['provider'] == 'openai':
            self.llm = OpenAIProvider(config['api_key'], config['model_name'])
        elif config['provider'] == 'anthropic':
            self.llm = AnthropicProvider(config['api_key'], config['model_name'])
        else:
            raise ValueError(f"Unsupported provider: {config['provider']}")
        
        # Initialize cache
        self.cache = LLMCache() if config.get('cache_responses', True) else None
        
        # Initialize RAG
        self.rag = RAGSystem(
            vector_store=config.get('vector_store', 'chromadb'),
            embedding_model=config.get('embedding_model', 'text-embedding-ada-002')
        ) if config.get('enable_rag', True) else None
        
        # Initialize agents
        self.cleaning_agent = DataCleaningAgent(self.llm)
        
        # Load prompts
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates."""
        prompts_dir = Path(self.config.get('prompts_dir', './prompts'))
        prompts = {}
        
        # Default prompts if files don't exist
        default_prompts = {
            "feature_engineering": """
                You are an expert data scientist. Suggest feature engineering for this dataset:
                
                Dataset Info:
                {dataset_info}
                
                Task Type: {task_type}
                Target Column: {target_column}
                
                Provide specific, actionable feature engineering suggestions with Python code.
            """,
            
            "model_selection": """
                Based on this dataset, recommend the best ML models:
                
                Dataset Characteristics:
                {data_characteristics}
                
                Consider model complexity, training time, and interpretability.
                Rank your recommendations and explain why.
            """,
            
            "explain_model": """
                Explain this model's performance in simple terms:
                
                Model: {model_name}
                Metrics: {metrics}
                Feature Importance: {feature_importance}
                
                Provide insights that a non-technical stakeholder would understand.
            """,
            
            "error_analysis": """
                Analyze these prediction errors:
                
                Error Statistics: {error_stats}
                Sample Errors: {sample_errors}
                
                Identify patterns and suggest improvements.
            """,
            
            "generate_report": """
                Generate an executive summary for this AutoML experiment:
                
                Experiment ID: {experiment_id}
                Best Model: {best_model}
                Performance: {performance}
                Key Features: {key_features}
                
                Format as a professional report with recommendations.
            """
        }
        
        # Load from files if available
        for name, default in default_prompts.items():
            prompt_file = prompts_dir / f"{name}.txt"
            if prompt_file.exists():
                prompts[name] = prompt_file.read_text()
            else:
                prompts[name] = default
        
        return prompts
    
    async def suggest_features(self, df: pd.DataFrame, target_column: str,
                              task_type: str = "classification") -> List[Dict[str, Any]]:
        """Suggest feature engineering with code."""
        # Check cache
        cache_key = f"features_{hashlib.sha256(df.to_csv().encode()).hexdigest()[:8]}_{target_column}"
        
        if self.cache:
            cached = self.cache.get(cache_key, "feature_suggestions")
            if cached:
                return json.loads(cached.content)
        
        # Prepare dataset info
        dataset_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(3).to_dict('records'),
            "correlation_with_target": df.corrwith(df[target_column]).to_dict() if target_column in df else {}
        }
        
        # Generate prompt
        prompt = self.prompts["feature_engineering"].format(
            dataset_info=json.dumps(dataset_info, indent=2),
            task_type=task_type,
            target_column=target_column
        )
        
        # Add RAG context if available
        if self.rag:
            context = self.rag.search(f"feature engineering {task_type}", k=3)
            if context:
                prompt += "\n\nRelevant examples:\n" + "\n".join([doc for doc, _ in context])
        
        # Generate suggestions
        response = await self.llm.generate(prompt, temperature=0.7, max_tokens=1500)
        
        # Parse response
        suggestions = self._parse_feature_suggestions(response.content)
        
        # Cache result
        if self.cache:
            cache_response = LLMResponse(
                content=json.dumps(suggestions),
                model=response.model,
                tokens_used=0,
                cost=0
            )
            self.cache.set(cache_key, cache_response)
        
        return suggestions
    
    def _parse_feature_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """Parse feature suggestions from LLM response."""
        suggestions = []
        
        # Extract code blocks
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        
        # Extract descriptions
        lines = text.split('\n')
        current_suggestion = {}
        
        for i, line in enumerate(lines):
            if re.match(r"^\d+\.", line) or line.startswith("-"):
                # New suggestion
                if current_suggestion:
                    suggestions.append(current_suggestion)
                
                current_suggestion = {
                    "name": f"feature_{len(suggestions) + 1}",
                    "description": line.strip(),
                    "code": "",
                    "importance": "medium"
                }
            
            # Check for importance keywords
            if "high importance" in line.lower() or "critical" in line.lower():
                current_suggestion["importance"] = "high"
            elif "low importance" in line.lower() or "optional" in line.lower():
                current_suggestion["importance"] = "low"
        
        # Add last suggestion
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        # Assign code blocks to suggestions
        for i, code in enumerate(code_blocks[:len(suggestions)]):
            suggestions[i]["code"] = code.strip()
        
        return suggestions
    
    async def explain_model(self, model_name: str, metrics: Dict[str, float],
                          feature_importance: Dict[str, float] = None) -> str:
        """Generate natural language explanation of model."""
        prompt = self.prompts["explain_model"].format(
            model_name=model_name,
            metrics=json.dumps(metrics, indent=2),
            feature_importance=json.dumps(feature_importance, indent=2) if feature_importance else "Not available"
        )
        
        response = await self.llm.generate(prompt, temperature=0.5, max_tokens=800)
        return response.content
    
    async def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                           X: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze prediction errors with LLM insights."""
        # Calculate error statistics
        errors = np.abs(y_true - y_pred)
        error_stats = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "error_quantiles": {
                "25%": float(np.percentile(errors, 25)),
                "50%": float(np.percentile(errors, 50)),
                "75%": float(np.percentile(errors, 75)),
                "95%": float(np.percentile(errors, 95))
            }
        }
        
        # Get worst predictions
        worst_indices = np.argsort(errors)[-10:]
        sample_errors = []
        
        for idx in worst_indices:
            error_detail = {
                "index": int(idx),
                "true": float(y_true[idx]),
                "predicted": float(y_pred[idx]),
                "error": float(errors[idx])
            }
            
            if X is not None:
                error_detail["features"] = X.iloc[idx].to_dict()
            
            sample_errors.append(error_detail)
        
        # Generate analysis
        prompt = self.prompts["error_analysis"].format(
            error_stats=json.dumps(error_stats, indent=2),
            sample_errors=json.dumps(sample_errors[:5], indent=2)  # Limit to 5 for token efficiency
        )
        
        response = await self.llm.generate(prompt, temperature=0.5, max_tokens=1000)
        
        return {
            "statistics": error_stats,
            "worst_predictions": sample_errors,
            "analysis": response.content,
            "recommendations": self._extract_recommendations(response.content)
        }
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text."""
        recommendations = []
        
        # Look for numbered or bulleted recommendations
        patterns = [
            r"\d+\.\s*(.+)",  # 1. Recommendation
            r"[-•]\s*(.+)",   # - or • Recommendation
            r"Recommendation:\s*(.+)",  # Explicit recommendation
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            recommendations.extend(matches)
        
        # Clean and deduplicate
        recommendations = list(set([r.strip() for r in recommendations if len(r.strip()) > 10]))
        
        return recommendations[:10]  # Limit to top 10
    
    async def generate_code(self, task_description: str, 
                          data_sample: pd.DataFrame = None) -> str:
        """Generate AutoML code from natural language."""
        prompt = f"""
        Generate Python code for this AutoML task:
        
        Task: {task_description}
        
        {f"Data sample columns: {list(data_sample.columns)}" if data_sample is not None else ""}
        {f"Data shape: {data_sample.shape}" if data_sample is not None else ""}
        
        Use the automl_platform library.
        Include proper imports, configuration, and error handling.
        Add comments explaining each step.
        """
        
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=1500)
        
        # Extract and clean code
        code = self._extract_code(response.content)
        
        # Validate code structure
        if "import" not in code:
            code = "from automl_platform import AutoMLOrchestrator, AutoMLConfig\nimport pandas as pd\n\n" + code
        
        return code
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from text."""
        # Look for code blocks
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        # Fallback: extract lines that look like code
        lines = text.split('\n')
        code_lines = []
        
        for line in lines:
            # Simple heuristic for Python code
            if any([
                line.strip().startswith(('import ', 'from ', 'def ', 'class ')),
                '=' in line and not line.strip().startswith('#'),
                line.strip().startswith(('df', 'X', 'y', 'model', 'config')),
                re.match(r'^\s{4,}', line)  # Indented lines
            ]):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    async def generate_report(self, experiment_data: Dict[str, Any], 
                            format: str = "markdown") -> str:
        """Generate comprehensive AutoML report."""
        prompt = self.prompts["generate_report"].format(
            experiment_id=experiment_data.get('experiment_id', 'Unknown'),
            best_model=experiment_data.get('best_model', 'Unknown'),
            performance=json.dumps(experiment_data.get('metrics', {}), indent=2),
            key_features=json.dumps(experiment_data.get('top_features', []), indent=2)
        )
        
        response = await self.llm.generate(prompt, temperature=0.5, max_tokens=2000)
        
        # Format based on output type
        if format == "markdown":
            return response.content
        elif format == "html":
            return self._markdown_to_html(response.content)
        else:
            return response.content
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML."""
        try:
            import markdown
            return markdown.markdown(markdown_text)
        except ImportError:
            # Simple conversion
            html = markdown_text
            html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
            html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
            html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
            html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
            html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
            html = html.replace('\n\n', '</p><p>').replace('\n', '<br>')
            return f"<p>{html}</p>"
    
    async def chat(self, message: str, context: Dict[str, Any] = None) -> str:
        """Interactive chat interface."""
        # Add context to message
        if context:
            message = f"""
            Context:
            {json.dumps(context, indent=2)}
            
            User Question: {message}
            """
        
        # Search for relevant context in RAG
        if self.rag:
            relevant_docs = self.rag.search(message, k=3)
            if relevant_docs:
                message += "\n\nRelevant Information:\n"
                for doc, score in relevant_docs:
                    message += f"- {doc[:200]}...\n"
        
        # Generate response
        response = await self.llm.generate(message, temperature=0.7, max_tokens=1000)
        
        return response.content
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        return {
            "total_tokens": self.llm.total_tokens,
            "total_cost": self.llm.total_cost,
            "provider": self.config['provider'],
            "model": self.config['model_name'],
            "cache_enabled": self.cache is not None,
            "rag_enabled": self.rag is not None
        }


# Example usage and testing
async def main():
    """Example usage of LLM module."""
    
    # Configuration
    config = {
        'provider': 'openai',
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_name': 'gpt-4',
        'cache_responses': True,
        'enable_rag': True,
        'vector_store': 'chromadb',
        'embedding_model': 'text-embedding-ada-002',
        'prompts_dir': './prompts'
    }
    
    # Initialize assistant
    assistant = AutoMLLLMAssistant(config)
    
    # Create sample data
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [30000, 45000, 55000, 65000, 80000],
        'education_years': [12, 14, 16, 16, 18],
        'purchased': [0, 0, 1, 1, 1]
    })
    
    # Test feature suggestions
    print("Testing feature suggestions...")
    features = await assistant.suggest_features(df, 'purchased', 'classification')
    print(f"Suggested {len(features)} features")
    for feat in features:
        print(f"- {feat['description']}")
    
    # Test model explanation
    print("\nTesting model explanation...")
    explanation = await assistant.explain_model(
        "RandomForestClassifier",
        {"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
        {"age": 0.3, "income": 0.5, "education_years": 0.2}
    )
    print(explanation[:500] + "...")
    
    # Test data cleaning
    print("\nTesting data cleaning agent...")
    analysis = await assistant.cleaning_agent.analyze_data_issues(df)
    print(f"Found issues: {analysis['issues']}")
    
    # Test code generation
    print("\nTesting code generation...")
    code = await assistant.generate_code(
        "Train a classification model to predict customer purchases based on demographics",
        df
    )
    print("Generated code:")
    print(code[:500] + "...")
    
    # Test chat
    print("\nTesting chat interface...")
    response = await assistant.chat(
        "What's the best model for imbalanced classification?",
        {"dataset_size": 10000, "class_ratio": "1:10"}
    )
    print(response[:500] + "...")
    
    # Show usage stats
    print("\nUsage Statistics:")
    print(assistant.get_usage_stats())


if __name__ == "__main__":
    asyncio.run(main())

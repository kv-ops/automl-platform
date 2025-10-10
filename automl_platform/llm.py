"""
Enhanced LLM Integration Module with Advanced RAG, WebSocket Chat, and Improved Data Cleaning
Complete implementation for production AutoML platform
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
from functools import lru_cache

from .safe_code_execution import (
    UnsafeCodeExecutionError,
    ensure_safe_cleaning_code,
    execution_not_allowed_message,
)

# LLM providers
import importlib.util
from typing import TYPE_CHECKING, Any

openai_spec = importlib.util.find_spec("openai")
if openai_spec is not None:
    import openai  # type: ignore
else:
    openai = None  # type: ignore[assignment]

anthropic_spec = importlib.util.find_spec("anthropic")
if anthropic_spec is not None:
    from anthropic import Anthropic
else:
    Anthropic = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from anthropic import Anthropic as _Anthropic

chromadb_spec = importlib.util.find_spec("chromadb")
if chromadb_spec is not None:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
else:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]

faiss_spec = importlib.util.find_spec("faiss")
if faiss_spec is not None:
    import faiss  # type: ignore
else:
    faiss = None  # type: ignore[assignment]

langchain_spec = importlib.util.find_spec("langchain")
if langchain_spec is not None:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore
    from langchain.schema import Document  # type: ignore
    from langchain.chains import RetrievalQA  # type: ignore
    from langchain.memory import ConversationBufferMemory  # type: ignore
    from langchain.vectorstores import Chroma, FAISS  # type: ignore
else:
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    RetrievalQA = None  # type: ignore[assignment]
    ConversationBufferMemory = None  # type: ignore[assignment]
    Chroma = None  # type: ignore[assignment]
    FAISS = None  # type: ignore[assignment]

websockets_spec = importlib.util.find_spec("websockets")
if websockets_spec is not None:
    import websockets  # type: ignore
else:
    websockets = None  # type: ignore[assignment]

# Vector stores and embeddings
import pickle

# WebSocket support
from aiohttp import web

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
    """OpenAI GPT provider with GPT-4 Vision support."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo-preview"):
        super().__init__(api_key, model_name)
        
        if openai is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider but is not installed. "
                "Install it via `pip install openai>=1.10.0`."
            )

        openai.api_key = api_key
        
        # Updated pricing for 2024
        self.pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    
    async def generate(self, prompt: str, temperature: float = 0.7, 
                       max_tokens: int = 1000, **kwargs) -> LLMResponse:
        """Generate response from OpenAI."""
        try:
            # Support for function calling
            functions = kwargs.pop('functions', None)
            
            messages = [{"role": "user", "content": prompt}]
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            if functions:
                params["functions"] = functions
                params["function_call"] = "auto"
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                **params
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            # Calculate cost
            model_pricing = self.pricing.get(self.model_name, self.pricing["gpt-3.5-turbo"])
            cost = (response.usage.prompt_tokens * model_pricing["input"] + 
                   response.usage.completion_tokens * model_pricing["output"]) / 1000
            
            self.total_tokens += tokens
            self.total_cost += cost
            
            # Handle function calls
            function_call = response.choices[0].message.get("function_call")
            
            return LLMResponse(
                content=content or json.dumps(function_call) if function_call else content,
                model=self.model_name,
                tokens_used=tokens,
                cost=cost,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "function_call": function_call
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with Claude 3 support."""

    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        super().__init__(api_key, model_name)
        if Anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider but is not installed. "
                "Install it via `pip install anthropic`."
            )

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
    """Enhanced Redis-based cache for LLM responses with semantic similarity."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        if OpenAIEmbeddings is None:
            raise ImportError(
                "langchain is required for semantic caching but is not installed. "
                "Install it via `pip install langchain`."
            )
        self.embeddings = OpenAIEmbeddings()
    
    def _get_key(self, prompt: str, model: str) -> str:
        """Generate cache key."""
        content = f"{model}:{prompt}"
        return f"llm:cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    async def get_semantic(self, prompt: str, model: str, threshold: float = 0.95) -> Optional[LLMResponse]:
        """Get cached response using semantic similarity."""
        try:
            # Get prompt embedding
            prompt_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, prompt
            )
            
            # Search for similar prompts in cache
            pattern = f"llm:cache:*"
            for key in self.redis_client.scan_iter(match=pattern):
                cached_data = self.redis_client.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    if data.get('model') == model and 'embedding' in data:
                        # Calculate similarity
                        similarity = np.dot(prompt_embedding, data['embedding'])
                        if similarity > threshold:
                            response_dict = data.copy()
                            response_dict['cached'] = True
                            del response_dict['embedding']
                            return LLMResponse(**response_dict)
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}")
        
        return None
    
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
    
    def set(self, prompt: str, response: LLMResponse, embedding: Optional[List[float]] = None):
        """Cache response with optional embedding."""
        key = self._get_key(prompt, response.model)
        try:
            data = asdict(response)
            if embedding:
                data['embedding'] = embedding
            self.redis_client.setex(
                key, self.ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")


class AdvancedRAGSystem:
    """Advanced Retrieval-Augmented Generation system with hybrid search."""
    
    def __init__(self, vector_store: str = "chromadb", embedding_model: str = "text-embedding-ada-002"):
        self.vector_store_type = vector_store
        self.embedding_model = embedding_model
        if OpenAIEmbeddings is None or RecursiveCharacterTextSplitter is None or ConversationBufferMemory is None:
            raise ImportError(
                "langchain is required for AdvancedRAGSystem but is not installed. "
                "Install it via `pip install langchain`."
            )
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        
        # Initialize vector store
        if vector_store == "chromadb":
            if chromadb is None or Settings is None or Chroma is None:
                raise ImportError(
                    "chromadb and langchain-vectorstores are required for Chroma vector store support. "
                    "Install them via `pip install chromadb langchain`."
                )
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            self.collection = self.client.get_or_create_collection("automl_docs")
            self.vector_store = Chroma(
                client=self.client,
                collection_name="automl_docs",
                embedding_function=self.embeddings
            )
        elif vector_store == "faiss":
            if faiss is None or FAISS is None:
                raise ImportError(
                    "faiss and langchain-vectorstores are required for FAISS vector store support. "
                    "Install them via `pip install faiss-cpu langchain`."
                )
            self.index = None
            self.documents = []
            self.metadata = []
            self.vector_store = None
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None, 
                     doc_type: str = "general"):
        """Add documents to vector store with metadata."""
        # Split documents
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            for j, chunk in enumerate(chunks):
                texts.append(chunk)
                meta = metadata[i].copy() if metadata else {}
                meta.update({
                    "doc_type": doc_type,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat()
                })
                metadatas.append(meta)
        
        if self.vector_store_type == "chromadb":
            self.vector_store.add_texts(texts, metadatas=metadatas)
        
        elif self.vector_store_type == "faiss":
            # Generate embeddings
            embeddings = np.array(self.embeddings.embed_documents(texts))
            
            # Initialize or update FAISS index
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                # Add metadata index
                self.index = faiss.IndexIDMap(self.index)
            
            # Add to index with IDs
            ids = np.arange(len(self.documents), len(self.documents) + len(texts))
            self.index.add_with_ids(embeddings, ids)
            self.documents.extend(texts)
            self.metadata.extend(metadatas)
            
            # Create FAISS vector store
            self.vector_store = FAISS(
                embedding_function=self.embeddings.embed_query,
                index=self.index,
                docstore=self.documents,
                index_to_docstore_id={i: i for i in range(len(self.documents))}
            )
    
    def hybrid_search(self, query: str, k: int = 5, 
                     filter_dict: Dict = None) -> List[Tuple[str, float, Dict]]:
        """Hybrid search combining semantic and keyword search."""
        results = []
        
        # Semantic search
        query_embedding = self.embeddings.embed_query(query)
        
        if self.vector_store_type == "chromadb":
            # ChromaDB search with filters
            where_clause = filter_dict if filter_dict else {}
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause
            )
            
            if search_results["documents"]:
                for doc, dist, meta in zip(
                    search_results["documents"][0],
                    search_results["distances"][0],
                    search_results["metadatas"][0]
                ):
                    results.append((doc, float(dist), meta))
        
        elif self.vector_store_type == "faiss" and self.index:
            # FAISS search
            query_vec = np.array([query_embedding])
            distances, indices = self.index.search(query_vec, k)
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        float(dist),
                        self.metadata[idx] if idx < len(self.metadata) else {}
                    ))
        
        # Keyword search enhancement
        keywords = self._extract_keywords(query)
        for keyword in keywords:
            for i, doc in enumerate(self.documents[:k*2]):  # Search more docs
                if keyword.lower() in doc.lower():
                    # Boost keyword matches
                    found = False
                    for j, (d, _, _) in enumerate(results):
                        if d == doc:
                            results[j] = (d, results[j][1] * 0.8, results[j][2])
                            found = True
                            break
                    if not found and len(results) < k:
                        results.append((doc, 1.0, self.metadata[i] if i < len(self.metadata) else {}))
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        import nltk
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text.lower())
            keywords = [w for w in word_tokens if w not in stop_words and w.isalnum()]
            return keywords[:5]  # Top 5 keywords
        except:
            # Fallback to simple split
            return text.lower().split()[:5]
    
    def create_qa_chain(self, llm_provider: LLMProvider):
        """Create a QA chain with the vector store."""
        from langchain.chains import ConversationalRetrievalChain
        
        # Create LangChain LLM wrapper
        class LLMWrapper:
            def __init__(self, provider):
                self.provider = provider
            
            async def agenerate(self, prompts, **kwargs):
                responses = []
                for prompt in prompts:
                    response = await self.provider.generate(prompt, **kwargs)
                    responses.append(response.content)
                return responses
        
        llm = LLMWrapper(llm_provider)
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True
        )
        
        return qa_chain


class EnhancedDataCleaningAgent:
    """Enhanced intelligent data cleaning agent with advanced capabilities."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.cleaning_history = []
        self.cleaning_patterns = self._load_cleaning_patterns()
    
    def _load_cleaning_patterns(self) -> Dict[str, Any]:
        """Load common data cleaning patterns."""
        return {
            "missing_values": {
                "numeric": ["mean", "median", "mode", "forward_fill", "backward_fill", "interpolate"],
                "categorical": ["mode", "constant", "forward_fill", "backward_fill"],
                "time_series": ["interpolate", "forward_fill", "backward_fill", "seasonal_decompose"]
            },
            "outliers": {
                "methods": ["iqr", "zscore", "isolation_forest", "local_outlier_factor"],
                "actions": ["remove", "cap", "transform", "flag"]
            },
            "encoding": {
                "categorical": ["one_hot", "label", "target", "ordinal", "binary"],
                "text": ["tfidf", "count", "word2vec", "bert"]
            },
            "scaling": {
                "methods": ["standard", "minmax", "robust", "maxabs", "quantile"]
            }
        }
    
    async def analyze_data_issues(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze data quality issues with detailed recommendations."""
        sample = df.head(sample_size)
        
        # Comprehensive data profiling
        profile = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
            "unique_counts": df.nunique().to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "sample": sample.to_dict('records')[:5]
        }
        
        # Statistical analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile["statistics"] = df[numeric_cols].describe().to_dict()
            
            # Detect skewness
            from scipy import stats
            profile["skewness"] = {col: stats.skew(df[col].dropna()) for col in numeric_cols}
            
            # Detect outliers
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                if outlier_count > 0:
                    outliers[col] = outlier_count
            profile["outliers"] = outliers
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            profile["high_cardinality"] = {
                col: df[col].nunique() 
                for col in categorical_cols 
                if df[col].nunique() > len(df) * 0.5
            }
        
        # Create comprehensive prompt
        prompt = f"""
        Analyze this dataset for quality issues and provide detailed recommendations:
        
        Dataset Profile:
        - Shape: {profile['shape']}
        - Memory Usage: {profile['memory_usage']:.2f} MB
        - Missing Values: {json.dumps(profile['missing_percent'], indent=2)}
        - Duplicates: {profile['duplicates']}
        - Outliers: {json.dumps(profile.get('outliers', {}), indent=2)}
        - Skewness: {json.dumps(profile.get('skewness', {}), indent=2)}
        - High Cardinality: {json.dumps(profile.get('high_cardinality', {}), indent=2)}
        
        Provide a structured JSON response with:
        {{
            "critical_issues": [...],
            "data_quality_score": 0-100,
            "cleaning_steps": [
                {{
                    "step": 1,
                    "issue": "...",
                    "action": "...",
                    "code": "...",
                    "impact": "high/medium/low"
                }}
            ],
            "feature_engineering_opportunities": [...],
            "warnings": [...],
            "estimated_cleaning_time": "minutes"
        }}
        """
        
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=1500)
        
        try:
            analysis = json.loads(response.content)
            analysis["profile"] = profile
        except:
            analysis = {
                "critical_issues": ["Could not parse LLM response"],
                "data_quality_score": self._calculate_quality_score(profile),
                "cleaning_steps": self._generate_default_cleaning_steps(profile),
                "feature_engineering_opportunities": [],
                "warnings": [],
                "profile": profile
            }
        
        return analysis
    
    def _calculate_quality_score(self, profile: Dict) -> float:
        """Calculate data quality score."""
        score = 100.0
        
        # Penalize for missing values
        max_missing = max(profile.get('missing_percent', {}).values(), default=0)
        score -= min(max_missing, 30)
        
        # Penalize for duplicates
        duplicate_ratio = profile.get('duplicates', 0) / profile['shape'][0]
        score -= duplicate_ratio * 20
        
        # Penalize for outliers
        outlier_ratio = sum(profile.get('outliers', {}).values()) / (profile['shape'][0] * len(profile.get('outliers', {})))
        score -= outlier_ratio * 10
        
        return max(0, score)
    
    def _generate_default_cleaning_steps(self, profile: Dict) -> List[Dict]:
        """Generate default cleaning steps based on profile."""
        steps = []
        
        # Handle missing values
        for col, missing_pct in profile.get('missing_percent', {}).items():
            if missing_pct > 0:
                steps.append({
                    "step": len(steps) + 1,
                    "issue": f"Missing values in {col} ({missing_pct:.1f}%)",
                    "action": "Fill with median" if col in profile.get('statistics', {}) else "Fill with mode",
                    "code": f"df['{col}'].fillna(df['{col}'].median(), inplace=True)" if col in profile.get('statistics', {}) else f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)",
                    "impact": "high" if missing_pct > 20 else "medium"
                })
        
        # Handle duplicates
        if profile.get('duplicates', 0) > 0:
            steps.append({
                "step": len(steps) + 1,
                "issue": f"Duplicate rows ({profile['duplicates']})",
                "action": "Remove duplicates",
                "code": "df.drop_duplicates(inplace=True)",
                "impact": "medium"
            })
        
        return steps
    
    async def suggest_cleaning_code(self, df: pd.DataFrame, issue: str) -> str:
        """Generate Python code to fix data issues with validation."""
        prompt = f"""
        Generate production-ready Python code to fix this data issue:
        
        Issue: {issue}
        
        Dataset info:
        - Columns: {list(df.columns)}
        - Shape: {df.shape}
        - Data types: {df.dtypes.to_dict()}
        
        Requirements:
        1. Use pandas and numpy
        2. Handle edge cases
        3. Include error handling
        4. Add logging
        5. Preserve data types
        6. Include comments
        
        The dataframe variable is 'df'.
        """
        
        response = await self.llm.generate(prompt, temperature=0.2, max_tokens=1000)
        
        # Extract and validate code
        code = self._extract_code(response.content)
        
        # Add safety checks
        if "exec" in code or "eval" in code or "__import__" in code:
            logger.warning("Potentially unsafe code detected")
            code = "# Unsafe code detected - manual review required\n" + code
        
        # Log cleaning action
        self.cleaning_history.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "code": code,
            "validated": self._validate_code(code)
        })
        
        return code
    
    def _validate_code(self, code: str) -> bool:
        """Validate Python code syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    async def auto_clean(self, df: pd.DataFrame, target_quality_score: float = 80.0) -> pd.DataFrame:
        """Automatically clean data to reach target quality score."""
        df_cleaned = df.copy()
        max_iterations = 5
        
        for iteration in range(max_iterations):
            # Analyze current state
            analysis = await self.analyze_data_issues(df_cleaned)
            
            current_score = analysis.get('data_quality_score', 0)
            logger.info(f"Iteration {iteration + 1}: Quality score = {current_score:.1f}")
            
            if current_score >= target_quality_score:
                logger.info(f"Target quality score reached: {current_score:.1f}")
                break
            
            # Apply cleaning steps
            for step in analysis.get('cleaning_steps', [])[:3]:  # Apply top 3 steps per iteration
                code = step.get('code', '')
                if not code or not self._validate_code(code):
                    continue

                try:
                    ensure_safe_cleaning_code(code)
                except UnsafeCodeExecutionError as exc:
                    logger.error(f"Unsafe cleaning step rejected: {exc}")
                    raise

                logger.warning(execution_not_allowed_message())
        
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
    """Main LLM assistant for AutoML platform with WebSocket support."""
    
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
        
        # Initialize Advanced RAG
        self.rag = AdvancedRAGSystem(
            vector_store=config.get('vector_store', 'chromadb'),
            embedding_model=config.get('embedding_model', 'text-embedding-ada-002')
        ) if config.get('enable_rag', True) else None
        
        # Initialize agents
        self.cleaning_agent = EnhancedDataCleaningAgent(self.llm)
        
        # Initialize WebSocket handler
        self.websocket_handler = WebSocketChatHandler(self)
        
        # Load prompts
        self.prompts = self._load_enhanced_prompts()
    
    def _load_enhanced_prompts(self) -> Dict[str, str]:
        """Load enhanced prompt templates."""
        prompts_dir = Path(self.config.get('prompts_dir', './prompts'))
        prompts = {}
        
        # Enhanced prompts for better suggestions
        default_prompts = {
            "feature_engineering": """
                You are an expert data scientist specializing in feature engineering.
                
                Dataset Info:
                {dataset_info}
                
                Task Type: {task_type}
                Target Column: {target_column}
                
                Provide specific, actionable feature engineering suggestions:
                1. Time-based features if applicable
                2. Interaction features between correlated columns
                3. Polynomial features for non-linear relationships
                4. Domain-specific transformations
                5. Text/categorical encoding strategies
                
                Include Python code using pandas and numpy.
                Format as JSON with: name, description, code, importance (high/medium/low)
            """,
            
            "model_selection": """
                Based on this dataset, recommend the best ML models:
                
                Dataset Characteristics:
                {data_characteristics}
                
                Consider:
                - Data size and dimensionality
                - Feature types (numeric, categorical, text, time)
                - Class imbalance if classification
                - Potential for non-linear relationships
                - Need for interpretability
                
                Rank models by expected performance and provide rationale.
                Include both traditional and neural network approaches.
            """,
            
            "column_analysis": """
                Analyze these columns for ML modeling:
                
                Columns: {columns}
                Sample Data: {sample_data}
                
                For each column, determine:
                1. Data type and distribution
                2. Potential as feature or target
                3. Required preprocessing
                4. Feature importance potential
                5. Encoding strategy if categorical
                
                Return structured JSON analysis.
            """,
            
            "code_generation": """
                Generate production-ready AutoML code:
                
                Task: {task_description}
                Dataset Info: {dataset_info}
                
                Requirements:
                1. Use the automl_platform library
                2. Include proper error handling
                3. Add logging and progress tracking
                4. Implement data validation
                5. Include model explainability
                6. Add performance monitoring
                
                Structure:
                - Imports
                - Configuration
                - Data loading and validation
                - Feature engineering
                - Model training with HPO
                - Evaluation and explainability
                - Deployment preparation
            """,
            
            "explain_model": """
                Explain this model's performance for non-technical stakeholders:
                
                Model: {model_name}
                Metrics: {metrics}
                Feature Importance: {feature_importance}
                
                Provide:
                1. Plain English explanation of what the model does
                2. Key performance indicators interpretation
                3. Most important factors driving predictions
                4. Strengths and limitations
                5. Recommendations for improvement
                6. Business impact assessment
            """,
            
            "error_analysis": """
                Analyze these prediction errors to improve model:
                
                Error Statistics: {error_stats}
                Sample Errors: {sample_errors}
                
                Identify:
                1. Systematic patterns in errors
                2. Feature ranges with poor performance
                3. Potential data quality issues
                4. Model bias indicators
                5. Suggestions for improvement
                
                Provide actionable recommendations with priority levels.
            """,
            
            "generate_report": """
                Generate comprehensive AutoML experiment report:
                
                Experiment ID: {experiment_id}
                Best Model: {best_model}
                Performance: {performance}
                Key Features: {key_features}
                
                Structure:
                # Executive Summary
                - Key findings and recommendations
                - Business impact
                
                # Technical Details
                - Data overview and quality
                - Feature engineering performed
                - Models evaluated
                - Best model architecture
                - Performance metrics
                
                # Insights
                - Feature importance analysis
                - Error analysis
                - Model limitations
                
                # Recommendations
                - Deployment considerations
                - Monitoring strategy
                - Improvement opportunities
                
                Format as professional markdown report.
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
        """Suggest advanced feature engineering with code."""
        # Check cache
        cache_key = f"features_{hashlib.sha256(df.to_csv().encode()).hexdigest()[:8]}_{target_column}"
        
        if self.cache:
            cached = self.cache.get(cache_key, "feature_suggestions")
            if cached:
                return json.loads(cached.content)
        
        # Prepare comprehensive dataset info
        dataset_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(3).to_dict('records'),
            "correlation_with_target": df.corrwith(df[target_column]).to_dict() if target_column in df else {},
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict(),
            "datetime_columns": [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])],
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns)
        }
        
        # Generate prompt
        prompt = self.prompts["feature_engineering"].format(
            dataset_info=json.dumps(dataset_info, indent=2),
            task_type=task_type,
            target_column=target_column
        )
        
        # Add RAG context if available
        if self.rag:
            context = self.rag.hybrid_search(
                f"feature engineering {task_type} {' '.join(df.columns[:5])}", 
                k=3
            )
            if context:
                prompt += "\n\nRelevant examples:\n" + "\n".join([doc for doc, _, _ in context])
        
        # Generate suggestions
        response = await self.llm.generate(prompt, temperature=0.7, max_tokens=2000)
        
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
        
        # Try to parse as JSON first
        try:
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
                return suggestions
        except:
            pass
        
        # Fallback to text parsing
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        
        lines = text.split('\n')
        current_suggestion = {}
        
        for i, line in enumerate(lines):
            if re.match(r"^\d+\.", line) or line.startswith("-"):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                
                current_suggestion = {
                    "name": f"feature_{len(suggestions) + 1}",
                    "description": line.strip(),
                    "code": "",
                    "importance": "medium"
                }
            
            if "high importance" in line.lower() or "critical" in line.lower():
                current_suggestion["importance"] = "high"
            elif "low importance" in line.lower() or "optional" in line.lower():
                current_suggestion["importance"] = "low"
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        # Assign code blocks
        for i, code in enumerate(code_blocks[:len(suggestions)]):
            suggestions[i]["code"] = code.strip()
        
        return suggestions
    
    async def chat(self, message: str, context: Dict[str, Any] = None) -> str:
        """Interactive chat interface with context awareness."""
        # Add context to message
        if context:
            message = f"""
            Context:
            {json.dumps(context, indent=2)}
            
            User Question: {message}
            """
        
        # Search for relevant context in RAG
        if self.rag:
            relevant_docs = self.rag.hybrid_search(message, k=5)
            if relevant_docs:
                message += "\n\nRelevant Information:\n"
                for doc, score, metadata in relevant_docs:
                    message += f"- {doc[:200]}... (relevance: {1-score:.2f})\n"
        
        # Generate response
        response = await self.llm.generate(message, temperature=0.7, max_tokens=1000)
        
        return response.content
    
    async def generate_code(self, task_description: str, 
                          data_sample: pd.DataFrame = None) -> str:
        """Generate complete AutoML code from natural language."""
        dataset_info = ""
        if data_sample is not None:
            dataset_info = f"""
            Columns: {list(data_sample.columns)}
            Shape: {data_sample.shape}
            Dtypes: {data_sample.dtypes.to_dict()}
            Sample: {data_sample.head(2).to_dict('records')}
            """
        
        prompt = self.prompts["code_generation"].format(
            task_description=task_description,
            dataset_info=dataset_info
        )
        
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=2000)
        
        # Extract and validate code
        code = self._extract_code(response.content)
        
        # Add imports if missing
        if "import" not in code:
            code = """from automl_platform import AutoMLOrchestrator, AutoMLConfig
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" + code
        
        return code
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from text."""
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        lines = text.split('\n')
        code_lines = []
        
        for line in lines:
            if any([
                line.strip().startswith(('import ', 'from ', 'def ', 'class ')),
                '=' in line and not line.strip().startswith('#'),
                line.strip().startswith(('df', 'X', 'y', 'model', 'config')),
                re.match(r'^\s{4,}', line)
            ]):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive LLM usage statistics."""
        return {
            "total_tokens": self.llm.total_tokens,
            "total_cost": self.llm.total_cost,
            "provider": self.config['provider'],
            "model": self.config['model_name'],
            "cache_enabled": self.cache is not None,
            "rag_enabled": self.rag is not None,
            "cleaning_history": len(self.cleaning_agent.cleaning_history),
            "websocket_sessions": len(self.websocket_handler.user_sessions) if hasattr(self, 'websocket_handler') else 0
        }


class WebSocketChatHandler:
    """WebSocket handler for real-time LLM chat."""
    
    def __init__(self, llm_assistant: 'AutoMLLLMAssistant'):
        self.llm_assistant = llm_assistant
        self.connections = set()
        self.user_sessions = {}
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        # Register connection
        self.connections.add(websocket)
        session_id = hashlib.sha256(str(websocket.remote_address).encode()).hexdigest()[:8]
        
        # Initialize session
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                "history": [],
                "context": {},
                "created_at": datetime.now().isoformat()
            }
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "Connected to AutoML Assistant",
                "session_id": session_id
            }))
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(websocket, message, session_id)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Unregister connection
            self.connections.remove(websocket)
    
    async def handle_message(self, websocket, message: str, session_id: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "chat")
            
            if msg_type == "chat":
                # Regular chat message
                user_message = data.get("message", "")
                context = data.get("context", {})
                
                # Update session
                self.user_sessions[session_id]["history"].append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate response
                response = await self.llm_assistant.chat(
                    user_message,
                    context={**self.user_sessions[session_id]["context"], **context}
                )
                
                # Update history
                self.user_sessions[session_id]["history"].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send response
                await websocket.send(json.dumps({
                    "type": "response",
                    "message": response,
                    "session_id": session_id
                }))
                
            elif msg_type == "code_generation":
                # Generate code request
                task = data.get("task", "")
                df_info = data.get("dataframe_info", None)
                
                code = await self.llm_assistant.generate_code(task, df_info)
                
                await websocket.send(json.dumps({
                    "type": "code",
                    "code": code,
                    "session_id": session_id
                }))
                
            elif msg_type == "data_analysis":
                # Analyze data request
                df_json = data.get("dataframe", {})
                df = pd.DataFrame(df_json)
                
                analysis = await self.llm_assistant.cleaning_agent.analyze_data_issues(df)
                
                await websocket.send(json.dumps({
                    "type": "analysis",
                    "analysis": analysis,
                    "session_id": session_id
                }))
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e),
                "session_id": session_id
            }))
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        if self.connections:
            await asyncio.gather(
                *[ws.send(json.dumps(message)) for ws in self.connections],
                return_exceptions=True
            )

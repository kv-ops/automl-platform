"""
Validator Agent - Validates data against sector standards using web search
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
import aiohttp
import importlib.util
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import time
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path

from .agent_config import AgentConfig, AgentType
from .prompts.validator_prompts import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_PROMPT

_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    from openai import AsyncOpenAI  # type: ignore
else:
    AsyncOpenAI = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import AsyncOpenAI as _AsyncOpenAIType

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    Agent responsible for validating data against sector-specific standards
    Uses OpenAI Assistant with Code Interpreter and Web Search capabilities
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize Validator Agent"""
        self.config = config
        if AsyncOpenAI is not None and config.openai_api_key:
            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            self.client = None
            if AsyncOpenAI is None:
                logger.warning(
                    "AsyncOpenAI client unavailable because the 'openai' package is not installed. "
                    "ValidatorAgent will rely on local validation heuristics."
                )
            else:
                logger.warning("OpenAI API key missing; ValidatorAgent will use local validation heuristics only.")
        self.assistant = None
        self.assistant_id = config.get_assistant_id(AgentType.VALIDATOR)
        
        # Cache for web search results
        self.search_cache = {}
        self.cache_dir = Path(config.cache_dir) / "validator"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize assistant
        if self.client is not None:
            asyncio.create_task(self._initialize_assistant())
    
    async def _initialize_assistant(self):
        """Create or retrieve OpenAI Assistant"""
        try:
            if self.client is None:
                logger.debug("ValidatorAgent _initialize_assistant skipped because client is unavailable.")
                return

            if self.assistant_id:
                self.assistant = await self.client.beta.assistants.retrieve(
                    assistant_id=self.assistant_id
                )
                logger.info(f"Retrieved existing Validator assistant: {self.assistant_id}")
            else:
                self.assistant = await self.client.beta.assistants.create(
                    name="Data Validator Agent",
                    instructions=VALIDATOR_SYSTEM_PROMPT,
                    model=self.config.model,
                    tools=self.config.get_agent_tools(AgentType.VALIDATOR)
                )
                self.assistant_id = self.assistant.id
                self.config.save_assistant_id(AgentType.VALIDATOR, self.assistant_id)
                logger.info(f"Created new Validator assistant: {self.assistant_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize validator assistant: {e}")
    
    async def validate(self, df: pd.DataFrame, profile_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against sector standards
        
        Args:
            df: Input dataframe
            profile_report: Report from profiler agent
            
        Returns:
            Dictionary containing validation results
        """
        try:
            if self.client is None:
                logger.info("ValidatorAgent falling back to basic validation because OpenAI client is unavailable.")
                sector = self.config.user_context.get("secteur_activite", "general")
                return self._basic_validation(df, sector)

            # Ensure assistant is initialized
            if not self.assistant:
                await self._initialize_assistant()
            
            # Get sector-specific validation references
            sector = self.config.user_context.get("secteur_activite", "general")
            references = await self._search_sector_standards(sector, df.columns.tolist())
            
            # Create a thread
            thread = await self.client.beta.threads.create()
            
            # Prepare validation request
            message_content = VALIDATOR_USER_PROMPT.format(
                sector=sector,
                columns=json.dumps(list(df.columns)),
                profile_summary=json.dumps(profile_report, indent=2),
                references=json.dumps(references, indent=2),
                target_variable=self.config.user_context.get("target_variable", "unknown")
            )
            
            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            
            # Run the assistant with function calling
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Handle function calls for web search
            result = await self._handle_run_with_functions(thread.id, run.id, df)
            
            # Parse and return results
            return self._parse_validation_results(result, references)
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return self._basic_validation(df, sector)
    
    async def _search_sector_standards(self, sector: str, columns: List[str]) -> Dict[str, Any]:
        """Search for sector-specific standards and references"""
        references = {
            "sector": sector,
            "standards": [],
            "column_mappings": {},
            "sources": []
        }
        
        # Get sector keywords
        keywords = self.config.get_sector_keywords(sector)
        
        # Build search queries
        queries = []
        
        # Sector-specific queries
        if sector == "finance":
            queries.extend([
                "IFRS financial data standards columns",
                "Basel III risk data requirements",
                "financial dataset column naming conventions"
            ])
        elif sector == "sante":
            queries.extend([
                "HL7 FHIR data standards",
                "ICD-10 medical coding structure",
                "healthcare data column standards"
            ])
        elif sector == "retail":
            queries.extend([
                "retail data standards SKU UPC",
                "product classification standards GS1",
                "retail analytics data schema"
            ])
        else:
            queries.append(f"{sector} industry data standards")
        
        # Search for each query
        for query in queries[:3]:  # Limit to avoid excessive API calls
            results = await self._web_search(query)
            references["sources"].extend(results.get("urls", []))
            
            # Extract standards from results
            for result in results.get("results", []):
                if "standard" in result.get("title", "").lower():
                    references["standards"].append({
                        "name": result["title"],
                        "url": result["url"],
                        "snippet": result["snippet"]
                    })
        
        # Try to map columns to standards
        for col in columns[:10]:  # Limit for performance
            col_query = f"{sector} {col} data standard definition"
            col_results = await self._web_search(col_query)
            
            if col_results.get("results"):
                references["column_mappings"][col] = {
                    "potential_standard": col_results["results"][0]["title"],
                    "definition": col_results["results"][0]["snippet"]
                }
        
        return references
    
    async def _web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search with caching"""
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists() and self.config.enable_caching:
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                    # Check if cache is still valid
                    if time.time() - cached.get("timestamp", 0) < self.config.web_search_config["cache_ttl"]:
                        return cached["data"]
            except Exception:
                pass
        
        # Perform actual search
        results = await self._perform_web_search(query)
        
        # Cache results
        if self.config.enable_caching:
            try:
                with open(cache_file, "w") as f:
                    json.dump({
                        "timestamp": time.time(),
                        "data": results
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")
        
        return results
    
    async def _perform_web_search(self, query: str) -> Dict[str, Any]:
        """Perform actual web search"""
        # This is a simplified implementation
        # In production, you would use a proper search API
        
        results = {
            "query": query,
            "results": [],
            "urls": []
        }
        
        # Simulate search results based on query
        if "IFRS" in query or "Basel" in query:
            results["results"].append({
                "title": "IFRS Standards for Financial Reporting",
                "url": "https://www.ifrs.org/standards/",
                "snippet": "International Financial Reporting Standards provide global standards for financial data..."
            })
            results["urls"].append("https://www.ifrs.org/standards/")
        
        if "HL7" in query or "FHIR" in query:
            results["results"].append({
                "title": "HL7 FHIR Data Standards",
                "url": "https://www.hl7.org/fhir/",
                "snippet": "Fast Healthcare Interoperability Resources standard for healthcare data exchange..."
            })
            results["urls"].append("https://www.hl7.org/fhir/")
        
        if "SKU" in query or "UPC" in query:
            results["results"].append({
                "title": "GS1 Product Identification Standards",
                "url": "https://www.gs1.org/standards/",
                "snippet": "Global standards for product identification including SKU and UPC codes..."
            })
            results["urls"].append("https://www.gs1.org/standards/")
        
        # Add generic result if no specific match
        if not results["results"]:
            results["results"].append({
                "title": f"Data Standards for {query}",
                "url": f"https://example.com/standards",
                "snippet": f"Industry standards and best practices for {query}"
            })
        
        return results
    
    async def _handle_run_with_functions(self, thread_id: str, run_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle assistant run with function calling for web search"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.timeout_seconds:
            run_status = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run_status.status == 'completed':
                # Get messages
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread_id
                )
                
                # Extract assistant's response
                for message in messages.data:
                    if message.role == 'assistant':
                        return {"content": message.content[0].text.value}
            
            elif run_status.status == 'requires_action':
                # Handle function calls
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    if tool_call.function.name == "web_search":
                        # Parse arguments
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query", "")
                        
                        # Perform web search
                        search_results = await self._web_search(query)
                        
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(search_results)
                        })
                
                # Submit tool outputs
                await self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
            
            elif run_status.status == 'failed':
                raise Exception(f"Run failed: {run_status.last_error}")
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Assistant run timed out")
    
    def _parse_validation_results(self, result: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, Any]:
        """Parse validation results from assistant"""
        try:
            content = result.get("content", "")
            
            # Try to extract JSON
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.findall(json_pattern, content)
            
            validation_report = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "suggestions": [],
                "column_validations": {},
                "sources": references.get("sources", []),
                "standards_found": references.get("standards", [])
            }
            
            if json_matches:
                for match in sorted(json_matches, key=len, reverse=True):
                    try:
                        parsed = json.loads(match)
                        validation_report.update(parsed)
                        break
                    except json.JSONDecodeError:
                        continue
            
            # Extract information from text if no JSON
            if "invalid" in content.lower() or "error" in content.lower():
                validation_report["valid"] = False
            
            # Parse text for issues
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()
                if "issue:" in line_lower or "problem:" in line_lower:
                    validation_report["issues"].append(line.strip())
                elif "warning:" in line_lower:
                    validation_report["warnings"].append(line.strip())
                elif "suggestion:" in line_lower or "recommend:" in line_lower:
                    validation_report["suggestions"].append(line.strip())
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Failed to parse validation results: {e}")
            return self._basic_validation_report(references)
    
    def _basic_validation(self, df: pd.DataFrame, sector: str) -> Dict[str, Any]:
        """Basic validation without OpenAI"""
        report = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "column_validations": {},
            "sources": []
        }
        
        # Basic sector-specific validations
        if sector == "finance":
            # Check for common financial columns
            expected_cols = ["amount", "date", "currency", "transaction_id"]
            missing = [col for col in expected_cols if col not in [c.lower() for c in df.columns]]
            if missing:
                report["warnings"].append(f"Missing expected financial columns: {missing}")
            
            # Check for negative amounts
            for col in df.select_dtypes(include=[np.number]).columns:
                if "amount" in col.lower() or "balance" in col.lower():
                    if (df[col] < 0).any():
                        report["warnings"].append(f"Column '{col}' contains negative values")
        
        elif sector == "sante":
            # Check for patient identifiers
            if not any("patient" in col.lower() or "id" in col.lower() for col in df.columns):
                report["warnings"].append("No patient identifier column found")
            
            # Check for date columns
            if not any("date" in col.lower() for col in df.columns):
                report["warnings"].append("No date column found for medical records")
        
        # General validations
        for col in df.columns:
            col_report = {"valid": True, "issues": []}
            
            # Check for special characters in column names
            if not col.replace('_', '').replace('-', '').isalnum():
                col_report["issues"].append("Contains special characters")
                report["suggestions"].append(f"Rename column '{col}' to use only alphanumeric characters")
            
            # Check for spaces in column names
            if ' ' in col:
                col_report["issues"].append("Contains spaces")
                report["suggestions"].append(f"Replace spaces in column '{col}' with underscores")
            
            report["column_validations"][col] = col_report
        
        return report
    
    def _basic_validation_report(self, references: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic validation report structure"""
        return {
            "valid": True,
            "issues": [],
            "warnings": ["Validation completed with limited analysis"],
            "suggestions": ["Consider manual review of sector standards"],
            "column_validations": {},
            "sources": references.get("sources", []),
            "standards_found": references.get("standards", [])
        }
    
    async def validate_column_names(self, columns: List[str], sector: str) -> Dict[str, str]:
        """Validate and suggest standard column names"""
        mappings = {}
        
        # Common sector mappings
        sector_mappings = {
            "finance": {
                "amt": "amount",
                "dt": "date",
                "curr": "currency",
                "tx_id": "transaction_id",
                "acct": "account",
                "bal": "balance"
            },
            "sante": {
                "pat_id": "patient_id",
                "diag": "diagnosis",
                "med": "medication",
                "dos": "date_of_service",
                "prov": "provider"
            },
            "retail": {
                "prod": "product",
                "qty": "quantity",
                "price": "unit_price",
                "cat": "category",
                "inv": "inventory"
            }
        }
        
        sector_map = sector_mappings.get(sector, {})
        
        for col in columns:
            col_lower = col.lower()
            
            # Check direct mapping
            if col_lower in sector_map:
                mappings[col] = sector_map[col_lower]
            
            # Check partial matches
            else:
                for abbrev, standard in sector_map.items():
                    if abbrev in col_lower or standard in col_lower:
                        mappings[col] = standard
                        break
        
        return mappings

"""
Validator Agent - Validates data against sector standards using web search
HYBRID ARCHITECTURE: Claude for reasoning + OpenAI for web search
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

# OpenAI for web search capabilities
_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    from openai import AsyncOpenAI
else:
    AsyncOpenAI = None

# Claude for intelligent reasoning
_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

if TYPE_CHECKING:
    from openai import AsyncOpenAI as _AsyncOpenAIType

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    Agent responsible for validating data against sector-specific standards
    HYBRID ARCHITECTURE:
    - Uses OpenAI Assistant with Web Search for finding standards
    - Uses Claude for intelligent validation reasoning and analysis
    """
    
    def __init__(self, config: AgentConfig, use_claude: bool = True):
        """
        Initialize Validator Agent with hybrid architecture
        
        Args:
            config: Agent configuration
            use_claude: Whether to use Claude for validation reasoning
        """
        self.config = config
        self.use_claude = use_claude and AsyncAnthropic is not None
        
        # Initialize OpenAI client for web search
        if AsyncOpenAI is not None and config.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            self.openai_client = None
            if AsyncOpenAI is None:
                logger.warning(
                    "AsyncOpenAI client unavailable - web search capabilities disabled"
                )
            else:
                logger.warning("OpenAI API key missing - web search capabilities disabled")
        
        # Initialize Claude client for reasoning
        if self.use_claude:
            self.claude_client = AsyncAnthropic()
            self.claude_model = "claude-sonnet-4-20250514"
            logger.info("💎 Claude SDK enabled for validation reasoning")
        else:
            self.claude_client = None
            if use_claude:
                logger.warning("⚠️ Claude SDK requested but not available")
            else:
                logger.info("📋 Using rule-based validation")
        
        # Assistant configuration
        self.assistant = None
        self.assistant_id = config.get_assistant_id(AgentType.VALIDATOR)

        # Cache for web search results
        self.search_cache = {}
        self.cache_dir = Path(config.cache_dir) / "validator"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.validation_metrics = {
            "openai_searches": 0,
            "claude_analyses": 0,
            "cache_hits": 0,
            "total_validations": 0
        }

    # FIXED: Lazy initialization - no automatic init
      self._init_lock = asyncio.Lock()
      self._initialized = False
    
    async def _initialize_assistant(self):
        """Create or retrieve OpenAI Assistant for web search"""
        try:
            if self.openai_client is None:
                logger.debug("ValidatorAgent _initialize_assistant skipped - OpenAI unavailable")
                return

            if self.assistant_id:
                self.assistant = await self.openai_client.beta.assistants.retrieve(
                    assistant_id=self.assistant_id
                )
                logger.info(f"Retrieved existing Validator assistant: {self.assistant_id}")
            else:
                self.assistant = await self.openai_client.beta.assistants.create(
                    name="Data Validator Agent - Web Search",
                    instructions=VALIDATOR_SYSTEM_PROMPT,
                    model=self.config.model,
                    tools=self.config.get_agent_tools(AgentType.VALIDATOR)
                )
                self.assistant_id = self.assistant.id
                self.config.save_assistant_id(AgentType.VALIDATOR, self.assistant_id)
                logger.info(f"Created new Validator assistant: {self.assistant_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize validator assistant: {e}")

    async def _ensure_assistant_initialized(self):
        """Thread-safe initialization with double-check locking"""
        if self.openai_client is None or self._initialized:
            return

        async with self._init_lock:
            if self._initialized:  # Double-check
                return
            
            await self._initialize_assistant()
            self._initialized = True

    async def validate(self, df: pd.DataFrame, profile_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against sector standards
        HYBRID APPROACH: OpenAI for search + Claude for reasoning
        
        Args:
            df: Input dataframe
            profile_report: Report from profiler agent
            
        Returns:
            Dictionary containing validation results
        """
        self.validation_metrics["total_validations"] += 1
        
        try:
            sector = self.config.user_context.get("secteur_activite", "general")
            
            # Phase 1: Use OpenAI to search for sector standards (if available)
            if self.openai_client is not None:
                await self._ensure_assistant_initialized()
                references = await self._search_sector_standards(sector, df.columns.tolist())
                logger.info(f"🔍 Found {len(references.get('standards', []))} standards via OpenAI search")
            else:
                logger.info("⚠️ OpenAI unavailable, using basic references")
                references = self._get_basic_references(sector)
            
            # Phase 2: Use Claude for intelligent validation analysis (if available)
            if self.use_claude:
                logger.info("💎 Using Claude for validation reasoning...")
                validation_report = await self._claude_validate(
                    df, profile_report, references, sector
                )
                self.validation_metrics["claude_analyses"] += 1
            else:
                # Fallback to OpenAI assistant or basic validation
                if self.openai_client is not None:
                    logger.info("📋 Using OpenAI assistant for validation")
                    validation_report = await self._openai_validate(
                        df, profile_report, references, sector
                    )
                else:
                    logger.info("📋 Using basic validation")
                    validation_report = self._basic_validation(df, sector)
            
            # Enrich report with references
            validation_report["sources"] = references.get("sources", [])
            validation_report["standards_found"] = references.get("standards", [])
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return self._basic_validation(df, sector)
    
    async def _claude_validate(
        self,
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        references: Dict[str, Any],
        sector: str
    ) -> Dict[str, Any]:
        """
        Use Claude for intelligent validation reasoning
        STRATEGIC VALIDATION DECISIONS
        """
        # Prepare data summary for Claude
        data_summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_counts': df.isnull().sum().to_dict(),
            'sample_values': {}
        }
        
        # Add sample values (first row) for context
        for col in df.columns[:10]:  # Limit to first 10 columns
            try:
                data_summary['sample_values'][col] = str(df[col].iloc[0])
            except:
                data_summary['sample_values'][col] = "N/A"
        
        prompt = f"""Analyze this dataset for compliance with sector standards and data quality.

Sector: {sector}
Target Variable: {self.config.user_context.get('target_variable', 'unknown')}

Dataset Summary:
{json.dumps(data_summary, indent=2)}

Profile Report (Quality Issues):
{json.dumps(profile_report.get('quality_issues', []), indent=2)}

Industry Standards Found:
{json.dumps(references.get('standards', [])[:3], indent=2)}  # Top 3 standards

Column Standard Mappings:
{json.dumps(references.get('column_mappings', {}), indent=2)}

Perform comprehensive validation covering:
1. **Sector Compliance**: Check against industry standards for {sector}
2. **Data Quality**: Assess completeness, consistency, validity
3. **Column Naming**: Evaluate if column names follow conventions
4. **Data Types**: Verify appropriate types for sector
5. **Business Logic**: Identify violations (e.g., negative amounts in finance)

Respond ONLY with valid JSON:
{{
  "valid": true/false,
  "overall_score": 0-100,
  "issues": [
    {{"severity": "critical|high|medium|low", "column": "col_name", "issue": "description", "impact": "what this means"}}
  ],
  "warnings": [
    {{"column": "col_name", "warning": "description", "recommendation": "what to do"}}
  ],
  "suggestions": [
    {{"type": "naming|quality|compliance", "suggestion": "description", "priority": "high|medium|low"}}
  ],
  "column_validations": {{
    "col_name": {{"valid": true/false, "issues": ["issue1"], "standard_compliance": "compliant|non-compliant|unknown"}}
  }},
  "sector_compliance": {{
    "compliant": true/false,
    "missing_standards": ["standard1"],
    "violations": ["violation1"]
  }},
  "reasoning": "2-3 sentence summary of validation findings"
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=3000,
                system="You are an expert data validator specializing in sector-specific compliance and data quality. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON response
            validation_report = json.loads(response_text)
            
            logger.info(f"💎 Claude validation: Overall score {validation_report.get('overall_score', 0)}/100")
            logger.info(f"   Issues: {len(validation_report.get('issues', []))}, "
                       f"Warnings: {len(validation_report.get('warnings', []))}")
            
            return validation_report
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Claude response was not valid JSON: {e}")
            return self._basic_validation(df, sector)
        except Exception as e:
            logger.warning(f"⚠️ Claude validation failed: {e}")
            return self._basic_validation(df, sector)
    
    async def _openai_validate(
        self,
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        references: Dict[str, Any],
        sector: str
    ) -> Dict[str, Any]:
        """
        Use OpenAI Assistant for validation (fallback from Claude)
        """
        try:
            # Create a thread
            thread = await self.openai_client.beta.threads.create()
            
            # Prepare validation request
            message_content = VALIDATOR_USER_PROMPT.format(
                sector=sector,
                columns=json.dumps(list(df.columns)),
                profile_summary=json.dumps(profile_report, indent=2),
                references=json.dumps(references, indent=2),
                target_variable=self.config.user_context.get("target_variable", "unknown")
            )
            
            await self.openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            
            # Run the assistant
            run = await self.openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Handle function calls for web search
            result = await self._handle_run_with_functions(thread.id, run.id, df)
            
            # Parse and return results
            return self._parse_validation_results(result, references)
            
        except Exception as e:
            logger.error(f"OpenAI validation failed: {e}")
            return self._basic_validation(df, sector)
    
    async def _search_sector_standards(self, sector: str, columns: List[str]) -> Dict[str, Any]:
        """
        Search for sector-specific standards using OpenAI web search
        """
        self.validation_metrics["openai_searches"] += 1
        
        references = {
            "sector": sector,
            "standards": [],
            "column_mappings": {},
            "sources": []
        }
        
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
    
    def _get_basic_references(self, sector: str) -> Dict[str, Any]:
        """
        Get basic sector references without web search
        """
        references = {
            "sector": sector,
            "standards": [],
            "column_mappings": {},
            "sources": []
        }
        
        # Hardcoded common standards
        if sector == "finance":
            references["standards"] = [
                {
                    "name": "IFRS Standards",
                    "url": "https://www.ifrs.org/standards/",
                    "snippet": "International Financial Reporting Standards"
                }
            ]
        elif sector == "sante":
            references["standards"] = [
                {
                    "name": "HL7 FHIR",
                    "url": "https://www.hl7.org/fhir/",
                    "snippet": "Fast Healthcare Interoperability Resources"
                }
            ]
        elif sector == "retail":
            references["standards"] = [
                {
                    "name": "GS1 Standards",
                    "url": "https://www.gs1.org/standards/",
                    "snippet": "Global product identification standards"
                }
            ]
        
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
                        self.validation_metrics["cache_hits"] += 1
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
        """
        Perform actual web search
        Simplified implementation - production would use proper search API
        """
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
            run_status = await self.openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run_status.status == 'completed':
                # Get messages
                messages = await self.openai_client.beta.threads.messages.list(
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
                await self.openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
            
            elif run_status.status == 'failed':
                raise Exception(f"Run failed: {run_status.last_error}")
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Assistant run timed out")
    
    def _parse_validation_results(self, result: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, Any]:
        """Parse validation results from OpenAI assistant"""
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
        """Basic validation without OpenAI or Claude"""
        report = {
            "valid": True,
            "overall_score": 70,  # Default moderate score
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "column_validations": {},
            "sources": [],
            "sector_compliance": {
                "compliant": None,
                "missing_standards": [],
                "violations": []
            }
        }
        
        # Basic sector-specific validations
        if sector == "finance":
            # Check for common financial columns
            expected_cols = ["amount", "date", "currency", "transaction_id"]
            missing = [col for col in expected_cols if col not in [c.lower() for c in df.columns]]
            if missing:
                report["warnings"].append({
                    "column": "general",
                    "warning": f"Missing expected financial columns: {missing}",
                    "recommendation": "Consider adding standard financial columns"
                })
            
            # Check for negative amounts
            for col in df.select_dtypes(include=[np.number]).columns:
                if "amount" in col.lower() or "balance" in col.lower():
                    if (df[col] < 0).any():
                        report["warnings"].append({
                            "column": col,
                            "warning": "Contains negative values",
                            "recommendation": "Verify if negative amounts are expected"
                        })
        
        elif sector == "sante":
            # Check for patient identifiers
            if not any("patient" in col.lower() or "id" in col.lower() for col in df.columns):
                report["warnings"].append({
                    "column": "general",
                    "warning": "No patient identifier column found",
                    "recommendation": "Add patient identifier for compliance"
                })
            
            # Check for date columns
            if not any("date" in col.lower() for col in df.columns):
                report["warnings"].append({
                    "column": "general",
                    "warning": "No date column found for medical records",
                    "recommendation": "Add date/timestamp columns"
                })
        
        # General validations
        for col in df.columns:
            col_report = {
                "valid": True,
                "issues": [],
                "standard_compliance": "unknown"
            }
            
            # Check for special characters in column names
            if not col.replace('_', '').replace('-', '').isalnum():
                col_report["issues"].append("Contains special characters")
                report["suggestions"].append({
                    "type": "naming",
                    "suggestion": f"Rename column '{col}' to use only alphanumeric characters",
                    "priority": "medium"
                })
                col_report["valid"] = False
            
            # Check for spaces in column names
            if ' ' in col:
                col_report["issues"].append("Contains spaces")
                report["suggestions"].append({
                    "type": "naming",
                    "suggestion": f"Replace spaces in column '{col}' with underscores",
                    "priority": "high"
                })
                col_report["valid"] = False
            
            report["column_validations"][col] = col_report
        
        # Calculate overall score
        total_issues = len(report["issues"]) + len(report["warnings"]) * 0.5
        score_penalty = min(50, total_issues * 5)  # Max 50 point penalty
        report["overall_score"] = max(20, 70 - score_penalty)
        
        report["reasoning"] = f"Basic validation completed with {len(report['issues'])} issues and {len(report['warnings'])} warnings. Overall score: {report['overall_score']}/100."
        
        return report
    
    def _basic_validation_report(self, references: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic validation report structure"""
        return {
            "valid": True,
            "overall_score": 60,
            "issues": [],
            "warnings": [{"column": "general", "warning": "Validation completed with limited analysis", "recommendation": "Consider manual review"}],
            "suggestions": [{"type": "compliance", "suggestion": "Review sector standards manually", "priority": "medium"}],
            "column_validations": {},
            "sources": references.get("sources", []),
            "standards_found": references.get("standards", []),
            "reasoning": "Limited validation performed due to tool availability."
        }
    
    async def validate_column_names(self, columns: List[str], sector: str) -> Dict[str, str]:
        """
        Validate and suggest standard column names
        Can optionally use Claude for intelligent suggestions
        """
        # Basic mappings
        mappings = self._basic_column_mappings(columns, sector)
        
        # Enhance with Claude if available
        if self.use_claude and len(columns) > 0:
            try:
                enhanced_mappings = await self._claude_suggest_column_names(columns, sector)
                mappings.update(enhanced_mappings)
            except Exception as e:
                logger.warning(f"Claude column name suggestions failed: {e}")
        
        return mappings
    
    def _basic_column_mappings(self, columns: List[str], sector: str) -> Dict[str, str]:
        """Basic rule-based column mappings"""
        mappings = {}
        
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
            
            if col_lower in sector_map:
                mappings[col] = sector_map[col_lower]
            else:
                for abbrev, standard in sector_map.items():
                    if abbrev in col_lower or standard in col_lower:
                        mappings[col] = standard
                        break
        
        return mappings
    
    async def _claude_suggest_column_names(self, columns: List[str], sector: str) -> Dict[str, str]:
        """Use Claude to suggest better column names"""
        prompt = f"""Suggest standardized column names for this {sector} dataset.

Current columns: {json.dumps(columns)}

Provide better names following {sector} industry standards.
Consider:
- Industry standard naming conventions
- Clarity and descriptiveness
- Consistent formatting (snake_case)

Respond ONLY with JSON mapping old names to suggested names:
{{"old_name": "suggested_standard_name"}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                system=f"You are an expert in {sector} data standards.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            suggestions = json.loads(response_text)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Claude column name suggestions failed: {e}")
            return {}
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics for monitoring"""
        return self.validation_metrics.copy()

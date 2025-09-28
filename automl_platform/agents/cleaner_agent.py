"""
Cleaner Agent - Applies intelligent data cleaning transformations
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
import importlib.util
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING
import time
import io
import base64

from .agent_config import AgentConfig, AgentType
from .prompts.cleaner_prompts import CLEANER_SYSTEM_PROMPT, CLEANER_USER_PROMPT

_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    from openai import AsyncOpenAI  # type: ignore
else:
    AsyncOpenAI = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI as _AsyncOpenAIType

logger = logging.getLogger(__name__)


class CleanerAgent:
    """
    Agent responsible for applying intelligent cleaning transformations
    Uses OpenAI Assistant with Code Interpreter and File Access
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize Cleaner Agent"""
        self.config = config
        if AsyncOpenAI is not None and config.openai_api_key:
            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            self.client = None
            if AsyncOpenAI is None:
                logger.warning(
                    "AsyncOpenAI client unavailable because the 'openai' package is not installed. "
                    "CleanerAgent will fall back to local cleaning strategies."
                )
            else:
                logger.warning("OpenAI API key missing; CleanerAgent will rely on local cleaning strategies only.")
        self.assistant = None
        self.assistant_id = config.get_assistant_id(AgentType.CLEANER)
        
        # Track transformations
        self.transformations_history = []

        # Assistant initialization tracking
        self._initialization_task: Optional[asyncio.Task] = None
        self._initialization_lock: Optional[asyncio.Lock] = None

        if self.client is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                self._initialization_task = loop.create_task(self._initialize_assistant())
    
    async def _initialize_assistant(self):
        """Create or retrieve OpenAI Assistant"""
        try:
            if self.client is None:
                logger.debug("CleanerAgent _initialize_assistant skipped because client is unavailable.")
                return

            if self.assistant_id:
                self.assistant = await self.client.beta.assistants.retrieve(
                    assistant_id=self.assistant_id
                )
                logger.info(f"Retrieved existing Cleaner assistant: {self.assistant_id}")
            else:
                self.assistant = await self.client.beta.assistants.create(
                    name="Data Cleaner Agent",
                    instructions=CLEANER_SYSTEM_PROMPT,
                    model=self.config.model,
                    tools=self.config.get_agent_tools(AgentType.CLEANER)
                )
                self.assistant_id = self.assistant.id
                self.config.save_assistant_id(AgentType.CLEANER, self.assistant_id)
                logger.info(f"Created new Cleaner assistant: {self.assistant_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize cleaner assistant: {e}")

    async def _ensure_assistant_initialized(self):
        if self.client is None or self.assistant:
            return

        if self._initialization_lock is None:
            self._initialization_lock = asyncio.Lock()

        async with self._initialization_lock:
            if self.assistant:
                return

            if self._initialization_task is not None:
                await self._initialization_task
                self._initialization_task = None
            else:
                await self._initialize_assistant()
    
    async def clean(
        self, 
        df: pd.DataFrame, 
        profile_report: Dict[str, Any],
        validation_report: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Apply intelligent cleaning transformations
        
        Args:
            df: Input dataframe
            profile_report: Report from profiler agent
            validation_report: Report from validator agent
            
        Returns:
            Tuple of (cleaned_dataframe, list_of_transformations)
        """
        try:
            if self.client is None:
                logger.info("CleanerAgent using basic cleaning because OpenAI client is unavailable.")
                return self._basic_cleaning(df, profile_report)

            # Ensure assistant is initialized
            await self._ensure_assistant_initialized()
            
            # Create a thread
            thread = await self.client.beta.threads.create()
            
            # Prepare cleaning request
            cleaning_context = self._prepare_cleaning_context(df, profile_report, validation_report)
            
            message_content = CLEANER_USER_PROMPT.format(
                data_summary=json.dumps(cleaning_context["data_summary"], indent=2),
                quality_issues=json.dumps(cleaning_context["quality_issues"], indent=2),
                validation_issues=json.dumps(cleaning_context["validation_issues"], indent=2),
                sector=self.config.user_context.get("secteur_activite", "general"),
                target_variable=self.config.user_context.get("target_variable", "unknown")
            )
            
            # Upload data sample as file if needed
            if len(df) > 100:
                file_id = await self._upload_data_sample(df.head(100))
                message_content += f"\n\nData sample file ID: {file_id}"
            
            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            
            # Run the assistant
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion
            result = await self._wait_for_run_completion(thread.id, run.id)
            
            # Parse transformations
            transformations = self._parse_transformations(result)
            
            # Apply transformations
            cleaned_df = await self._apply_transformations(df, transformations)
            
            # Record transformations
            self.transformations_history.extend(transformations)
            
            return cleaned_df, transformations
            
        except Exception as e:
            logger.error(f"Error in cleaning: {e}")
            # Fallback to basic cleaning
            return self._basic_cleaning(df, profile_report)
    
    def _prepare_cleaning_context(
        self, 
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        validation_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for cleaning"""
        context = {
            "data_summary": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            "quality_issues": profile_report.get("quality_issues", []),
            "validation_issues": validation_report.get("issues", [])
        }
        
        # Add specific issues per column
        column_issues = {}
        
        for col in df.columns:
            issues = []
            
            # Missing values
            missing_ratio = df[col].isnull().mean()
            if missing_ratio > 0:
                issues.append({
                    "type": "missing_values",
                    "severity": "high" if missing_ratio > 0.5 else "medium" if missing_ratio > 0.2 else "low",
                    "ratio": float(missing_ratio)
                })
            
            # Outliers for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                if outliers > 0:
                    issues.append({
                        "type": "outliers",
                        "count": int(outliers),
                        "ratio": float(outliers / len(df))
                    })
            
            # High cardinality for categorical
            if df[col].dtype == 'object':
                cardinality = df[col].nunique()
                if cardinality > 100:
                    issues.append({
                        "type": "high_cardinality",
                        "unique_values": int(cardinality)
                    })
                
                # Check for inconsistent values
                if col in validation_report.get("column_validations", {}):
                    col_validation = validation_report["column_validations"][col]
                    if not col_validation.get("valid", True):
                        issues.extend(col_validation.get("issues", []))
            
            if issues:
                column_issues[col] = issues
        
        context["column_issues"] = column_issues
        
        return context
    
    async def _upload_data_sample(self, df_sample: pd.DataFrame) -> str:
        """Upload data sample for assistant to analyze"""
        try:
            # Convert to CSV
            csv_buffer = io.StringIO()
            df_sample.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Create file
            file = await self.client.files.create(
                file=("data_sample.csv", csv_content.encode()),
                purpose="assistants"
            )
            
            return file.id
            
        except Exception as e:
            logger.error(f"Failed to upload data sample: {e}")
            return ""
    
    async def _wait_for_run_completion(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        """Wait for assistant run to complete"""
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
                
            elif run_status.status == 'failed':
                raise Exception(f"Run failed: {run_status.last_error}")
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Assistant run timed out")
    
    def _parse_transformations(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse transformation instructions from assistant"""
        transformations = []
        
        try:
            content = result.get("content", "")
            
            # Try to extract JSON
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.findall(json_pattern, content)
            
            if json_matches:
                for match in sorted(json_matches, key=len, reverse=True):
                    try:
                        parsed = json.loads(match)
                        if "transformations" in parsed:
                            transformations = parsed["transformations"]
                            break
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON, parse text for transformation instructions
            if not transformations:
                transformations = self._extract_transformations_from_text(content)
            
        except Exception as e:
            logger.error(f"Failed to parse transformations: {e}")
        
        # Validate transformations
        valid_transformations = []
        for t in transformations:
            if self._validate_transformation(t):
                valid_transformations.append(t)
        
        return valid_transformations
    
    def _extract_transformations_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Extract transformation instructions from text"""
        transformations = []
        
        # Common transformation patterns
        lines = content.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Fill missing values
            if "fill" in line_lower and "missing" in line_lower:
                # Extract column name and method
                import re
                col_match = re.search(r"column[s]?\s+['\"]?(\w+)['\"]?", line, re.IGNORECASE)
                if col_match:
                    transformations.append({
                        "column": col_match.group(1),
                        "action": "fill_missing",
                        "params": {"method": "median"}  # Default
                    })
            
            # Remove outliers
            elif "remove" in line_lower and "outlier" in line_lower:
                col_match = re.search(r"column[s]?\s+['\"]?(\w+)['\"]?", line, re.IGNORECASE)
                if col_match:
                    transformations.append({
                        "column": col_match.group(1),
                        "action": "handle_outliers",
                        "params": {"method": "clip"}
                    })
            
            # Normalize
            elif "normalize" in line_lower or "scale" in line_lower:
                col_match = re.search(r"column[s]?\s+['\"]?(\w+)['\"]?", line, re.IGNORECASE)
                if col_match:
                    transformations.append({
                        "column": col_match.group(1),
                        "action": "normalize",
                        "params": {"method": "robust"}
                    })
        
        return transformations
    
    def _validate_transformation(self, transformation: Dict[str, Any]) -> bool:
        """Validate a transformation specification"""
        required_fields = ["column", "action"]
        
        for field in required_fields:
            if field not in transformation:
                return False
        
        # Check action is valid
        valid_actions = [
            "fill_missing", "handle_outliers", "normalize", "encode",
            "remove_column", "rename_column", "convert_dtype",
            "normalize_currency", "standardize_format", "clip_values"
        ]
        
        if transformation["action"] not in valid_actions:
            logger.warning(f"Unknown transformation action: {transformation['action']}")
            return False
        
        return True
    
    async def _apply_transformations(
        self, 
        df: pd.DataFrame, 
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply transformations to dataframe"""
        cleaned_df = df.copy()
        
        for trans in transformations:
            try:
                column = trans["column"]
                action = trans["action"]
                params = trans.get("params", {})
                
                if column not in cleaned_df.columns and action != "remove_column":
                    logger.warning(f"Column {column} not found, skipping transformation")
                    continue
                
                # Apply transformation based on action
                if action == "fill_missing":
                    method = params.get("method", "median")
                    if method == "median" and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                    elif method == "mean" and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                    elif method == "mode":
                        mode_val = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else "missing"
                        cleaned_df[column].fillna(mode_val, inplace=True)
                    elif method == "forward_fill":
                        cleaned_df[column].fillna(method='ffill', inplace=True)
                    elif method == "backward_fill":
                        cleaned_df[column].fillna(method='bfill', inplace=True)
                    else:
                        cleaned_df[column].fillna(params.get("value", "missing"), inplace=True)
                
                elif action == "handle_outliers":
                    method = params.get("method", "clip")
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        if method == "clip":
                            lower = cleaned_df[column].quantile(params.get("lower_percentile", 0.01))
                            upper = cleaned_df[column].quantile(params.get("upper_percentile", 0.99))
                            cleaned_df[column] = cleaned_df[column].clip(lower, upper)
                        elif method == "remove":
                            Q1 = cleaned_df[column].quantile(0.25)
                            Q3 = cleaned_df[column].quantile(0.75)
                            IQR = Q3 - Q1
                            mask = (cleaned_df[column] >= Q1 - 1.5 * IQR) & (cleaned_df[column] <= Q3 + 1.5 * IQR)
                            cleaned_df = cleaned_df[mask]
                
                elif action == "normalize":
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        method = params.get("method", "minmax")
                        if method == "minmax":
                            min_val = cleaned_df[column].min()
                            max_val = cleaned_df[column].max()
                            if max_val > min_val:
                                cleaned_df[column] = (cleaned_df[column] - min_val) / (max_val - min_val)
                        elif method == "zscore":
                            mean = cleaned_df[column].mean()
                            std = cleaned_df[column].std()
                            if std > 0:
                                cleaned_df[column] = (cleaned_df[column] - mean) / std
                        elif method == "robust":
                            median = cleaned_df[column].median()
                            q75 = cleaned_df[column].quantile(0.75)
                            q25 = cleaned_df[column].quantile(0.25)
                            iqr = q75 - q25
                            if iqr > 0:
                                cleaned_df[column] = (cleaned_df[column] - median) / iqr
                
                elif action == "encode":
                    if cleaned_df[column].dtype == 'object':
                        method = params.get("method", "label")
                        if method == "label":
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            cleaned_df[column] = le.fit_transform(cleaned_df[column].astype(str))
                        elif method == "onehot":
                            # One-hot encoding
                            dummies = pd.get_dummies(cleaned_df[column], prefix=column)
                            cleaned_df = pd.concat([cleaned_df.drop(columns=[column]), dummies], axis=1)
                
                elif action == "remove_column":
                    if column in cleaned_df.columns:
                        cleaned_df = cleaned_df.drop(columns=[column])
                
                elif action == "rename_column":
                    new_name = params.get("new_name")
                    if new_name and column in cleaned_df.columns:
                        cleaned_df = cleaned_df.rename(columns={column: new_name})
                
                elif action == "convert_dtype":
                    target_dtype = params.get("dtype", "float")
                    if column in cleaned_df.columns:
                        if target_dtype == "float":
                            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                        elif target_dtype == "int":
                            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce').fillna(0).astype(int)
                        elif target_dtype == "str":
                            cleaned_df[column] = cleaned_df[column].astype(str)
                        elif target_dtype == "datetime":
                            cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                
                elif action == "normalize_currency":
                    target_currency = params.get("target_currency", "EUR")
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        # Simplified currency normalization
                        conversion_rate = params.get("conversion_rate", 1.0)
                        cleaned_df[column] = cleaned_df[column] * conversion_rate
                
                elif action == "standardize_format":
                    format_type = params.get("format")
                    if format_type and column in cleaned_df.columns:
                        if "date" in format_type.lower():
                            cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                            if params.get("format"):
                                cleaned_df[column] = cleaned_df[column].dt.strftime(params["format"])
                
                elif action == "clip_values":
                    min_val = params.get("min")
                    max_val = params.get("max")
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        if min_val is not None:
                            cleaned_df[column] = cleaned_df[column].clip(lower=min_val)
                        if max_val is not None:
                            cleaned_df[column] = cleaned_df[column].clip(upper=max_val)
                
                logger.info(f"Applied transformation: {action} on column {column}")
                
            except Exception as e:
                logger.error(f"Failed to apply transformation {trans}: {e}")
                continue
        
        return cleaned_df
    
    def _basic_cleaning(self, df: pd.DataFrame, profile_report: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict]]:
        """Basic cleaning fallback"""
        logger.info("Using basic cleaning fallback")
        
        cleaned_df = df.copy()
        transformations = []
        
        # Remove duplicates
        if cleaned_df.duplicated().sum() > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            transformations.append({
                "action": "remove_duplicates",
                "params": {"rows_removed": df.duplicated().sum()}
            })
        
        # Handle missing values
        for col in cleaned_df.columns:
            missing_ratio = cleaned_df[col].isnull().mean()
            
            if missing_ratio > 0.5:
                # Drop column if more than 50% missing
                cleaned_df = cleaned_df.drop(columns=[col])
                transformations.append({
                    "column": col,
                    "action": "remove_column",
                    "params": {"reason": "high_missing_ratio", "ratio": missing_ratio}
                })
            elif missing_ratio > 0:
                # Fill missing values
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                    transformations.append({
                        "column": col,
                        "action": "fill_missing",
                        "params": {"method": "median"}
                    })
                else:
                    cleaned_df[col].fillna("missing", inplace=True)
                    transformations.append({
                        "column": col,
                        "action": "fill_missing",
                        "params": {"method": "constant", "value": "missing"}
                    })
        
        # Handle outliers in numeric columns
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((cleaned_df[col] < Q1 - 1.5 * IQR) | (cleaned_df[col] > Q3 + 1.5 * IQR)).sum()
            
            if outliers > len(cleaned_df) * 0.05:  # More than 5% outliers
                # Clip outliers
                lower = cleaned_df[col].quantile(0.01)
                upper = cleaned_df[col].quantile(0.99)
                cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                
                transformations.append({
                    "column": col,
                    "action": "handle_outliers",
                    "params": {"method": "clip", "outliers_found": int(outliers)}
                })
        
        return cleaned_df, transformations
    
    async def suggest_transformations(
        self, 
        df: pd.DataFrame, 
        profile_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get transformation suggestions without applying them"""
        suggestions = []
        
        # Analyze each column
        for col in df.columns:
            missing_ratio = df[col].isnull().mean()
            
            # Suggest handling missing values
            if missing_ratio > 0:
                if missing_ratio > 0.5:
                    suggestions.append({
                        "column": col,
                        "action": "remove_column",
                        "reason": f"High missing ratio ({missing_ratio:.1%})",
                        "priority": "high"
                    })
                else:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        suggestions.append({
                            "column": col,
                            "action": "fill_missing",
                            "params": {"method": "median"},
                            "reason": f"Missing values ({missing_ratio:.1%})",
                            "priority": "medium"
                        })
                    else:
                        suggestions.append({
                            "column": col,
                            "action": "fill_missing",
                            "params": {"method": "mode"},
                            "reason": f"Missing values ({missing_ratio:.1%})",
                            "priority": "medium"
                        })
            
            # Suggest handling outliers
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                
                if outliers > len(df) * 0.05:
                    suggestions.append({
                        "column": col,
                        "action": "handle_outliers",
                        "params": {"method": "clip"},
                        "reason": f"Outliers detected ({outliers} values)",
                        "priority": "low"
                    })
            
            # Suggest encoding for high cardinality
            if df[col].dtype == 'object':
                cardinality = df[col].nunique()
                if cardinality > 100:
                    suggestions.append({
                        "column": col,
                        "action": "encode",
                        "params": {"method": "target_encoding"},
                        "reason": f"High cardinality ({cardinality} unique values)",
                        "priority": "medium"
                    })
        
        return suggestions

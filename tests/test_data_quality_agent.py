"""
Tests for data quality agent module
====================================
Tests for data quality assessment and cleaning agents.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.data_quality_agent import (
    DataQualityAssessment,
    AkkioStyleCleaningAgent,
    DataRobotStyleQualityMonitor,
    IntelligentDataQualityAgent,
    RiskLevel,
)


class TestRiskLevel:
    """Unit tests for the shared RiskLevel enum utilities."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("high", RiskLevel.HIGH),
            ("High", RiskLevel.HIGH),
            ("HIGH", RiskLevel.HIGH),
            (True, RiskLevel.HIGH),
            (False, RiskLevel.NONE),
            ("  medium  ", RiskLevel.MEDIUM),
            ("super_high", RiskLevel.NONE),
        ],
    )
    def test_from_string_handles_case_and_booleans(self, value, expected):
        assert RiskLevel.from_string(value) is expected

    def test_from_string_unknown_defaults_to_none_level(self):
        assert RiskLevel.from_string("unknown") is RiskLevel.NONE

    def test_from_string_allows_custom_default(self):
        assert RiskLevel.from_string("unknown", default=RiskLevel.MEDIUM) is RiskLevel.MEDIUM

    def test_from_string_handles_none_and_invalid_types(self):
        assert RiskLevel.from_string(None) is RiskLevel.NONE
        assert RiskLevel.from_string(123, default=RiskLevel.HIGH) is RiskLevel.HIGH
        assert RiskLevel.from_string(["high"]) is RiskLevel.NONE
        assert RiskLevel.from_string({"risk": "high"}, default=RiskLevel.LOW) is RiskLevel.LOW


class TestDataQualityAssessment:
    """Tests for DataQualityAssessment dataclass"""
    
    def test_data_quality_assessment_creation(self):
        """Test creating DataQualityAssessment object"""
        assessment = DataQualityAssessment(
            quality_score=85.5,
            alerts=[{'type': 'missing', 'message': 'High missing values'}],
            warnings=[{'type': 'outlier', 'message': 'Outliers detected'}],
            recommendations=[{'priority': 'high', 'action': 'Impute missing'}],
            statistics={'rows': 1000, 'columns': 20},
            drift_risk=RiskLevel.MEDIUM,
            target_leakage_risk=RiskLevel.MEDIUM,
            visualization_data={'missing_heatmap': {}}
        )
        
        assert assessment.quality_score == 85.5
        assert len(assessment.alerts) == 1
        assert assessment.alerts[0]['type'] == 'missing'
        assert len(assessment.warnings) == 1
        assert assessment.warnings[0]['type'] == 'outlier'
        assert len(assessment.recommendations) == 1
        assert assessment.statistics['rows'] == 1000
        assert assessment.drift_risk == RiskLevel.MEDIUM
        assert assessment.target_leakage_risk == RiskLevel.MEDIUM
        assert 'missing_heatmap' in assessment.visualization_data
    
    def test_data_quality_assessment_defaults(self):
        """Test DataQualityAssessment with minimal parameters"""
        assessment = DataQualityAssessment(
            quality_score=90.0,
            alerts=[],
            warnings=[],
            recommendations=[],
            statistics={},
            drift_risk=RiskLevel.LOW,
            target_leakage_risk=RiskLevel.LOW,
            visualization_data={}
        )
        
        assert assessment.quality_score == 90.0
        assert assessment.alerts == []
        assert assessment.warnings == []
        assert assessment.drift_risk == RiskLevel.LOW
        assert assessment.target_leakage_risk == RiskLevel.LOW

    def test_data_quality_assessment_rejects_invalid_leakage_risk(self):
        """Ensure target_leakage_risk enforces the allowed literal values."""
        with pytest.raises(ValueError):
            DataQualityAssessment(
                quality_score=75.0,
                alerts=[],
                warnings=[],
                recommendations=[],
                statistics={},
                drift_risk=RiskLevel.LOW,
                target_leakage_risk='extreme',
                visualization_data={}
            )


    def test_data_quality_assessment_normalizes_legacy_boolean_risks(self):
        """Booleans from historical payloads are normalized into the enum values."""
        assessment = DataQualityAssessment(
            quality_score=88.0,
            alerts=[],
            warnings=[],
            recommendations=[],
            statistics={},
            drift_risk=True,
            target_leakage_risk=False,
            visualization_data={},
        )

        assert assessment.drift_risk == RiskLevel.HIGH
        assert assessment.target_leakage_risk == RiskLevel.NONE

    def test_data_quality_assessment_normalizes_string_risks(self):
        """String inputs are still accepted and normalized into RiskLevel enums."""
        assessment = DataQualityAssessment(
            quality_score=82.0,
            alerts=[],
            warnings=[],
            recommendations=[],
            statistics={},
            drift_risk="medium",
            target_leakage_risk="high",
            visualization_data={},
        )

        assert assessment.drift_risk is RiskLevel.MEDIUM
        assert assessment.target_leakage_risk is RiskLevel.HIGH

    def test_data_quality_assessment_dict_serializes_enum_values(self):
        assessment = DataQualityAssessment(
            quality_score=91.0,
            alerts=[],
            warnings=[],
            recommendations=[],
            statistics={},
            drift_risk=RiskLevel.HIGH,
            target_leakage_risk=RiskLevel.LOW,
            visualization_data={},
        )

        payload = assessment.to_dict()
        assert payload["drift_risk"] == RiskLevel.HIGH.value
        assert payload["target_leakage_risk"] == RiskLevel.LOW.value
        # Ensure the payload is JSON serializable without custom encoders
        assert json.loads(json.dumps(payload))["drift_risk"] == "high"

    def test_data_quality_assessment_roundtrip_serialization(self):
        assessment = DataQualityAssessment(
            quality_score=73.0,
            alerts=[{"type": "missing", "message": "Check column"}],
            warnings=[],
            recommendations=[],
            statistics={},
            drift_risk=RiskLevel.LOW,
            target_leakage_risk="medium",
            visualization_data={},
        )

        payload = assessment.to_dict()
        cloned = DataQualityAssessment(**payload)

        assert cloned.drift_risk is RiskLevel.LOW
        assert cloned.target_leakage_risk is RiskLevel.MEDIUM
        assert cloned.alerts == assessment.alerts

    def test_data_quality_assessment_from_legacy_fixture(self):
        payload = json.loads(
            Path(__file__).parent.joinpath("fixtures", "legacy_data_quality_assessment.json").read_text()
        )

        assessment = DataQualityAssessment(**payload)

        assert assessment.drift_risk is RiskLevel.HIGH
        assert assessment.target_leakage_risk is RiskLevel.HIGH
        assert assessment.quality_score == pytest.approx(72.5)
        assert assessment.to_dict()["target_leakage_risk"] == "high"


class TestDataRobotStyleQualityMonitor:
    """Tests for DataRobot-style quality monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create quality monitor instance"""
        return DataRobotStyleQualityMonitor()
    
    @pytest.fixture
    def clean_data(self):
        """Create clean dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100) * 2 + 5,
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def problematic_data(self):
        """Create dataset with various quality issues"""
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric_clean': np.random.randn(100),
            'numeric_missing': np.concatenate([np.random.randn(30), [np.nan] * 70]),  # 70% missing
            'numeric_outliers': np.concatenate([np.random.randn(90), [100, -100, 200, -200, 300, 400, 500, -500, 600, -600]]),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'high_cardinality': [f'ID_{i}' for i in range(100)],  # Unique values
            'constant': [1] * 100,
            'target': np.concatenate([np.zeros(95, dtype=int), np.ones(5, dtype=int)])  # Severe imbalance
        })

        # Add duplicates
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)

        return df
    
    def test_initialization(self, monitor):
        """Test monitor initialization with thresholds"""
        assert monitor.quality_thresholds['missing_critical'] == 0.5
        assert monitor.quality_thresholds['missing_warning'] == 0.2
        assert monitor.quality_thresholds['outlier_critical'] == 0.15
        assert monitor.quality_thresholds['cardinality_high'] == 0.9
        assert monitor.quality_thresholds['correlation_high'] == 0.95
        assert monitor.quality_thresholds['imbalance_severe'] == 20
    
    def test_assess_quality_clean_data(self, monitor, clean_data):
        """Test quality assessment on clean data"""
        assessment = monitor.assess_quality(clean_data, target_column='target')

        assert isinstance(assessment, DataQualityAssessment)
        assert assessment.quality_score > 70  # Clean data should have good score
        assert len(assessment.alerts) == 0 or len(assessment.alerts) <= 1  # No or minimal alerts
        assert assessment.drift_risk in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
        assert assessment.target_leakage_risk in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_assess_quality_without_target(self, monitor, clean_data):
        """When no target column is provided, leakage risk should be marked as none."""
        features_only = clean_data.drop(columns=['target'])

        assessment = monitor.assess_quality(features_only)

        assert assessment.target_leakage_risk == RiskLevel.NONE
    
    def test_assess_quality_problematic_data(self, monitor, problematic_data):
        """Test quality assessment on problematic data"""
        assessment = monitor.assess_quality(problematic_data, target_column='target')
        
        assert isinstance(assessment, DataQualityAssessment)
        assert assessment.quality_score < 70  # Should have lower score due to issues
        assert len(assessment.alerts) > 0  # Should have critical alerts
        assert len(assessment.warnings) > 0  # Should have warnings
        assert len(assessment.recommendations) > 0
        
        # Check specific issues are detected
        alert_types = [alert['type'] for alert in assessment.alerts]
        assert 'missing_critical' in alert_types  # 60% missing should trigger
        assert 'class_imbalance' in alert_types  # 95:5 ratio should trigger
    
    def test_assess_missing_values(self, monitor, problematic_data):
        """Test missing values assessment"""
        report = monitor._assess_missing_values(problematic_data)
        
        assert 'alerts' in report
        assert 'warnings' in report
        assert 'penalty' in report
        assert 'column_missing_pct' in report
        
        # Check numeric_missing column with 60% missing
        assert 'numeric_missing' in report['column_missing_pct']
        assert report['column_missing_pct']['numeric_missing'] == pytest.approx(63.636, rel=1e-3)
        
        # Should have critical alert for high missing
        assert any('numeric_missing' in str(alert) for alert in report['alerts'])
        assert report['penalty'] > 10  # High penalty for critical missing
    
    def test_assess_outliers(self, monitor, problematic_data):
        """Test outlier detection using IQR method"""
        report = monitor._assess_outliers(problematic_data)
        
        assert 'warnings' in report
        assert 'penalty' in report
        assert 'outlier_columns' in report
        
        # Should detect outliers in numeric_outliers column
        assert 'numeric_outliers' in report['outlier_columns']
        
        # Check outlier percentage is reasonable
        outlier_pct = report['outlier_columns']['numeric_outliers']
        assert outlier_pct > 5  # Should detect some outliers
    
    def test_assess_data_types(self, monitor, problematic_data):
        """Test data type and cardinality assessment"""
        report = monitor._assess_data_types(problematic_data)
        
        assert 'warnings' in report
        assert 'penalty' in report
        assert 'type_counts' in report
        
        # Should detect high cardinality in ID column
        warning_messages = [w['message'] for w in report['warnings']]
        assert any('high_cardinality' in msg for msg in warning_messages)
        
        # Check type counts
        assert report['type_counts']['numeric'] >= 3
        assert report['type_counts']['categorical'] >= 1
    
    def test_assess_duplicates(self, monitor, problematic_data):
        """Test duplicate detection"""
        report = monitor._assess_duplicates(problematic_data)
        
        assert 'count' in report
        assert 'percentage' in report
        assert 'penalty' in report
        
        # We added 10 duplicates to 100 original rows
        assert report['count'] == 10
        assert report['percentage'] > 9  # Should be around 9.09%
        assert report['penalty'] > 0
    
    def test_assess_target_imbalance(self, monitor, problematic_data):
        """Test target variable imbalance detection"""
        report = monitor._assess_target(problematic_data, 'target')
        
        assert 'alerts' in report
        assert 'penalty' in report
        
        # Should detect severe class imbalance (95:5 ratio)
        alert_types = [alert['type'] for alert in report['alerts']]
        assert 'class_imbalance' in alert_types
        
        # Check the alert message mentions the ratio
        imbalance_alerts = [a for a in report['alerts'] if a['type'] == 'class_imbalance']
        assert len(imbalance_alerts) > 0
        assert 'ratio' in imbalance_alerts[0]['message'].lower()
    
    def test_assess_target_missing(self, monitor):
        """Test detection of missing target values"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, np.nan, 1, np.nan]  # 2 missing values
        })
        
        report = monitor._assess_target(df, 'target')
        
        alert_types = [alert['type'] for alert in report['alerts']]
        assert 'target_missing' in alert_types
        
        # Check message mentions count
        missing_alerts = [a for a in report['alerts'] if a['type'] == 'target_missing']
        assert '2' in missing_alerts[0]['message']
    
    def test_detect_target_leakage_correlation(self, monitor):
        """Test target leakage detection via correlation"""
        # Create data with perfect correlation
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        df['leaky_feature'] = df['target'] * 2 + np.random.randn(100) * 0.01  # Almost perfect correlation
        
        leakage_risk = monitor._detect_target_leakage(df, 'target')
        assert leakage_risk is RiskLevel.HIGH
    
    def test_detect_target_leakage_naming(self, monitor):
        """Test target leakage detection via column naming"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target_encoded': np.random.randn(100),  # Suspicious name
            'target': np.random.randn(100)
        })
        
        leakage_risk = monitor._detect_target_leakage(df, 'target')
        assert leakage_risk is RiskLevel.MEDIUM
        
        # Test other suspicious names
        df2 = pd.DataFrame({
            'feature1': np.random.randn(100),
            'label_feature': np.random.randn(100),  # Contains 'label'
            'target': np.random.randn(100)
        })
        
        leakage_risk = monitor._detect_target_leakage(df2, 'target')
        assert leakage_risk is RiskLevel.MEDIUM
    
    def test_assess_statistical_anomalies(self, monitor, problematic_data):
        """Test statistical anomaly detection"""
        report = monitor._assess_statistical_anomalies(problematic_data)
        
        assert 'warnings' in report
        
        # Should detect zero variance in constant column
        warning_types = [w['type'] for w in report['warnings']]
        assert 'zero_variance' in warning_types
        
        # Check constant column is identified
        zero_var_warnings = [w for w in report['warnings'] if w['type'] == 'zero_variance']
        assert any('constant' in w['column'] for w in zero_var_warnings)
    
    def test_assess_skewness(self, monitor):
        """Test skewness detection"""
        # Create highly skewed data
        np.random.seed(42)
        skewed = np.concatenate([
            np.ones(95),
            np.full(5, 100.0)
        ])
        df = pd.DataFrame({
            'normal': np.random.randn(100),
            'skewed': skewed  # Strong right skew
        })
        
        report = monitor._assess_statistical_anomalies(df)
        
        # Should detect high skewness
        warning_types = [w['type'] for w in report['warnings']]
        assert 'high_skewness' in warning_types
        
        skew_warnings = [w for w in report['warnings'] if w['type'] == 'high_skewness']
        assert any('skewed' in w['column'] for w in skew_warnings)
    
    def test_calculate_drift_risk_low(self, monitor):
        """Test drift risk calculation for low-risk data"""
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        risk = monitor._calculate_drift_risk(df)
        assert risk is RiskLevel.LOW
    
    def test_calculate_drift_risk_medium(self, monitor):
        """Test drift risk calculation for medium-risk data"""
        # Mix of high-cardinality categorical data and time features
        df = pd.DataFrame({
            **{f'num_{i}': np.random.randn(100) for i in range(30)},
            'cat1': np.random.choice(['A', 'B'], 100),
            'customer_id': [f'ID_{i}' for i in range(100)],
            'event_time': pd.date_range('2024-01-01', periods=100)
        })

        risk = monitor._calculate_drift_risk(df)
        assert risk is RiskLevel.MEDIUM
    
    def test_calculate_drift_risk_high(self, monitor):
        """Test drift risk calculation for high-risk data"""
        # Many features, high cardinality, and dates
        df = pd.DataFrame({
            **{f'num_{i}': np.random.randn(100) for i in range(60)},
            'high_card1': [f'ID_{i}' for i in range(100)],
            'high_card2': [f'USER_{i}' for i in range(100)],
            'date': pd.date_range('2024-01-01', periods=100)
        })
        
        risk = monitor._calculate_drift_risk(df)
        assert risk is RiskLevel.HIGH
    
    def test_generate_recommendations(self, monitor):
        """Test recommendation generation"""
        issues = [
            {'type': 'missing_critical', 'severity': 'critical'},
            {'type': 'outliers', 'severity': 'high'},
            {'type': 'high_cardinality', 'severity': 'medium'}
        ]
        
        missing_report = {'penalty': 15}
        outlier_report = {'penalty': 8}
        dtype_report = {'penalty': 6}
        
        recommendations = monitor._generate_recommendations(
            issues, missing_report, outlier_report, dtype_report
        )
        
        assert len(recommendations) > 0
        
        # Check for high priority missing data recommendation
        priorities = [r['priority'] for r in recommendations]
        assert 'high' in priorities
        
        # Check categories
        categories = [r['category'] for r in recommendations]
        assert 'missing_data' in categories
        assert 'outliers' in categories
        assert 'data_types' in categories
        assert 'best_practices' in categories


class TestAkkioStyleCleaningAgent:
    """Tests for Akkio-style conversational cleaning agent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider"""
        llm = Mock()
        llm.generate = AsyncMock()
        return llm
    
    @pytest.fixture
    def cleaning_agent(self, mock_llm):
        """Create cleaning agent with mock LLM"""
        return AkkioStyleCleaningAgent(mock_llm)
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with issues"""
        return pd.DataFrame({
            'price': [100, 200, 300, 1000000, 150],  # Has outlier
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],  # Has missing
            'quantity': [1, 2, 3, 4, 5],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })
    
    def test_initialization(self, cleaning_agent):
        """Test agent initialization"""
        assert cleaning_agent.conversation_history == []
        assert cleaning_agent.cleaning_actions == []
        assert cleaning_agent.undo_stack == []
        assert cleaning_agent.llm is not None
    
    @pytest.mark.asyncio
    async def test_chat_clean_remove_outliers(self, cleaning_agent, sample_df):
        """Test chat-based outlier removal"""
        # Mock LLM responses
        cleaning_agent.llm.generate.side_effect = [
            Mock(content=json.dumps({
                'action': 'remove',
                'description': 'Remove outliers from price column using IQR method',
                'columns': ['price'],
                'parameters': {'method': 'IQR', 'threshold': 1.5}
            })),
            Mock(content="""
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
""")
        ]
        
        df_result, response = await cleaning_agent.chat_clean(
            "Remove outliers from the price column",
            sample_df
        )
        
        # Check response format
        assert 'I understand you want to' in response
        assert 'Remove outliers from price column' in response
        assert '**Action:**' in response
        assert '```python' in response
        assert 'Shall I apply these changes?' in response
        
        # Check conversation history updated
        assert len(cleaning_agent.conversation_history) == 1
        assert cleaning_agent.conversation_history[0]['message'] == "Remove outliers from the price column"
        
        # Check cleaning action stored
        assert len(cleaning_agent.cleaning_actions) == 1
        assert 'intent' in cleaning_agent.cleaning_actions[0]
        assert 'code' in cleaning_agent.cleaning_actions[0]
    
    @pytest.mark.asyncio
    async def test_chat_clean_fill_missing(self, cleaning_agent, sample_df):
        """Test chat-based missing value imputation"""
        cleaning_agent.llm.generate.side_effect = [
            Mock(content=json.dumps({
                'action': 'fill',
                'description': 'Fill missing values in name column with "Unknown"',
                'columns': ['name'],
                'parameters': {'strategy': 'constant', 'value': 'Unknown'}
            })),
            Mock(content="df['name'].fillna('Unknown', inplace=True)")
        ]
        
        df_result, response = await cleaning_agent.chat_clean(
            "Fill missing names with Unknown",
            sample_df
        )
        
        assert 'Fill missing values' in response
        assert 'name column' in response
        assert 'fillna' in response
    
    @pytest.mark.asyncio
    async def test_analyze_cleaning_intent(self, cleaning_agent, sample_df):
        """Test cleaning intent analysis"""
        cleaning_agent.llm.generate.return_value = Mock(
            content=json.dumps({
                'action': 'transform',
                'description': 'Convert date strings to datetime',
                'columns': ['date'],
                'parameters': {'format': '%Y-%m-%d'},
                'safety_concerns': []
            })
        )
        
        intent = await cleaning_agent._analyze_cleaning_intent(
            "Convert date column to datetime format",
            sample_df
        )
        
        assert intent['action'] == 'transform'
        assert 'date' in intent['columns']
        assert intent['parameters']['format'] == '%Y-%m-%d'
    
    @pytest.mark.asyncio
    async def test_generate_cleaning_code(self, cleaning_agent, sample_df):
        """Test cleaning code generation"""
        intent = {
            'action': 'filter',
            'description': 'Keep only positive prices',
            'columns': ['price'],
            'parameters': {'condition': '> 0'}
        }
        
        cleaning_agent.llm.generate.return_value = Mock(
            content="```python\ndf = df[df['price'] > 0]\n```"
        )
        
        code = await cleaning_agent._generate_cleaning_code(intent, sample_df)
        
        assert "df = df[df['price'] > 0]" in code
        assert '```' not in code  # Should be cleaned
    
    def test_preview_changes_success(self, cleaning_agent, sample_df):
        """Test successful change preview"""
        code = "df = df[df['price'] < 500]"  # Remove outlier
        
        preview = cleaning_agent._preview_changes(sample_df, code)
        
        assert preview['affected_rows'] == 1  # One row with price=1000000
        assert preview['shape_before'] == (5, 4)
        assert preview['shape_after'] == (4, 4)
        assert 'sample_changes' in preview
        assert 'error' not in preview
    
    def test_preview_changes_error(self, cleaning_agent, sample_df):
        """Test preview with invalid code"""
        code = "df = df[df['nonexistent_column'] > 0]"  # Invalid column
        
        preview = cleaning_agent._preview_changes(sample_df, code)
        
        assert 'error' in preview
        assert preview['affected_rows'] == 0
        assert preview['affected_columns'] == []
    
    def test_apply_cleaning(self, cleaning_agent, sample_df):
        """Test applying cleaning action"""
        # Add action to history
        cleaning_agent.cleaning_actions.append({
            'timestamp': datetime.now(),
            'intent': {'description': 'Remove outliers'},
            'code': "df = df[df['price'] < 500]",
            'preview': {}
        })
        
        df_cleaned = cleaning_agent.apply_cleaning(sample_df)
        
        assert len(df_cleaned) == 4  # Outlier removed
        assert df_cleaned['price'].max() < 500
        assert len(cleaning_agent.undo_stack) == 1  # Original saved for undo
    
    def test_apply_cleaning_with_error(self, cleaning_agent, sample_df):
        """Test applying cleaning with error handling"""
        cleaning_agent.cleaning_actions.append({
            'timestamp': datetime.now(),
            'intent': {'description': 'Invalid operation'},
            'code': "df = df[df['invalid'] > 0]",
            'preview': {}
        })
        
        df_result = cleaning_agent.apply_cleaning(sample_df)
        
        # Should return original on error
        assert df_result.equals(sample_df)
    
    def test_undo_last_action(self, cleaning_agent, sample_df):
        """Test undoing last cleaning action"""
        # Save original state
        cleaning_agent.undo_stack.append(sample_df.copy())
        
        # Modify dataframe
        df_modified = sample_df.drop(0)  # Remove first row
        
        # Undo
        df_restored = cleaning_agent.undo_last_action(df_modified)
        
        assert len(df_restored) == len(sample_df)
        assert df_restored.equals(sample_df)
        assert len(cleaning_agent.undo_stack) == 0  # Stack popped
    
    def test_undo_empty_stack(self, cleaning_agent, sample_df):
        """Test undo with empty stack"""
        df_result = cleaning_agent.undo_last_action(sample_df)
        
        # Should return input unchanged
        assert df_result.equals(sample_df)


class TestIntelligentDataQualityAgent:
    """Tests for combined intelligent agent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider"""
        llm = Mock()
        llm.generate = AsyncMock(return_value=Mock(content='{}'))
        return llm
    
    @pytest.fixture
    def agent_with_llm(self, mock_llm):
        """Create agent with LLM"""
        return IntelligentDataQualityAgent(mock_llm)
    
    @pytest.fixture
    def agent_without_llm(self):
        """Create agent without LLM"""
        return IntelligentDataQualityAgent(None)
    
    def test_initialization_with_llm(self, agent_with_llm):
        """Test initialization with LLM provider"""
        assert agent_with_llm.cleaning_agent is not None
        assert agent_with_llm.quality_monitor is not None
    
    def test_initialization_without_llm(self, agent_without_llm):
        """Test initialization without LLM provider"""
        assert agent_without_llm.cleaning_agent is None
        assert agent_without_llm.quality_monitor is not None
    
    def test_assess_delegation(self, agent_with_llm):
        """Test that assess delegates to quality monitor"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        assessment = agent_with_llm.assess(df)
        
        assert isinstance(assessment, DataQualityAssessment)
        assert 0 <= assessment.quality_score <= 100
        assert isinstance(assessment.alerts, list)
        assert isinstance(assessment.warnings, list)
    
    @pytest.mark.asyncio
    async def test_clean_with_llm(self, agent_with_llm):
        """Test cleaning with LLM available"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        agent_with_llm.cleaning_agent.llm.generate.return_value = Mock(
            content=json.dumps({'action': 'none', 'description': 'No action needed'})
        )
        
        df_result, response = await agent_with_llm.clean("Clean data", df)
        
        assert isinstance(df_result, pd.DataFrame)
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_clean_without_llm(self, agent_without_llm):
        """Test cleaning without LLM raises error"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="LLM provider required"):
            await agent_without_llm.clean("Clean data", df)
    
    def test_get_quality_report_comprehensive(self, agent_with_llm):
        """Test comprehensive quality report generation"""
        assessment = DataQualityAssessment(
            quality_score=75.5,
            alerts=[
                {'message': 'High missing values in column X', 'action': 'Impute or drop'},
                {'message': 'Severe class imbalance', 'action': 'Use resampling'}
            ],
            warnings=[
                {'message': 'Outliers detected in column Y'},
                {'message': 'High cardinality in column Z'},
                {'message': 'Skewed distribution in column W'}
            ],
            recommendations=[
                {
                    'title': 'Address Missing Data',
                    'priority': 'high',
                    'description': 'Significant missing data detected',
                    'actions': ['Use KNN imputation', 'Drop columns with >50% missing']
                },
                {
                    'title': 'Handle Outliers',
                    'priority': 'medium',
                    'description': 'Multiple columns contain outliers',
                    'actions': ['Apply winsorization', 'Use robust scaling']
                }
            ],
            statistics={
                'rows': 1000,
                'columns': 20,
                'missing_cells': 500,
                'duplicate_rows': 10
            },
            drift_risk=RiskLevel.MEDIUM,
            target_leakage_risk=RiskLevel.HIGH,
            visualization_data={}
        )
        
        report = agent_with_llm.get_quality_report(assessment)
        
        # Check report structure
        assert '# Data Quality Assessment Report' in report
        assert 'Overall Quality Score: 75.5/100' in report
        
        # Check sections
        assert '## Critical Alerts (2)' in report
        assert '## Warnings (3)' in report
        assert '## Key Statistics' in report
        assert '## Risk Assessment' in report
        assert '## Top Recommendations' in report
        
        # Check content
        assert 'High missing values' in report
        assert 'Severe class imbalance' in report
        assert 'Data Drift Risk: **medium**' in report
        assert 'Target Leakage Risk: **High**' in report
        
        # Check recommendations formatting
        assert '### 1. Address Missing Data' in report
        assert 'Priority: high' in report
        assert '**Actions:**' in report
    
    def test_get_quality_report_minimal(self, agent_with_llm):
        """Test quality report with minimal issues"""
        assessment = DataQualityAssessment(
            quality_score=95.0,
            alerts=[],
            warnings=[],
            recommendations=[
                {
                    'title': 'Best Practices',
                    'priority': 'low',
                    'description': 'General recommendations',
                    'actions': ['Monitor performance']
                }
            ],
            statistics={'rows': 100, 'columns': 5},
            drift_risk=RiskLevel.LOW,
            target_leakage_risk=RiskLevel.LOW,
            visualization_data={}
        )
        
        report = agent_with_llm.get_quality_report(assessment)
        
        assert 'Overall Quality Score: 95.0/100' in report
        assert '## Critical Alerts (0)' in report
        assert '## Warnings (0)' in report
        assert 'Data Drift Risk: **low**' in report
        assert 'Target Leakage Risk: **Low**' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

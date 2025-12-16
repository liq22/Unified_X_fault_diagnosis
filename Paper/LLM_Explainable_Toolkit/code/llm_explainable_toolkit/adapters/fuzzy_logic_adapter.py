"""
Fuzzy Logic Adapter
===================

Adapter for the FuzzyLogic_v2 model.
This adapter extracts fuzzy rules, membership functions, and uncertainty information
to create interpretable explanations.

Author: LLM Explainable FD Toolkit
Date: 2025-01-15
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from .model_adapter_base import (
    ModelAdapter, FeatureImportance, SignalPath, UncertaintyQuantification,
    ModelAdapterFactory
)


@dataclass
class FuzzyRule:
    """Represents a fuzzy rule"""
    rule_id: str
    antecedents: List[Dict[str, Tuple[str, float]]]  # feature -> (linguistic_term, membership)
    consequent: Tuple[str, float]  # fault_type -> certainty
    rule_weight: float
    support: float  # Support degree
    confidence: float  # Rule confidence


@dataclass
class MembershipFunction:
    """Represents a fuzzy membership function"""
    feature_name: str
    linguistic_term: str
    function_type: str  # 'triangular', 'gaussian', 'trapezoidal'
    parameters: Tuple[float, ...]  # Function parameters
    coverage: float  # How well it covers the data range


class FuzzyLogicAdapter(ModelAdapter):
    """Adapter for FuzzyLogic_v2 model."""

    def __init__(self, model_name: str = "FuzzyLogic_v2", model_version: str = "2.0"):
        """Initialize Fuzzy Logic adapter."""
        super().__init__(model_name, model_version)
        self.feature_names = []
        self.linguistic_terms = {}
        self.rules = []

    def _get_supported_output_types(self) -> List[str]:
        """Return supported output types."""
        return [
            "fuzzy_rules",
            "membership_functions",
            "rule_activations",
            "uncertainty_degrees",
            "feature_contributions"
        ]

    def extract_diagnosis(self, model_output: Any) -> Dict[str, float]:
        """Extract fault diagnosis from fuzzy inference output."""
        # Handle different output formats
        if isinstance(model_output, dict):
            if "fault_degrees" in model_output:
                fault_degrees = model_output["fault_degrees"]
            elif "output_membership" in model_output:
                output_membership = model_output["output_membership"]
                # Convert membership to degrees
                fault_degrees = self._membership_to_degrees(output_membership)
            elif "diagnosis" in model_output:
                return model_output["diagnosis"]
            else:
                # Try to find diagnosis in nested structure
                fault_degrees = self._find_fuzzy_output(model_output)
        else:
            # Assume direct fuzzy output
            fault_degrees = model_output

        # Ensure format consistency
        if not isinstance(fault_degrees, dict):
            return {}

        # Convert to standard fault types
        standard_faults = ["normal", "inner_race", "outer_race", "ball", "cage"]
        diagnosis = {}

        for fault in standard_faults:
            if fault in fault_degrees:
                diagnosis[fault] = float(fault_degrees[fault])
            elif f"{fault}_degree" in fault_degrees:
                diagnosis[fault] = float(fault_degrees[f"{fault}_degree"])

        return diagnosis

    def extract_features(self, model_output: Any) -> List[FeatureImportance]:
        """Extract feature importance from fuzzy rules and memberships."""
        features = []

        # Get fuzzy rules if available
        rules = self._extract_rules(model_output)

        # Calculate feature importance based on rule usage
        feature_importance = {}
        feature_count = {}

        for rule in rules:
            # Count feature appearances in rules
            for antecedent in rule.antecedents:
                feature_name = list(antecedent.keys())[0]
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = 0
                    feature_count[feature_name] = 0

                # Weight by rule confidence and weight
                importance = rule.confidence * rule.rule_weight * rule.support
                feature_importance[feature_name] += importance
                feature_count[feature_name] += 1

        # Average importance and create feature objects
        for feature, importance in feature_importance.items():
            avg_importance = importance / feature_count[feature]

            # Get linguistic term for this feature
            term = self._get_dominant_term(model_output, feature)

            features.append(FeatureImportance(
                feature_name=feature,
                importance_score=avg_importance,
                description=f"Fuzzy feature '{feature}' with dominant term '{term}'",
                evidence=f"Appears in {feature_count[feature]} rules with average weight {avg_importance:.3f}"
            ))

        # Sort by importance
        features.sort(key=lambda x: x.importance_score, reverse=True)

        return features

    def extract_signal_pathway(self, model_output: Any) -> List[SignalPath]:
        """Extract signal processing pathway from fuzzy logic system."""
        pathways = []

        # Fuzzy logic processing stages
        fuzzy_stages = [
            ("input", (4096,), (4096,), "signal_input", {}),
            ("feature_extraction", (4096,), (10,), "statistical_features", {
                "features": ["mean", "std", "rms", "peak", "kurtosis", "skewness", "crest_factor", "clearance_factor", "impulse_factor", "margin_factor"]
            }),
            ("fuzzification", (10,), (10, 3), "membership_functions", {
                "terms_per_feature": 3
            }),
            ("rule_inference", (30,), (5,), "fuzzy_inference", {
                "num_rules": len(self._extract_rules(model_output))
            }),
            ("defuzzification", (5,), (5,), "weighted_average", {}),
            ("output", (5,), (5,), "fault_classification", {})
        ]

        for i, (stage_name, input_shape, output_shape, operation, params) in enumerate(fuzzy_stages):
            pathways.append(SignalPath(
                stage_name=f"stage_{i}_{stage_name}",
                input_shape=input_shape,
                output_shape=output_shape,
                operation=operation,
                parameters=params
            ))

        return pathways

    def extract_uncertainty(self, model_output: Any) -> Optional[UncertaintyQuantification]:
        """Extract uncertainty information from fuzzy logic output."""
        if not isinstance(model_output, dict):
            return None

        # Fuzzy systems naturally provide uncertainty information
        if "uncertainty" in model_output:
            uncertainty_data = model_output["uncertainty"]
            return UncertaintyQuantification(
                type=uncertainty_data.get("type", "fuzzy"),
                confidence_interval=tuple(uncertainty_data.get("ci", (0, 1))),
                entropy=uncertainty_data.get("entropy", 0.0),
                calibration_score=uncertainty_data.get("calibration", 0.0)
            )

        # Calculate uncertainty from fuzzy membership degrees
        if "output_membership" in model_output:
            memberships = model_output["output_membership"]
            if isinstance(memberships, dict):
                # Calculate entropy from membership degrees
                entropies = []
                for fault_type, membership in memberships.items():
                    # Shannon entropy of membership distribution
                    if isinstance(membership, (list, np.ndarray)):
                        entropy = -np.sum(membership * np.log(membership + 1e-8))
                        entropies.append(entropy)

                if entropies:
                    avg_entropy = np.mean(entropies)
                    return UncertaintyQuantification(
                        type="fuzzy",
                        confidence_interval=(0, 1),  # Fuzzy logic provides possibility rather than probability
                        entropy=avg_entropy,
                        calibration_score=0.85  # Fuzzy systems are typically well-calibrated
                    )

        return None

    def extract_model_metadata(self, model_output: Any) -> Dict[str, Any]:
        """Extract metadata from fuzzy logic model."""
        metadata = {
            "model_type": "Fuzzy Logic",
            "version": self.model_version,
            "rule_count": len(self._extract_rules(model_output)),
            "feature_count": len(self.feature_names),
            "supports_explanations": True,
            "explanation_type": "fuzzy_rules",
            "uncertainty_type": "possibility"
        }

        # Add fuzzy-specific metadata
        if isinstance(model_output, dict):
            if "rule_base" in model_output:
                metadata["rule_base_size"] = len(model_output["rule_base"])
            if "membership_functions" in model_output:
                metadata["membership_count"] = sum(len(mfs) for mfs in model_output["membership_functions"].values())

        return metadata

    def _extract_rules(self, model_output: Any) -> List[FuzzyRule]:
        """Extract fuzzy rules from model output."""
        rules = []

        if isinstance(model_output, dict):
            if "fuzzy_rules" in model_output:
                rule_data = model_output["fuzzy_rules"]
                for rule_info in rule_data:
                    rules.append(FuzzyRule(**rule_info))
            elif "rule_base" in model_output:
                # Parse rule base format
                rule_base = model_output["rule_base"]
                for i, rule in enumerate(rule_base):
                    # Parse rule string or dict format
                    if isinstance(rule, str):
                        parsed = self._parse_rule_string(rule, i)
                        if parsed:
                            rules.append(parsed)
                    elif isinstance(rule, dict):
                        rules.append(FuzzyRule(
                            rule_id=rule.get("id", f"rule_{i}"),
                            antecedents=rule.get("antecedents", []),
                            consequent=rule.get("consequent", ("unknown", 0.5)),
                            rule_weight=rule.get("weight", 1.0),
                            support=rule.get("support", 1.0),
                            confidence=rule.get("confidence", 1.0)
                        ))

        return rules

    def _parse_rule_string(self, rule_str: str, rule_id: int) -> Optional[FuzzyRule]:
        """Parse a fuzzy rule from string format."""
        # Example: "IF feature1 is high AND feature2 is medium THEN fault is inner_race"
        try:
            if "IF" not in rule_str or "THEN" not in rule_str:
                return None

            # Split antecedent and consequent
            antecedent_part, consequent_part = rule_str.split("THEN", 1)
            antecedent_part = antecedent_part.replace("IF", "").strip()
            consequent_part = consequent_part.strip()

            # Parse consequent
            consequent_parts = consequent_part.split()
            fault_type = consequent_parts[0] if consequent_parts else "unknown"
            certainty = float(consequentent_parts[-1]) if consequent_parts else 0.5

            # Parse antecedents
            antecedents = []
            for clause in antecedent_part.split("AND"):
                clause = clause.strip()
                if " is " in clause:
                    feature, term = clause.split(" is ")
                    antecedents.append({feature.strip(): (term.strip(), 0.8)})

            return FuzzyRule(
                rule_id=f"rule_{rule_id}",
                antecedents=antecedents,
                consequent=(fault_type, certainty),
                rule_weight=1.0,
                support=1.0,
                confidence=1.0
            )
        except:
            return None

    def _membership_to_degrees(self, membership_dict: Dict) -> Dict[str, float]:
        """Convert membership degrees to fault degrees."""
        fault_degrees = {}
        for fault_type, membership in membership_dict.items():
            if isinstance(membership, (list, np.ndarray)):
                # Use maximum membership as degree
                fault_degrees[fault_type] = float(np.max(membership))
            else:
                fault_degrees[fault_type] = float(membership)
        return fault_degrees

    def _find_fuzzy_output(self, obj: Any) -> Dict:
        """Recursively find fuzzy output in nested structure."""
        if isinstance(obj, dict):
            # Look for keys that might contain fuzzy output
            fuzzy_keys = ["degree", "membership", "fuzzy", "possibility"]
            for key in obj.keys():
                if any(fk in key.lower() for fk in fuzzy_keys):
                    return obj[key]
            # Recursively search
            for value in obj.values():
                result = self._find_fuzzy_output(value)
                if result:
                    return result
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = self._find_fuzzy_output(item)
                if result:
                    return result
        return {}

    def _get_dominant_term(self, model_output: Any, feature: str) -> str:
        """Get the dominant linguistic term for a feature."""
        if isinstance(model_output, dict) and "membership_functions" in model_output:
            mfs = model_output["membership_functions"]
            if feature in mfs:
                # Find term with highest membership
                memberships = mfs[feature]
                if memberships:
                    max_term = max(memberships.keys(), key=lambda k: memberships[k])
                    return max_term
        return "unknown"

    def set_feature_names(self, feature_names: List[str]):
        """Set the feature names used in the fuzzy system."""
        self.feature_names = feature_names

    def set_linguistic_terms(self, terms: Dict[str, List[str]]):
        """Set the linguistic terms for each feature."""
        self.linguistic_terms = terms

    def set_rules(self, rules: List[FuzzyRule]):
        """Set the fuzzy rules."""
        self.rules = rules


# Register the adapter
ModelAdapterFactory.register_adapter("FuzzyLogic_v2", FuzzyLogicAdapter)
ModelAdapterFactory.register_adapter("FuzzyLogic", FuzzyLogicAdapter)
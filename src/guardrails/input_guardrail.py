"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # TODO: Initialize guardrail framework
        # Example with Guardrails AI:
        # from guardrails import Guard
        # from guardrails.validators import ValidLength, ToxicLanguage
        # self.guard = Guard().use_many(
        #     ValidLength(min=10, max=1000),
        #     ToxicLanguage(threshold=0.5)
        # )

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for toxic language
        - Check for prompt injection attempts
        - Check query length and format
        - Check for off-topic queries
        """
        violations = []

        # TODO: Implement actual validation
        # Example structure:
        # result = self.guard.validate(query)
        # if not result.validation_passed:
        #     violations = result.errors

        # Placeholder checks
        if len(query) < 5:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low"
            })

        if len(query) > 2000:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium"
            })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": query  # Could be modified version
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        TODO: YOUR CODE HERE Implement toxicity detection
        """
        violations = []
        # Implement toxicity check
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        TODO: YOUR CODE HERE Implement prompt injection detection
        """
        violations = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        TODO: YOUR CODE HERE Implement relevance checking
        """
        violations = []
        # Check if query is about HCI research (or configured topic)
        return violations

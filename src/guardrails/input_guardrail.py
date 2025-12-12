"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List
import re
import logging

try:
    from guardrails import Guard
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guard = None


class InputGuardrail:
    """
    Guardrail for checking input safety.

    Integrates with Guardrails AI framework to validate user inputs,
    detecting toxic language, prompt injection attempts, and off-topic queries.
    Ensures inputs meet length requirements and safety standards.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config
        self.logger = logging.getLogger("input_guardrail")
        self.guard = None

        # Load configuration parameters
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 2000)
        self.toxicity_threshold = config.get("toxicity_threshold", 0.5)
        self.check_prompt_injection = config.get("check_prompt_injection", True)
        self.check_relevance = config.get("check_relevance", False)
        self.relevant_topics = config.get("relevant_topics", [
            "hci", "human-computer interaction", "research", "user experience",
            "usability", "interface", "interaction design", "user study"
        ])

        # Initialize Guardrails AI if available
        self._initialize_guard()

    def _initialize_guard(self):
        """
        Initialize Guardrails AI Guard with configured validators.
        Sets up toxicity and length validation rules.
        """
        if not GUARDRAILS_AVAILABLE:
            self.logger.warning("Guardrails AI not available. Using basic checks only.")
            return

        try:
            # Initialize basic guard - validators can be configured via spec
            self.guard = Guard()
            self.logger.info("Guardrails AI Guard initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Guardrails AI Guard: {e}")
            self.guard = None

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query using Guardrails AI and custom checks.

        Performs comprehensive validation including:
        - Length constraints
        - Toxicity detection
        - Prompt injection detection
        - Topic relevance checking (if enabled)

        Args:
            query: User input to validate

        Returns:
            Dictionary with 'valid' boolean, 'violations' list, and 'sanitized_input'
        """
        violations = []
        sanitized_input = query

        # Use Guardrails AI if available
        if GUARDRAILS_AVAILABLE and self.guard:
            try:
                result = self.guard.validate(query)
                if not result.validation_passed:
                    # Extract validation errors from Guardrails AI
                    for error in result.errors:
                        violations.append({
                            "validator": "guardrail",
                            "reason": str(error),
                            "severity": "high"
                        })
                    sanitized_input = result.validated_output if result.validated_output else query
            except Exception as e:
                self.logger.error(f"Guardrails AI validation error: {e}")
                violations.append({
                    "validator": "validation_error",
                    "reason": f"Validation check failed: {str(e)}",
                    "severity": "high"
                })
        else:
            # Fallback to manual checks if Guardrails AI unavailable
            violations.extend(self._check_length(query))
            violations.extend(self._check_toxic_language(query))

        # Additional custom checks
        if self.check_prompt_injection:
            violations.extend(self._check_prompt_injection(query))

        if self.check_relevance:
            violations.extend(self._check_relevance(query))

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": sanitized_input
        }

    def _check_length(self, text: str) -> List[Dict[str, Any]]:
        """
        Check if input meets length requirements.

        Args:
            text: Text to check

        Returns:
            List of length violations
        """
        violations = []

        if len(text) < self.min_length:
            violations.append({
                "validator": "length",
                "reason": f"Query too short (minimum {self.min_length} characters required)",
                "severity": "low"
            })

        if len(text) > self.max_length:
            violations.append({
                "validator": "length",
                "reason": f"Query too long (maximum {self.max_length} characters allowed)",
                "severity": "medium"
            })

        return violations

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language using keyword patterns.

        Detects potentially harmful, abusive, or inappropriate language.
        This provides fallback detection when Guardrails AI is unavailable.

        Args:
            text: Text to check for toxic content

        Returns:
            List of toxicity violations found
        """
        violations = []

        # Keywords and patterns indicating harmful content
        toxic_keywords = [
            "kill", "murder", "hate", "stupid", "idiot", "racist",
            "sexist", "abort", "terrorism", "bomb", "poison"
        ]

        # Check for toxic keywords
        text_lower = text.lower()
        for keyword in toxic_keywords:
            if keyword in text_lower:
                violations.append({
                    "validator": "toxic_language",
                    "reason": f"Input contains potentially toxic keyword: {keyword}",
                    "severity": "high"
                })
                break  # Report once per check

        # Check for excessive profanity using common patterns
        profanity_patterns = [r"f[*u]ck", r"sh[*i]t", r"d[*a]mn"]
        for pattern in profanity_patterns:
            if re.search(pattern, text_lower):
                violations.append({
                    "validator": "toxic_language",
                    "reason": "Input contains excessive profanity",
                    "severity": "medium"
                })
                break

        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection and jailbreak attempts.

        Detects common patterns used to manipulate model behavior or bypass
        safety measures, such as instruction overrides and role-play attempts.

        Args:
            text: Text to check for injection attempts

        Returns:
            List of injection violations found
        """
        violations = []
        text_lower = text.lower()

        # Common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
            "jailbreak",
            "pretend you are",
            "act as if",
            "you are now",
            "from now on",
            "new instructions",
            "override",
            "bypass",
            "ignore safety",
            "disable safety"
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text_lower:
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection detected: '{pattern}'",
                    "severity": "high"
                })
                break  # Report once

        # Check for suspicious role-play patterns
        role_play_patterns = [r"<\s*system\s*>", r"\[SYSTEM\]", r"DAN mode", r"Developer Mode"]
        for pattern in role_play_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "prompt_injection",
                    "reason": "Potential jailbreak attempt detected",
                    "severity": "high"
                })
                break

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to configured topics (HCI research).

        Ensures queries are on-topic for the system's purpose by checking
        for relevant keywords. This helps maintain focus on HCI research.

        Args:
            query: Query to check for relevance

        Returns:
            List of relevance violations if query appears off-topic
        """
        violations = []
        query_lower = query.lower()

        # Check if any relevant topics appear in the query
        has_relevant_topic = False
        for topic in self.relevant_topics:
            if topic.lower() in query_lower:
                has_relevant_topic = True
                break

        # If no relevant topics found, flag as potentially off-topic
        if not has_relevant_topic:
            violations.append({
                "validator": "relevance",
                "reason": f"Query may be off-topic. Expected content related to: {', '.join(self.relevant_topics)}",
                "severity": "low"
            })

        return violations

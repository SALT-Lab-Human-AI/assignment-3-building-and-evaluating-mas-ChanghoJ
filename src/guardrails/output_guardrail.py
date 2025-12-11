"""
Output Guardrail
Checks system outputs for safety violations.
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


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    Integrates with Guardrails AI framework to validate system outputs,
    detecting harmful content, PII, factual inconsistencies, and bias.
    Ensures responses are safe, accurate, and appropriately redacted.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config
        self.logger = logging.getLogger("output_guardrail")
        self.guard = None

        # Load configuration parameters
        self.check_pii = config.get("check_pii", True)
        self.check_harmful = config.get("check_harmful", True)
        self.check_consistency = config.get("check_consistency", False)
        self.check_bias = config.get("check_bias", False)
        self.toxicity_threshold = config.get("toxicity_threshold", 0.3)

        # PII redaction settings
        self.redact_pii = config.get("redact_pii", True)
        self.pii_types_to_check = config.get("pii_types", [
            "email", "phone", "ssn", "credit_card", "ip_address"
        ])

        # Initialize Guardrails AI if available
        self._initialize_guard()

    def _initialize_guard(self):
        """
        Initialize Guardrails AI Guard for output validation.
        Sets up validators for toxicity, PII, and other safety checks.
        """
        if not GUARDRAILS_AVAILABLE:
            self.logger.warning("Guardrails AI not available. Using basic checks only.")
            return

        try:
            # Initialize basic guard - validators configured via spec
            self.guard = Guard()
            self.logger.info("Guardrails AI Guard for output initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Guardrails AI output Guard: {e}")
            self.guard = None

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response using Guardrails AI and custom checks.

        Performs comprehensive validation including:
        - PII detection and redaction
        - Harmful/toxic content detection
        - Factual consistency with sources (if provided)
        - Bias detection in language

        Args:
            response: Generated response to validate
            sources: Optional list of sources used for fact-checking

        Returns:
            Dictionary with 'valid' boolean, 'violations' list, and 'sanitized_output'
        """
        violations = []

        # Use Guardrails AI if available
        if GUARDRAILS_AVAILABLE and self.guard:
            try:
                result = self.guard.validate(response)
                if not result.validation_passed:
                    # Extract validation errors from Guardrails AI
                    for error in result.errors:
                        violations.append({
                            "validator": "guardrail",
                            "reason": str(error),
                            "severity": "high"
                        })
            except Exception as e:
                self.logger.error(f"Guardrails AI validation error: {e}")
                violations.append({
                    "validator": "validation_error",
                    "reason": f"Output validation failed: {str(e)}",
                    "severity": "high"
                })
        else:
            # Fallback to custom checks if Guardrails AI unavailable
            if self.check_pii:
                violations.extend(self._check_pii(response))

            if self.check_harmful:
                violations.extend(self._check_harmful_content(response))

        # Additional custom checks
        if self.check_consistency and sources:
            violations.extend(self._check_factual_consistency(response, sources))

        if self.check_bias:
            violations.extend(self._check_bias(response))

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": self._sanitize(response, violations) if violations else response
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information (PII).

        Detects various PII patterns including emails, phone numbers,
        SSNs, credit cards, and IP addresses.

        Args:
            text: Text to scan for PII

        Returns:
            List of PII violations found
        """
        violations = []

        # Comprehensive regex patterns for PII detection
        pii_patterns = {
            "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email address"),
            "phone": (r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b", "Phone number"),
            "ssn": (r"\b(?!000|666|9)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", "Social Security Number"),
            "credit_card": (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "Credit card number"),
            "ip_address": (r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", "IP address"),
            "zipcode": (r"\b\d{5}(?:-\d{4})?\b", "ZIP code")
        }

        # Check only configured PII types
        for pii_type in self.pii_types_to_check:
            if pii_type in pii_patterns:
                pattern, description = pii_patterns[pii_type]
                matches = re.findall(pattern, text)
                if matches:
                    violations.append({
                        "validator": "pii",
                        "pii_type": pii_type,
                        "reason": f"Response contains {description}",
                        "severity": "high",
                        "match_count": len(matches),
                        "matches": matches[:3]  # Only store first 3 matches to avoid token bloat
                    })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful, toxic, or inappropriate content in responses.

        Uses keyword patterns and contextual detection to identify potentially
        harmful content that should not be returned to users.

        Args:
            text: Text to check for harmful content

        Returns:
            List of harmful content violations found
        """
        violations = []
        text_lower = text.lower()

        # Categories of harmful content
        harmful_keywords = {
            "violent": ["kill", "murder", "violent", "assault", "brutal"],
            "dangerous": ["bomb", "poison", "toxin", "lethal", "fatal"],
            "discriminatory": ["racist", "sexist", "homophobic", "transphobic"],
            "illegal": ["hack", "steal", "fraud", "illegal", "blackmail"]
        }

        for category, keywords in harmful_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    violations.append({
                        "validator": "harmful_content",
                        "category": category,
                        "reason": f"Response contains potentially harmful {category} content",
                        "severity": "high",
                        "keyword": keyword
                    })
                    break  # Report once per category

        # Check for concerning phrases that indicate harmful intent
        concerning_phrases = [
            "how to harm",
            "instructions for",
            "step by step",
            "follow these steps to",
            "execute this"
        ]

        for phrase in concerning_phrases:
            if phrase in text_lower:
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"Response may contain harmful instructions",
                    "severity": "high",
                    "phrase": phrase
                })
                break

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response claims are consistent with provided sources.

        Performs basic fact-checking by comparing claims in the response
        against information from source documents. This is a fallback check
        that uses keyword matching; more sophisticated checking would require
        semantic similarity or LLM-based verification.

        Args:
            response: Generated response to verify
            sources: List of source documents with content and metadata

        Returns:
            List of consistency violations found
        """
        violations = []

        if not sources:
            return violations

        # Extract key claims from response (simple heuristic)
        # This is a basic implementation - production systems would use more
        # sophisticated claim extraction and verification

        response_lower = response.lower()

        # Check if response references any sources
        source_mentions = 0
        source_text = " ".join([s.get("content", "").lower() for s in sources if "content" in s])

        # Basic verification: check if key terms from sources appear in response
        # or vice versa to ensure some connection
        source_terms = [term.lower() for term in source_text.split() if len(term) > 5]
        response_terms = set(response_lower.split())

        matching_terms = sum(1 for term in source_terms if term in response_terms)
        total_source_terms = len(set(source_terms))

        # If very few matching terms, response may not be grounded in sources
        if total_source_terms > 0:
            overlap_ratio = matching_terms / total_source_terms
            if overlap_ratio < 0.2:  # Less than 20% term overlap
                violations.append({
                    "validator": "factual_consistency",
                    "reason": "Response may not be well-grounded in provided sources",
                    "severity": "medium",
                    "overlap_ratio": overlap_ratio
                })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased, discriminatory, or stereotyping language in response.

        Detects language that may contain gender bias, racial bias, age bias,
        ability bias, and other forms of stereotyping that could cause harm
        or reinforce negative biases.

        Args:
            text: Text to check for biased language

        Returns:
            List of bias violations found
        """
        violations = []
        text_lower = text.lower()

        # Patterns indicating various types of bias
        bias_patterns = {
            "gender_bias": [
                "all women are", "all men are", "unlike women", "typical man",
                "female hysteria", "man up", "girls are", "boys are"
            ],
            "racial_bias": [
                "these people", "you people", "those types", "racial slur"
            ],
            "age_bias": [
                "old people are", "young people are stupid", "millennials are lazy"
            ],
            "ability_bias": [
                "mentally ill", "crazy", "lame", "retard", "wheelchair bound"
            ],
            "stereotyping": [
                "naturally gifted at", "so good at", "all", "never good at"
            ]
        }

        for bias_type, patterns in bias_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    violations.append({
                        "validator": "bias",
                        "bias_type": bias_type,
                        "reason": f"Response contains potential {bias_type}",
                        "severity": "medium",
                        "pattern": pattern
                    })
                    break  # Report once per category

        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing or redacting safety violations.

        Redacts PII and replaces harmful content with appropriate notices.
        Preserves readability while ensuring no sensitive information leaks.

        Args:
            text: Text to sanitize
            violations: List of detected violations to address

        Returns:
            Sanitized text with violations redacted
        """
        sanitized = text

        # Track what was redacted for logging
        redaction_types = set()

        # Redact PII if configured
        if self.redact_pii:
            for violation in violations:
                if violation.get("validator") == "pii":
                    pii_type = violation.get("pii_type", "unknown")
                    redaction_types.add(pii_type)

                    # Redact matches
                    for match in violation.get("matches", []):
                        if isinstance(match, tuple):
                            # Handle regex groups that return tuples
                            for group in match:
                                if group:
                                    sanitized = sanitized.replace(str(group), f"[{pii_type.upper()}]")
                        else:
                            sanitized = sanitized.replace(str(match), f"[{pii_type.upper()}]")

        # Handle harmful content - add notice
        harmful_violations = [v for v in violations if v.get("validator") == "harmful_content"]
        if harmful_violations:
            redaction_types.add("harmful_content")
            notice = "\n\n[Note: This response has been reviewed and contains potentially harmful content. Please review carefully.]"
            sanitized += notice

        # Handle bias - add notice
        bias_violations = [v for v in violations if v.get("validator") == "bias"]
        if bias_violations:
            redaction_types.add("bias")
            notice = "\n\n[Note: This response may contain biased language. Please interpret with caution.]"
            sanitized += notice

        # Handle consistency issues - add caveat
        consistency_violations = [v for v in violations if v.get("validator") == "factual_consistency"]
        if consistency_violations:
            redaction_types.add("factual_consistency")
            notice = "\n\n[Note: This response may not be fully grounded in the provided sources.]"
            sanitized += notice

        self.logger.info(f"Sanitized output - redacted: {', '.join(redaction_types)}")

        return sanitized

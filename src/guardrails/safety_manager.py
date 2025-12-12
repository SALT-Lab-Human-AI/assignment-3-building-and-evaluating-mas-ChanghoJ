"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import re

try:
    from guardrails import Guard
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guard = None


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    Integrates with Guardrails AI framework to enforce safety policies,
    detect harmful content, and handle violations with configurable responses.
    Logs all safety events for audit trails and analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Initialize Guardrails AI framework
        self.input_guard = None
        self.output_guard = None
        self._initialize_guardrails()

    def _initialize_guardrails(self):
        """
        Initialize Guardrails AI guards for input and output validation.
        Sets up configuration-based policies for content validation.
        """
        if not GUARDRAILS_AVAILABLE:
            self.logger.warning("Guardrails AI not available. Using basic checks only.")
            return

        try:
            # Input guardrail configuration
            input_config = self.config.get("input_guardrail", {})
            input_spec = input_config.get("spec", {})

            if input_spec:
                self.input_guard = Guard.from_dict(input_spec)
                self.logger.info("Input guardrail initialized from config")

            # Output guardrail configuration
            output_config = self.config.get("output_guardrail", {})
            output_spec = output_config.get("spec", {})

            if output_spec:
                self.output_guard = Guard.from_dict(output_spec)
                self.logger.info("Output guardrail initialized from config")

            if self.input_guard or self.output_guard:
                self.logger.info("Guardrails AI initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Guardrails AI: {e}")
            self.input_guard = None
            self.output_guard = None

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process using Guardrails AI.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean, optional 'violations' list,
            and 'sanitized_query' if applicable
        """
        if not self.enabled:
            return {"safe": True}

        violations = []
        sanitized_query = query

        # Use Guardrails AI if available
        if GUARDRAILS_AVAILABLE and self.input_guard:
            try:
                result = self.input_guard.validate(query)
                if not result.validation_passed:
                    # Extract validation errors
                    for error in result.errors:
                        violations.append({
                            "category": "guardrail_violation",
                            "reason": str(error),
                            "severity": "high"
                        })
                    sanitized_query = result.validated_output if result.validated_output else query
            except Exception as e:
                self.logger.error(f"Guardrails AI validation error: {e}")
                violations.append({
                    "category": "validation_error",
                    "reason": f"Safety check failed: {str(e)}",
                    "severity": "high"
                })
        else:
            # Fallback to keyword-based checks if Guardrails AI not available
            prohibited_keywords = ["hack", "attack", "exploit", "bypass", "malware", "virus"]
            for keyword in prohibited_keywords:
                if keyword.lower() in query.lower():
                    violations.append({
                        "category": "prohibited_keyword",
                        "reason": f"Query contains prohibited keyword: {keyword}",
                        "severity": "medium"
                    })

        is_safe = len(violations) == 0

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations,
            "sanitized_query": sanitized_query if not is_safe else query
        }

    def check_output_safety(self, response: str) -> Dict[str, Any]:
        """
        Check if output response is safe to return using Guardrails AI.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean, optional 'violations' list,
            and sanitized/redacted 'response' if applicable
        """
        if not self.enabled:
            return {"safe": True, "response": response}

        violations = []
        sanitized_response = response

        # Use Guardrails AI if available
        if GUARDRAILS_AVAILABLE and self.output_guard:
            try:
                result = self.output_guard.validate(response)
                if not result.validation_passed:
                    # Extract validation errors
                    for error in result.errors:
                        violations.append({
                            "category": "guardrail_violation",
                            "reason": str(error),
                            "severity": "high"
                        })
                    sanitized_response = result.validated_output if result.validated_output else response
            except Exception as e:
                self.logger.error(f"Guardrails AI output validation error: {e}")
                violations.append({
                    "category": "validation_error",
                    "reason": f"Output safety check failed: {str(e)}",
                    "severity": "high"
                })
        else:
            # Fallback basic checks if Guardrails AI not available
            # Check for potential PII patterns
            pii_patterns = {
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "phone": r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b",
                "ssn": r"\b(?!000|666|9)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
            }

            for pii_type, pattern in pii_patterns.items():
                if re.search(pattern, response):
                    violations.append({
                        "category": "potential_pii",
                        "reason": f"Response may contain {pii_type}",
                        "severity": "high"
                    })

        is_safe = len(violations) == 0

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        result = {
            "safe": is_safe,
            "violations": violations,
            "response": response
        }

        # Apply sanitization if configured
        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "sanitize":
                result["response"] = self._sanitize_response(response, violations)
            elif action == "refuse":
                result["response"] = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies."
                )

        return result

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.

        Redacts PII patterns and adds violation context to indicate
        why content was redacted.

        Args:
            response: Response to sanitize
            violations: List of detected violations

        Returns:
            Sanitized response with unsafe content redacted
        """
        sanitized = response

        # Redact potential PII
        pii_patterns = {
            "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
            "phone": (r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b", "[PHONE]"),
            "ssn": (r"\b(?!000|666|9)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", "[SSN]")
        }

        for pii_type, (pattern, replacement) in pii_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)

        # Add violation context if any violations detected
        violation_types = list(set(v.get("category", "unknown") for v in violations))
        violation_note = f"\n\n[Content redacted due to: {', '.join(violation_types)}]"

        return sanitized + violation_note if len(sanitized) > 0 else "[REDACTED]"

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        log_file = self.config.get("safety_log_file")
        if log_file and self.log_events:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []

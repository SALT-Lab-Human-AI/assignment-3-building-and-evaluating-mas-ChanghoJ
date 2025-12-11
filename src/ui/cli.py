"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any
import yaml
import logging
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()

class CLI:
    """
    Command-line interface for the research assistant.

    Provides an interactive loop with command handling, formatted output,
    citation display, optional agent trace previews, and safety indicators.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize AutoGen orchestrator
        try:
            self.orchestrator = AutoGenOrchestrator(self.config)
            self.logger = logging.getLogger("cli")
            self.logger.info("AutoGen orchestrator initialized successfully")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """
        Main CLI loop (async).
        """
        self._print_welcome()
        while self.running:
            try:
                # Get user input
                query = input("\nEnter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("Processing your query...")
                print("=" * 70)

                try:
                    result = await self._process_query_async(query)
                    self.query_count += 1
                    self._display_result(result)
                    self._save_query_result(query, result)
                except Exception as e:
                    print(f"\nError processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._print_goodbye()
                self.running = False
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Error in CLI loop")
    def _save_query_result(self, query: str, result: Dict[str, Any]):
        """Save query and result to outputs/cli_results.json after every query."""
        import json
        from pathlib import Path
        from datetime import datetime

        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "outputs"
        output_dir.mkdir(exist_ok=True)
        results_file = output_dir / "cli_results.json"

        # Prepare entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result
        }

        # Read existing results
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        else:
            data = []

        # Append new entry and save
        data.append(entry)
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)


    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear the screen")
        print("  stats   - Show system statistics")
        print("  quit    - Exit the application")
        print("\nOr enter a research query to get started!")

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThank you for using the Multi-Agent Research Assistant!")
        print("Goodbye!\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')  # nosec


    def _print_stats(self):
        """Print system statistics."""
        print("\nSystem Statistics:")
        print(f"  Queries processed: {self.query_count}")
        print(f"  System: {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic: {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model: {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        print("\n" + "=" * 70)
        print("RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return

        # Display response and safety indicators
        response = result.get("response", "")
        safety = result.get("metadata", {}).get("safety", {})
        if safety.get("violations"):
            print("\n⚠️  Safety warnings detected in response:")
            for v in safety.get("violations", []):
                print(f"  - {v.get('category', 'violation')}: {v.get('reason', '')}")
        print(f"\n{response}\n")

        # Extract and display citations from conversation
        citations = self._extract_citations(result)
        if citations:
            print("\n" + "-" * 70)
            print("📚 CITATIONS")
            print("-" * 70)
            for i, citation in enumerate(citations, 1):
                print(f"[{i}] {citation}")

        # Display metadata
        metadata = result.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("📊 METADATA")
            print("-" * 70)
            print(f"  • Messages exchanged: {metadata.get('num_messages', 0)}")
            print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
            print(f"  • Agents involved: {', '.join(metadata.get('agents_involved', []))}")
            if safety:
                status = "safe" if safety.get("safe", True) else "unsafe"
                print(f"  • Safety status: {status}")

        # Display conversation summary if verbose mode
        if self._should_show_traces():
            self._display_conversation_summary(result.get("conversation_history", []))

        print("=" * 70 + "\n")

    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Extract citations/URLs from conversation history."""
        citations = []

        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")

            # Find URLs in content
            import re
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)

            for url in urls:
                if url not in citations:
                    citations.append(url)

        return citations[:10]  # Limit to top 10

    def _should_show_traces(self) -> bool:
        """Check if agent traces should be displayed."""
        # Check config for verbose mode
        return self.config.get("ui", {}).get("verbose", False)

    def _display_conversation_summary(self, conversation_history: list):
        """Display a summary of the agent conversation."""
        if not conversation_history:
            return

        print("\n" + "-" * 70)
        print("🔍 CONVERSATION SUMMARY")
        print("-" * 70)

        for i, msg in enumerate(conversation_history, 1):
            agent = msg.get("source", "Unknown")
            content = msg.get("content", "")

            # Truncate long content
            preview = content[:150] + "..." if len(content) > 150 else content
            preview = preview.replace("\n", " ")

            print(f"\n{i}. {agent}:")
            print(f"   {preview}")


    async def _process_query_async(self, query: str) -> Dict[str, Any]:
        """Process a query via orchestrator asynchronously."""
        process_fn = getattr(self.orchestrator, "process_query", None)
        if not process_fn:
            raise AttributeError("Orchestrator missing process_query method")
        if asyncio.iscoroutinefunction(process_fn):
            return await process_fn(query)
        return process_fn(query)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args, unknown = parser.parse_known_args()


    # Run CLI
    cli = CLI(config_path=args.config)
    import asyncio
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()

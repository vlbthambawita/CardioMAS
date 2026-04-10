#!/usr/bin/env python3
"""
Multi-Agent Web Extractor CLI
<<<<<<< HEAD
Uses Ollama + Gemma 4 to scrape a website and extract structured data to .md files.
=======
Uses LangGraph + Ollama + Gemma 4 to scrape websites and extract structured data to .md files.
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
"""

import sys
import os
import re
from orchestrator import Orchestrator

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║          Multi-Agent Web Extractor                       ║
<<<<<<< HEAD
║          Powered by Ollama + Gemma 4                     ║
║                                                          ║
║  Agents:  ScrapeAgent → ExtractAgent → WriterAgent       ║
║  Output:  ./output/*.md                                  ║
=======
║          LangGraph  +  Ollama  +  Gemma 4                ║
║                                                          ║
║  Graph:  scrape → extract → write → END                  ║
║  Output: ./output/*.md                                   ║
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  <url>     Extract data from a website URL
  list      List all previously extracted .md files
  clear     Clear the screen
  help      Show this help message
  quit / q  Exit the program
"""


def is_valid_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip()))


def list_outputs(output_dir: str):
<<<<<<< HEAD
=======
    if not os.path.isdir(output_dir):
        print("  No output directory found.\n")
        return
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
    files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".md")],
        reverse=True,
    )
    if not files:
        print("  No extracted files yet.\n")
        return
    print(f"\n  {'#':<4} {'File':<55} {'Size'}")
<<<<<<< HEAD
    print("  " + "-" * 70)
=======
    print("  " + "-" * 72)
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
    for i, fname in enumerate(files, 1):
        path = os.path.join(output_dir, fname)
        size = os.path.getsize(path)
        print(f"  {i:<4} {fname:<55} {size:>6} B")
    print()


def prompt(text: str) -> str:
    try:
        return input(text).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return "quit"


def run_interactive(orchestrator: Orchestrator):
    print(BANNER)
    print(HELP_TEXT)

    while True:
        user_input = prompt("  URL or command > ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "q", "exit"):
            print("\n  Goodbye!\n")
            break

        elif user_input.lower() == "help":
            print(HELP_TEXT)

        elif user_input.lower() == "list":
<<<<<<< HEAD
            list_outputs(orchestrator.writer.output_dir)
=======
            list_outputs(orchestrator.output_dir)
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183

        elif user_input.lower() == "clear":
            os.system("clear")
            print(BANNER)

        elif is_valid_url(user_input):
            try:
                output_path = orchestrator.run(user_input)
                print(f"\n  Saved to: {output_path}\n")
            except Exception as e:
                print(f"\n  [Error] {e}\n")

        else:
            print(f"\n  Unknown command or invalid URL: '{user_input}'")
            print("  URLs must start with http:// or https://")
            print("  Type 'help' for available commands.\n")


def main():
<<<<<<< HEAD
    output_dir = "output"
    orchestrator = Orchestrator(output_dir=output_dir)

    # Non-interactive mode: URL passed as argument
=======
    orchestrator = Orchestrator(output_dir="output")

    # Non-interactive: URL passed as CLI argument
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
    if len(sys.argv) == 2:
        url = sys.argv[1]
        if not is_valid_url(url):
            print(f"Error: invalid URL '{url}'. Must start with http:// or https://")
            sys.exit(1)
        output_path = orchestrator.run(url)
        print(f"Done: {output_path}")
        return

<<<<<<< HEAD
    # Interactive mode
=======
    # Interactive REPL
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
    run_interactive(orchestrator)


if __name__ == "__main__":
    main()

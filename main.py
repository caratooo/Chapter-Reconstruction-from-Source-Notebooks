#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from dotenv import load_dotenv
from ingest import ingest_all, fetch_repo
from selector import run_selection, CANDIDATE_THEMES
from generate import generate_chapter

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a technical chapter from Jupyter notebooks"
    )
    parser.add_argument("--provider", type=str, default="gemini",
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider-specific)")
    parser.add_argument("--theme", type=int, default=None,
                        help="Theme index (0-3). If not set, prompts for selection.")
    parser.add_argument("--output", type=str, default="output/chapter.md",
                        help="Output file path (default: output/chapter.md)")
    parser.add_argument("--top-notebooks", type=int, default=3,
                        help="Number of top notebooks to select (default: 3)")
    parser.add_argument("--list-themes", action="store_true",
                        help="List available themes and exit")

    args = parser.parse_args()

    if args.list_themes:
        print("\nAvailable themes:")
        for i, theme in enumerate(CANDIDATE_THEMES):
            print(f"  [{i}] {theme['name']}")
            print(f"      {theme['description'][:100]}...")
        sys.exit(0)

    api_key = API_KEY
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Add it to a .env file or export it:")
        print("  echo GEMINI_API_KEY=your-key-here > .env")
        sys.exit(1)

    start_time = time.time()

    try:
        # === Stage 1: Ingest ===
        print("\n" + "=" * 60)
        print("Stage 1: Ingesting Notebooks")
        print("=" * 60)
        repo_path = fetch_repo()
        notebooks = ingest_all(repo_path)
        print(f"\n  Total: {len(notebooks)} notebooks ingested")

        # === Stage 2: Select ===
        print("\n" + "=" * 60)
        print("Stage 2: Semantic Selection")
        print("=" * 60)
        material = run_selection(
            notebooks,
            theme_index=args.theme,
            top_n_notebooks=args.top_notebooks,
        )
        print(f"\n  Theme: {material.theme_name}")
        print(f"  Notebooks: {material.selected_notebooks}")
        print(f"  Sections: {len(material.sections)}")

        if not material.sections:
            print("Error: No relevant sections found for this theme. Try a different theme or lower the threshold.")
            sys.exit(1)

        # === Stage 3: Generate ===
        print("\n" + "=" * 60)
        print("Stage 3: Chapter Generation")
        print("=" * 60)
        chapter = generate_chapter(
            material,
            provider=args.provider,
            api_key=api_key,
            model_name=args.model,
        )

        # Build output path from theme name if user didn't specify a custom path
        if args.output == "output/chapter.md":
            slug = re.sub(r"[^\w]+", "_", material.theme_name).strip("_").lower()
            output_path = f"output/{slug}.md"
        else:
            output_path = args.output

        # Never overwrite an existing file
        if os.path.exists(output_path):
            base, ext = os.path.splitext(output_path)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}"):
                counter += 1
            output_path = f"{base}_{counter}{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(chapter)

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Done! Chapter written to: {output_path}")
        print(f"  Length: {len(chapter)} characters")
        print(f"  Time: {elapsed:.1f}s")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

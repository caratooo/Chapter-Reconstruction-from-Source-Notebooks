import os
import re
import time
from selector import SelectedMaterial
from google import genai

CHAPTER_SYSTEM_PROMPT = """You are a technical writer creating a chapter for a machine learning textbook.
Write in a clear, accessible style. Use precise technical language.
Include code examples when they illustrate key concepts.
Do NOT include any preamble like "Here is..." — just output the requested content directly."""

class GeminiBackend:
    """Google Gemini API backend (using google-genai SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.name = f"Gemini ({model_name})"

    def generate(self, prompt: str, system: str = None, max_retries: int = 3) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
                return response.text
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "quota" in err.lower()
                is_unavailable = "503" in err or "unavailable" in err.lower()
                if attempt < max_retries - 1 and (is_rate_limit or is_unavailable):
                    wait = 2 ** (attempt + 1)
                    label = "Model unavailable" if is_unavailable else "Rate limited"
                    print(f"    {label}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise


def create_backend(provider: str, api_key: str, model_name: str = None):
    """Factory for LLM backends."""
    if provider == "gemini":
        return GeminiBackend(api_key, model_name or "gemini-3.1-flash-lite-preview")
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini'.")


def _prepare_source_material(material: SelectedMaterial, max_chars: int = 80000) -> str:
    """Build a condensed version of notebook content for the LLM."""
    parts = []
    current_nb = None
    total_chars = 0

    for sec in material.sections:
        if total_chars > max_chars:
            break
        if sec.notebook_id != current_nb:
            current_nb = sec.notebook_id
            nb = material.notebook_contents.get(sec.notebook_id)
            title = nb.title if nb else sec.notebook_id
            parts.append(f"\n### Source: {title} ({sec.notebook_id})")

        text = sec.cell.source.strip()
        if sec.cell.cell_type == "code":
            if len(text) > 500:
                text = text[:500] + "\n# ... (truncated)"
            parts.append(f"```python\n{text}\n```")
        else:
            parts.append(text)

        total_chars += len(text)

    full_text = "\n\n".join(parts)

    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[... source material truncated ...]"

    return full_text


def _generate_outline(backend, material: SelectedMaterial, source_material: str) -> str:
    """Step 1: Generate chapter outline from source material."""
    print("  Generating outline...")

    prompt = f"""Based on the following source material from machine learning notebooks,
    create a detailed chapter outline.

    THEME: {material.theme_name}
    DESCRIPTION: {material.theme_description}

    The chapter MUST include these sections in this order:
    1. Title — a clear, descriptive chapter title
    2. Table of Contents — structured outline
    3. Summary — a short high-level summary (3-4 sentences)
    4. Introduction — motivation and scope
    5. Core Concepts — key ideas, methods, techniques (this should be the largest section, with 3-6 subsections)
    6. Workflow / System Explanation — how things fit together in practice
    7. Practical Insights — important observations, tips, implications
    8. Limitations / Tradeoffs — assumptions, caveats, constraints

    For each section in the outline, list:
    - The subsection titles
    - Which source notebooks contribute to each subsection
    - Key code examples to include

    SOURCE MATERIAL (first 15000 chars):
    {source_material[:15000]}

    Respond with the outline as a structured markdown document. Include subsection titles and brief notes about what each should cover."""

    return backend.generate(prompt, system=CHAPTER_SYSTEM_PROMPT)


def _generate_section(
    backend,
    section_name: str,
    section_description: str,
    source_material: str,
    outline: str,
    material: SelectedMaterial,
) -> str:
    """Step 2: Generate a single section of the chapter."""
    print(f"  Writing section: {section_name}...")

    if section_name == "Summary":
        # For the summary, we want to keep it very high-level and avoid code examples.
        # So we provide a more focused prompt and less source material.
        prompt = f"""You are writing the "{section_name}" section of a chapter titled "{material.theme_name}".
        
        FULL CHAPTER OUTLINE (for context):
        {outline}

        INSTRUCTIONS FOR THIS SECTION:
        {section_description}

        SOURCE MATERIAL:
        {source_material}

        Write this section now. Requirements:
        - Write clearly and make it AT MOST 5 sentences long — this is just a high-level summary, not a detailed explanation
        - Focus on high-level themes and key takeaways, not details
        - Do NOT include any code examples in this section
        - Do NOT start with any preamble — begin directly with the section heading

        Output ONLY the markdown for this section."""
        
    else:
        prompt = f"""You are writing the "{section_name}" section of a chapter titled "{material.theme_name}".

        FULL CHAPTER OUTLINE (for context):
        {outline}

        INSTRUCTIONS FOR THIS SECTION:
        {section_description}

        SOURCE MATERIAL:
        {source_material}

        Write this section now. Requirements:
        - Write in clear, technical prose suitable for a textbook
        - Include relevant Python code examples (using ```python blocks) when they illustrate concepts (DONT ADD ANY FOR THE SUMMARY SECTION)
        - Use ## and ### headings for structure within the section
        - Reference specific techniques, algorithms, and their implementations
        - Be detailed and substantive (aim for 500-1500 words per major section)
        - Do NOT include content from other sections — only write what belongs in "{section_name}"
        - Do NOT start with any preamble — begin directly with the section heading

        Output ONLY the markdown for this section."""

    return backend.generate(prompt, system=CHAPTER_SYSTEM_PROMPT)


def _generate_toc(sections: dict[str, str]) -> str:
    """Generate a table of contents from the written sections."""
    toc_lines = []
    section_number = 1

    for section_name, content in sections.items():
        toc_lines.append(f"{section_number}. [{section_name}](#{_slugify(section_name)})")

        subsection_num = 1
        for line in content.split("\n"):
            if line.startswith("### "):
                sub_title = line.replace("### ", "").strip()
                # Strip any existing numbering (e.g. "3.1 Title", "1. Title", "3.1. Title")
                sub_title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", sub_title)
                toc_lines.append(f"   - {section_number}.{subsection_num}. {sub_title}")
                subsection_num += 1

        section_number += 1

    return "\n".join(toc_lines)


def _slugify(text: str) -> str:
    """Convert text to a markdown anchor slug."""
    return re.sub(r"[^\w\s-]", "", text.lower()).replace(" ", "-")


def _clean_chapter(text: str) -> str:
    """Clean up common issues in generated text."""
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"^Here (?:is|are) .*?:\n\n", "", text)
    text = text.strip() + "\n"
    return text


def generate_chapter(
    material: SelectedMaterial,
    provider: str = "gemini",
    api_key: str = "",
    model_name: str = None,
) -> str:
    """Full generation pipeline: outline -> sections -> assembly."""
    backend = create_backend(provider, api_key, model_name)
    print(f"  Using LLM backend: {backend.name}")

    source_material = _prepare_source_material(material)
    print(f"  Source material: {len(source_material):,} chars from {len(material.sections)} sections")

    # Step 1: Generate outline
    outline = _generate_outline(backend, material, source_material)
    print(f"  Outline: {len(outline)} chars")

    # Step 2: Define sections and generate each one
    sections_to_write = [
        (
            "Summary",
            "Write a concise 3-4 sentence high level summary of the entire chapter. "
            "Cover the main themes, key techniques discussed, and what the reader will learn.",
        ),
        (
            "Introduction",
            "Write the introduction section. Explain the motivation for studying these topics, "
            "the scope of the chapter, and how the topics connect to each other. "
            "Set up the narrative arc of the chapter.",
        ),
        (
            "Core Concepts",
            "Write the Core Concepts section — this is the main body of the chapter. "
            "Cover the key algorithms, techniques, and ideas from the source material. "
            "Include subsections for each major topic. Use code examples to illustrate. "
            "This should be the longest and most detailed section.",
        ),
        (
            "Workflow / System Explanation",
            "Write the Workflow section explaining how the concepts fit together in practice. "
            "Show how a practitioner would combine these techniques in a real ML pipeline. "
            "Include practical code patterns and workflow diagrams (described in text).",
        ),
        (
            "Practical Insights",
            "Write the Practical Insights section covering tips, best practices, and "
            "important observations from the source material. Focus on advice that helps "
            "practitioners avoid common mistakes and make better decisions.",
        ),
        (
            "Limitations and Tradeoffs",
            "Write the Limitations/Tradeoffs section covering assumptions, caveats, "
            "computational constraints, and when each technique is or isn't appropriate. "
            "Be specific about the conditions under which methods break down.",
        ),
    ]

    generated_sections = {}
    for section_name, section_desc in sections_to_write:
        time.sleep(1)
        # Core Concepts gets the full material; others get a focused subset
        if section_name == "Core Concepts":
            mat = source_material
        else:
            mat = source_material[:20000]

        generated_sections[section_name] = _generate_section(
            backend, section_name, section_desc, mat, outline, material,
        )

    # Step 3: Assemble
    print("  Assembling final chapter...")

    source_list = ", ".join(material.selected_notebooks)

    chapter_parts = [
        f"# {material.theme_name}\n",
        f"*Generated from notebooks: {source_list}*\n",
        "---\n",
        "## Table of Contents\n",
        _generate_toc(generated_sections),
        "\n---\n",
        generated_sections["Summary"],
        "\n---\n",
        generated_sections["Introduction"],
        "\n---\n",
        generated_sections["Core Concepts"],
        "\n---\n",
        generated_sections["Workflow / System Explanation"],
        "\n---\n",
        generated_sections["Practical Insights"],
        "\n---\n",
        generated_sections["Limitations and Tradeoffs"],
    ]

    chapter = "\n\n".join(chapter_parts)
    chapter = _clean_chapter(chapter)

    return chapter

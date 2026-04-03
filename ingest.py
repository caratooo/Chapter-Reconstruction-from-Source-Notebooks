import json
import re
import os
import glob
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

HANDSON_ML3_URL = "https://github.com/ageron/handson-ml3"
_DEFAULT_CLONE_DIR = os.path.join(tempfile.gettempdir(), "handson-ml3-cache")

@dataclass
class Cell:
    cell_type: str  # "markdown" or "code"
    source: str
    heading_level: Optional[int] = None  # 1-6 if it's a heading cell
    heading_text: Optional[str] = None

@dataclass
class NotebookContent:
    notebook_id: str
    filepath: str
    title: str
    cells: list[Cell] = field(default_factory=list)
    markdown_text: str = ""  # all markdown concatenated
    code_text: str = ""  # all code concatenated

    @property
    def full_text(self) -> str:
        return self.markdown_text + "\n" + self.code_text


def fetch_repo() -> str:
    # Check that git is available before attempting any operations
    try:
        subprocess.run(
            ["git", "--version"], capture_output=True, check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Git is not installed or not on PATH. "
            "Install it from https://git-scm.com/ and try again."
        )

    clone_dir = _DEFAULT_CLONE_DIR
    if os.path.isdir(os.path.join(clone_dir, ".git")):
        print(f"  Updating cached repo at {clone_dir} ...")
        try:
            subprocess.run(
                ["git", "-C", clone_dir, "pull", "--ff-only", "--quiet"],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("  Warning: git pull failed, using existing cached repo.")
    else:
        print(f"  Cloning {HANDSON_ML3_URL} into {clone_dir} ...")
        os.makedirs(clone_dir, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", HANDSON_ML3_URL, clone_dir],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to clone repository. Check your network connection.\n{e}"
            )
    return clone_dir


def _extract_title(cells: list[Cell], raw_cells: list[dict], filename: str) -> str:
    """Extract title from bold chapter header, first heading, or filename."""
    # Check raw cells for bold chapter title pattern like **Chapter 4 – Training Models**
    for raw_cell in raw_cells[:5]:
        if raw_cell.get("cell_type") == "markdown":
            src = "".join(raw_cell.get("source", []))
            # Match **Chapter N – Title** or **Chapter N: Title**
            m = re.match(r"\*\*Chapter\s+\d+\s*[–:—-]\s*(.+?)\*\*", src.strip())
            if m:
                return m.group(1).strip()
            # Match just **Some Title**
            m = re.match(r"^\*\*(.+?)\*\*$", src.strip())
            if m and len(m.group(1)) < 100:
                return m.group(1).strip()

    for cell in cells:
        if cell.cell_type == "markdown" and cell.heading_level == 1:
            if cell.heading_text and cell.heading_text.lower() != "setup":
                return cell.heading_text
    # Fallback: derive from filename
    name = os.path.splitext(os.path.basename(filename))[0]
    # Remove leading number prefix like "04_"
    name = re.sub(r"^\d+_", "", name)
    return name.replace("_", " ").title()


def _extract_heading(line: str) -> tuple[Optional[int], Optional[str]]:
    """Check if a line is a markdown heading."""
    m = re.match(r"^(#{1,6})\s+(.+)", line.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None, None


def _is_boilerplate_code(source: str) -> bool:
    """Filter out cells that are just imports, magic commands, or trivial."""
    stripped = source.strip()
    if not stripped:
        return True
    lines = [l.strip() for l in stripped.split("\n") if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return True
    # Pure import blocks
    if all(l.startswith(("import ", "from ", "%", "!")) for l in lines):
        return True
    return False


def _is_exercise_section(cell: Cell) -> bool:
    """Detect exercise/extra sections to potentially exclude."""
    if cell.heading_text:
        lower = cell.heading_text.lower()
        if any(kw in lower for kw in ["exercise", "extra material", "bonus"]):
            return True
    return False


def _is_setup_boilerplate(source: str) -> bool:
    """Detect setup/preamble cells that aren't useful content."""
    lower = source.lower().strip()
    boilerplate_patterns = [
        "this project requires python",
        "this notebook contains all the sample code",
        "colab.research.google.com",
        "requires scikit-learn",
        "common imports",
        "let's start by importing",
        "assert sys.version_info",
    ]
    return any(p in lower for p in boilerplate_patterns)


def parse_notebook(filepath: str, strip_outputs: bool = True) -> NotebookContent:
    """Parse a single .ipynb file into structured content."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in notebook {filepath}: {e}")

    raw_cells_list = nb.get("cells", [])
    cells = []
    in_exercise_section = False
    in_setup_section = False

    for raw_cell in raw_cells_list:
        cell_type = raw_cell.get("cell_type", "")
        source = "".join(raw_cell.get("source", []))

        if cell_type == "markdown":
            # Skip boilerplate preamble cells
            if _is_setup_boilerplate(source):
                continue

            # Check first line for heading
            first_line = source.strip().split("\n")[0] if source.strip() else ""
            level, text = _extract_heading(first_line)

            cell = Cell(
                cell_type="markdown",
                source=source,
                heading_level=level,
                heading_text=text,
            )

            # Track setup section (# Setup heading)
            if text and text.lower() == "setup":
                in_setup_section = True
                continue
            if in_setup_section and level and level <= 1:
                in_setup_section = False

            # Track exercise sections
            if _is_exercise_section(cell):
                in_exercise_section = True
                continue
            if in_exercise_section and level and level <= 2:
                in_exercise_section = False  # new top-level section resets

            if not in_exercise_section and not in_setup_section:
                cells.append(cell)

        elif cell_type == "code" and not in_exercise_section and not in_setup_section:
            if not _is_boilerplate_code(source):
                cell = Cell(cell_type="code", source=source)
                cells.append(cell)

    title = _extract_title(cells, raw_cells_list, filepath)

    md_text = "\n\n".join(c.source for c in cells if c.cell_type == "markdown")
    code_text = "\n\n".join(c.source for c in cells if c.cell_type == "code")

    return NotebookContent(
        notebook_id=os.path.basename(filepath),
        filepath=filepath,
        title=title,
        cells=cells,
        markdown_text=md_text,
        code_text=code_text,
    )


def ingest_all(repo_path: str, pattern: str = "[0-9]*.ipynb") -> list[NotebookContent]:
    """Ingest all matching notebooks from the repository."""
    notebooks = []
    paths = sorted(glob.glob(os.path.join(repo_path, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No notebooks matching '{pattern}' found in {repo_path}. "
            "Check that the repository was cloned correctly."
        )
    for p in paths:
        try:
            nb = parse_notebook(p)
            if nb.cells:  # skip empty
                notebooks.append(nb)
                print(f"  Ingested: {nb.notebook_id} ({len(nb.cells)} cells) - '{nb.title}'")
        except Exception as e:
            print(f"  Warning: Could not parse {p}: {e}")
    return notebooks

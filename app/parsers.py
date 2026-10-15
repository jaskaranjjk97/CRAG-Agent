import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app import config


def parse_filing(file_path: Path) -> Document:
    """Parse a single 10-K filing (HTML or markdown) into a clean Document."""
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # cleaning of html or htm files
    if file_path.suffix.lower() in (".htm", ".html"):
        soup = BeautifulSoup(text, "html.parser")

        # Remove the tags (like:script,styles)
        for tag in soup(["script", "styles"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    elif file_path.suffix.lowe() == ".md":
        pass  # The file is already clean

    else:
        raise ValueError(f"unsupported file uploaded:{file_path.suffix}")

    # Collapsing the excessive whitespaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    # Taking only 300000 characters
    text = text[:300000]

    return Document(
        page_content=text,
        metadata={
            "source": file_path.name,
            "title": file_path.stem.replace("-", " ").title(),
            "ticker": file_path.stem.split("-")[0].upper(),
            "year": file_path.stem.split("-")[-1],
        },
    )


def load_all_filings() -> list[Document]:
    """Load every filing from data/filings/."""
    docs = []
    for f in sorted(config.FILINGS_DIR.glob("*")):
        if f.suffix.lower() not in (".htm", ".html", ".md", ".txt"):
            continue
        try:
            docs.append(parse_filing(f))
        except (OSError, ValueError, UnicodeDecodeError) as e:
            print(f"Skipping {f.name}: {e}")
    return docs

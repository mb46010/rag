import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def migrate_file(file_path: Path, dest_dir: Path):
    """Migrate a single policy file to the new format."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Skip if no content field or already using markdown_file
        if "content" not in data:
            if "markdown_file" in data:
                logger.info(f"Skipping {file_path.name}: Already migrated (has markdown_file).")
                # We still copy the JSON to dest if it doesn't exist?
                # Let's just write it to dest to ensure dest has everything
            else:
                logger.warning(f"Skipping {file_path.name}: No 'content' or 'markdown_file' field found.")
            return

        content = data.pop("content")
        filename_stem = file_path.stem
        md_filename = f"{filename_stem}.md"

        # Write markdown content
        md_path = dest_dir / md_filename
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Update JSON data
        data["markdown_file"] = md_filename

        # Write new JSON
        json_path = dest_dir / file_path.name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"Migrated {file_path.name} -> {json_path.name} + {md_filename}")

    except Exception as e:
        logger.error(f"Failed to migrate {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Migrate policies to external Markdown format.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("documents_old"),
        help="Source directory containing original JSON files.",
    )
    parser.add_argument(
        "--dest-dir", type=Path, default=Path("documents"), help="Destination directory for migrated files."
    )

    args = parser.parse_args()

    source_dir: Path = args.source_dir
    dest_dir: Path = args.dest_dir

    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting migration from {source_dir} to {dest_dir}")

    json_files = list(source_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        migrate_file(file_path, dest_dir)

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()

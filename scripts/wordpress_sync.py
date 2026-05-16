"""
WordPress -> Qdrant sync (CLI).

Reads incremental state from ./tmp/wordpress_sync_state.json (path configurable
via WORDPRESS_STATE_FILE) and fetches only posts/pages/CPTs/media modified
since the last successful run.

Examples:
  python scripts/wordpress_sync.py
  python scripts/wordpress_sync.py --full-resync
  python scripts/wordpress_sync.py --content-types posts,pages
  python scripts/wordpress_sync.py --content-types media --since 2026-01-01T00:00:00
  python scripts/wordpress_sync.py -v                  # DEBUG logging

Schedule on Windows:
  schtasks /Create /SC HOURLY /TN "HubeAli-WPSync" ^
    /TR "python \"d:\\Spect AI\\Hub e Ali\\scripts\\wordpress_sync.py\""
"""

import argparse
import logging
import sys
from pathlib import Path

# Make `app.*` importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("wp-sync")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet the very noisy libraries unless --verbose is passed.
    if not verbose:
        for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_content_types(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync WordPress content into the RAG vector store.")
    parser.add_argument(
        "--content-types",
        help='Comma-separated content types: posts, pages, media, or a CPT slug (e.g. "lectures"). '
             "Defaults to everything configured.",
    )
    parser.add_argument(
        "--full-resync",
        action="store_true",
        help="Ignore the state file and re-ingest everything.",
    )
    parser.add_argument(
        "--since",
        help="ISO-8601 GMT timestamp (e.g. 2026-01-01T00:00:00). "
             "Overrides the state watermark for this run only.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="After a successful run, delete chunks for wp-* doc_ids that "
             "exist in Qdrant but were not visited this run (i.e. items deleted "
             "from WordPress). Implies --full-resync. Skipped automatically if "
             "the run had any errors. Only touches doc_id prefixes for the "
             "content types being synced - local PDFs and unrelated content "
             "are never affected.",
    )
    parser.add_argument(
        "--list-linked-pdfs",
        action="store_true",
        help="Audit mode: discover all PDF links referenced in any current WP "
             "post/page/CPT, dedupe across the site, and print the list. "
             "No downloads, no indexing, no writes to state. Use this to verify "
             "the inventory before running a real sync.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging (includes httpx wire-level traces).",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    # Imported after logging is configured so its init logs respect the level.
    from app.services.ingestion.wordpress_sync import WordPressSync

    sync = WordPressSync()
    try:
        report = sync.sync(
            content_types=_parse_content_types(args.content_types),
            since=args.since,
            full_resync=args.full_resync,
            prune=args.prune,
            list_linked_pdfs_only=args.list_linked_pdfs,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)

    print(report.summary())
    sys.exit(1 if report.total_errors else 0)


if __name__ == "__main__":
    main()

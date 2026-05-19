"""
WordPress -> Qdrant incremental sync (synchronous).

Reuses the existing ingestion pipeline:
  - PDFExtractor for media-library PDFs
  - MultilingualChunker for chunking
  - QdrantIndexer for embedding + upsert
  - language_detector.detect_language for per-page language tags

Doc-id scheme (deterministic so re-syncs overwrite cleanly):
  - wp-post-{id}
  - wp-page-{id}
  - wp-media-{id}              (PDF from WP media library)
  - wp-{cpt_slug}-{id}         (custom post type)

Content-type keys used in the state file:
  - "posts", "pages", "media", "cpt:{slug}"

The orchestrator is synchronous. From an asyncio context (e.g. APScheduler
hook in app/main.py), invoke it via asyncio.to_thread(sync.sync, ...).
"""

import hashlib
import logging
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Comment
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from app.core.config import settings
from app.core.dependencies import (
    get_embedder,
    get_pdf_extractor,
    get_qdrant_client,
)
from app.services.ingestion.chunker import MultilingualChunker
from app.services.ingestion.citation_extractor import (
    extract_chapter_from_title,
    extract_verse_range_from_title,
    extract_volume,
)
from app.services.ingestion.indexer import QdrantIndexer
from app.services.ingestion.language_detector import detect_language
from app.services.ingestion.pdf_extractor import ExtractedPage
from app.services.ingestion.wordpress_client import WordPressClient, describe_exception
from app.services.ingestion.wordpress_state import WordPressSyncState
from app.services.qdrant_setup import COLLECTION_NAME, setup_collection

logger = logging.getLogger(__name__)


@dataclass
class ContentTypeReport:
    fetched: int = 0
    indexed: int = 0  # number of source items successfully indexed
    chunks: int = 0   # total chunks upserted
    skipped: int = 0  # items intentionally skipped (e.g. 404 media files missing on disk)
    errors: int = 0


@dataclass
class SyncReport:
    per_type: dict[str, ContentTypeReport] = field(default_factory=dict)
    pruned: int = 0  # orphaned wp-* doc_ids removed by --prune

    def for_type(self, key: str) -> ContentTypeReport:
        return self.per_type.setdefault(key, ContentTypeReport())

    @property
    def total_errors(self) -> int:
        return sum(r.errors for r in self.per_type.values())

    def summary(self) -> str:
        if not self.per_type:
            return "WordPress sync: nothing to do (no content types configured)"
        rows = []
        for key, r in sorted(self.per_type.items()):
            rows.append(
                f"  {key}: fetched={r.fetched} indexed={r.indexed} "
                f"chunks={r.chunks} skipped={r.skipped} errors={r.errors}"
            )
        if self.pruned:
            rows.append(f"  pruned: {self.pruned} orphan doc_id(s) removed")
        return "WordPress sync report:\n" + "\n".join(rows)


class WordPressSync:
    """
    Orchestrates a single sync pass. Safe to instantiate multiple times -
    pipeline singletons are shared via app.core.dependencies caches.
    """

    def __init__(self) -> None:
        if not settings.WORDPRESS_URL:
            raise RuntimeError("WORDPRESS_URL is not configured")

        self.state = WordPressSyncState(settings.WORDPRESS_STATE_FILE)
        self.client_factory = lambda: WordPressClient(
            base_url=settings.WORDPRESS_URL,
            username=settings.WORDPRESS_USERNAME,
            app_password=settings.WORDPRESS_APP_PASSWORD,
            user_agent=settings.WORDPRESS_USER_AGENT or None,
            verify_ssl=settings.WORDPRESS_VERIFY_SSL,
        )

        # Pipeline singletons.
        self.qdrant = get_qdrant_client()
        setup_collection(self.qdrant, recreate=False)
        self.embedder = get_embedder()
        self.pdf_extractor = get_pdf_extractor()
        self.chunker = MultilingualChunker(embedding_model=self.embedder.dense_model)
        self.indexer = QdrantIndexer(client=self.qdrant, embedder=self.embedder)

        # Linked-PDF discovery state (populated during text-content processing,
        # drained after the main loop in sync()).
        self._linked_pdfs: dict[str, dict] = {}
        self._allowed_pdf_hosts = self._compute_allowed_hosts()

    @staticmethod
    def _compute_allowed_hosts() -> set[str]:
        """Hosts allowed for linked-PDF downloads: WP host + any extras from settings."""
        hosts: set[str] = set()
        wp_host = urlsplit(settings.WORDPRESS_URL).hostname
        if wp_host:
            hosts.add(wp_host.lower())
        for h in (settings.WORDPRESS_LINKED_PDF_HOSTS or "").split(","):
            h = h.strip().lower()
            if h:
                hosts.add(h)
        return hosts

    # ----------------------- public API -----------------------

    def sync(
        self,
        content_types: Optional[list[str]] = None,
        since: Optional[str] = None,
        full_resync: bool = False,
        prune: bool = False,
        list_linked_pdfs_only: bool = False,
        linked_pdfs_only: bool = False,
    ) -> SyncReport:
        """
        Run one sync pass.

        content_types: subset of {"posts", "pages", "media", "cpt:<slug>"};
                       None means "everything configured".
        since: ISO-8601 GMT timestamp to override the state watermark for this run.
        full_resync: if True, ignore the state file entirely (re-ingest everything).
        prune: if True, after a successful run delete any wp-* doc_ids that
               exist in Qdrant but were not visited this run.
               Implies full_resync.
        list_linked_pdfs_only: audit mode - iterate text content only to
               harvest PDF anchors; print the dedupe'd list and exit.
               Skips text re-indexing, media, watermarks, and writes to Qdrant.
        linked_pdfs_only: ingest just the linked PDFs - iterate text content
               only to harvest PDF anchors, then download and index each unique
               PDF. Does NOT re-index text content or advance text watermarks,
               so post/page chunks remain as they are.
        """
        report = SyncReport()
        wanted = self._resolve_content_types(content_types)
        if not wanted:
            logger.warning("WordPress sync: no content types selected")
            return report

        if prune and not full_resync:
            logger.info("prune=True forces full_resync=True (orphan detection requires a complete pass)")
            full_resync = True

        # Both PDF-focused modes share the same "iterate text content for anchors
        # only, do not re-index posts/pages, do not advance text watermarks" path.
        # They differ only in whether the drain phase downloads + indexes.
        text_discovery_only = list_linked_pdfs_only or linked_pdfs_only
        if text_discovery_only:
            wanted = [k for k in wanted if k != "media"]
            if list_linked_pdfs_only:
                logger.info("list-linked-pdfs mode: dry-run discovery, no writes")
            else:
                logger.info("linked-pdfs-only mode: discover anchors and ingest PDFs only")

        user_display = settings.WORDPRESS_USERNAME or "<none>"
        auth_display = "app-password" if settings.WORDPRESS_APP_PASSWORD else "anonymous"
        logger.info(
            f"WordPress sync starting: url={settings.WORDPRESS_URL} "
            f"user={user_display} auth={auth_display} "
            f"content_types={wanted} full_resync={full_resync} prune={prune}"
        )

        # Reset linked-PDF queue for this run (instance is reused by APScheduler).
        self._linked_pdfs = {}

        # Whether linked-PDF discovery is in play for this run.
        text_types_in_scope = [k for k in wanted if k != "media"]
        do_linked_pdfs = (
            settings.WORDPRESS_INGEST_LINKED_PDFS
            and bool(text_types_in_scope)
        )

        # Determine prune scope. Linked PDFs are safe to include only when this
        # run visits EVERY configured text content type - otherwise we'd flag
        # PDFs referenced from unvisited types as orphans and wrongly delete
        # them. Same hazard with --content-types filtering on a regular run:
        # restricting to e.g. "posts" means we never see PDFs that pages link
        # to. We log a warning when --prune is in play with an incomplete scope.
        configured_text_keys = {k for k in self._resolve_content_types(None) if k != "media"}
        visited_text_keys = {k for k in wanted if k != "media"}
        all_text_visited = bool(configured_text_keys) and configured_text_keys <= visited_text_keys

        if text_discovery_only:
            # In PDF-focused modes the only safe prefix is wp-linkedpdf-, since
            # text content isn't re-indexed (so its doc_ids would never end up
            # in `touched` and would all be flagged as orphans).
            prefixes_in_scope: list[str] = ["wp-linkedpdf-"] if all_text_visited else []
            if prune and not all_text_visited:
                logger.warning(
                    f"Prune skipped: --content-types {sorted(visited_text_keys)} "
                    f"doesn't cover all configured text types {sorted(configured_text_keys)}; "
                    "cannot safely identify orphan linked PDFs."
                )
        else:
            prefixes_in_scope = [self._doc_prefix_for_key(k) for k in wanted]
            if do_linked_pdfs:
                if all_text_visited:
                    prefixes_in_scope.append("wp-linkedpdf-")
                elif prune:
                    logger.warning(
                        f"Prune scope excludes wp-linkedpdf-: --content-types "
                        f"{sorted(visited_text_keys)} is a subset of configured text "
                        f"types {sorted(configured_text_keys)}. Linked PDFs referenced "
                        "only from unvisited types would be incorrectly flagged as "
                        "orphans. Run without --content-types (or include all text "
                        "types) to prune linked PDFs."
                    )
        existing_doc_ids: set[str] = set()
        if prune:
            existing_doc_ids = self._list_doc_ids_with_prefixes(prefixes_in_scope)
            logger.info(
                f"Prune: found {len(existing_doc_ids)} existing doc_id(s) in scope "
                f"(prefixes={prefixes_in_scope})"
            )

        touched_doc_ids: set[str] = set()

        with self.client_factory() as wp:
            # Fail fast with a clean diagnostic if the site is unreachable / auth wrong.
            try:
                wp.ping()
            except Exception as e:
                logger.error(f"WordPress connectivity probe failed: {describe_exception(e)}")
                for key in wanted:
                    report.for_type(key).errors += 1
                logger.info(report.summary())
                return report

            for key in wanted:
                watermark = None if full_resync else (since or self.state.get(key))
                if text_discovery_only:
                    logger.info(f"WP discover [{key}] starting (anchors only)")
                else:
                    logger.info(f"WP sync [{key}] starting (modified_after={watermark!r})")
                try:
                    if key == "media":
                        self._sync_media(wp, watermark, report.for_type(key), touched_doc_ids)
                    elif text_discovery_only:
                        route, doc_prefix = self._route_for(key)
                        self._discover_pdfs_in_route(
                            wp,
                            route=route,
                            doc_prefix=doc_prefix,
                            report=report.for_type(key),
                        )
                    else:
                        route, doc_prefix = self._route_for(key)
                        self._sync_text_route(
                            wp,
                            route=route,
                            doc_prefix=doc_prefix,
                            state_key=key,
                            modified_after=watermark,
                            report=report.for_type(key),
                            touched=touched_doc_ids,
                        )
                except Exception as e:
                    logger.error(f"WP sync [{key}] failed: {describe_exception(e)}")
                    logger.debug("Full traceback:", exc_info=True)
                    report.for_type(key).errors += 1

            # After all text content is fetched, drain the linked-PDF queue.
            if do_linked_pdfs:
                # Resurrect any previously-failed URLs so they get retried on
                # this run (even if their host post hasn't changed since last
                # sync). The failed-PDF list is persisted in the state file
                # under a reserved key.
                for failed_url in self.state.get_failed_pdfs():
                    if failed_url not in self._linked_pdfs:
                        self._linked_pdfs[failed_url] = {
                            "url": failed_url,
                            "anchor_text": None,
                            "first_seen_in": "<previously-failed>",
                            "referenced_by": ["<previously-failed>"],
                        }

                logger.info(f"WP sync [linked_pdfs] starting (dry_run={list_linked_pdfs_only})")
                try:
                    self._process_linked_pdfs(
                        wp,
                        report=report.for_type("linked_pdfs"),
                        touched=touched_doc_ids,
                        full_resync=full_resync,
                        dry_run=list_linked_pdfs_only,
                    )
                except Exception as e:
                    logger.error(f"WP sync [linked_pdfs] failed: {describe_exception(e)}")
                    logger.debug("Full traceback:", exc_info=True)
                    report.for_type("linked_pdfs").errors += 1

        if prune:
            if report.total_errors:
                logger.warning(
                    f"Prune skipped: {report.total_errors} error(s) during sync — "
                    "we won't risk deleting chunks for items we may have failed to visit. "
                    "Resolve the errors and re-run."
                )
            else:
                orphans = existing_doc_ids - touched_doc_ids
                if orphans:
                    logger.info(f"Prune: removing {len(orphans)} orphan doc_id(s): "
                                f"{sorted(orphans)[:10]}{'...' if len(orphans) > 10 else ''}")
                    for doc_id in sorted(orphans):
                        self._delete_doc_chunks(doc_id)
                    report.pruned = len(orphans)
                else:
                    logger.info("Prune: no orphans to remove")

        logger.info(report.summary())
        return report

    # ----------------------- text content (posts / pages / CPTs) -----------------------

    def _discover_pdfs_in_route(
        self,
        wp: WordPressClient,
        route: str,
        doc_prefix: str,
        report: ContentTypeReport,
    ) -> None:
        """
        Iterate a text route only to harvest PDF anchors from each item's HTML.
        Does NOT chunk, does NOT embed, does NOT advance the watermark, does
        NOT touch Qdrant. Used by --list-linked-pdfs and --linked-pdfs-only.
        """
        for item in wp.list_content(route, modified_after=None):
            report.fetched += 1
            item_id = item.get("id")
            raw_html = (item.get("content") or {}).get("rendered", "")
            if not raw_html or item_id is None:
                continue
            self._collect_pdf_links(
                raw_html,
                host_url=item.get("link") or settings.WORDPRESS_URL,
                host_doc_id=f"{doc_prefix}-{item_id}",
            )

    def _sync_text_route(
        self,
        wp: WordPressClient,
        route: str,
        doc_prefix: str,
        state_key: str,
        modified_after: Optional[str],
        report: ContentTypeReport,
        touched: set[str],
    ) -> None:
        # Fail-stop watermark: advance only while the run is unbroken. If item N
        # fails, items after N succeed but their modified_gmt does NOT advance
        # the watermark - otherwise we'd leapfrog the failed item and never
        # retry it on a subsequent run.
        safe_watermark: Optional[str] = None
        had_failure = False

        for item in wp.list_content(route, modified_after=modified_after):
            report.fetched += 1
            item_id = item.get("id")
            modified_gmt = item.get("modified_gmt") or item.get("modified") or ""
            if item_id is not None:
                touched.add(f"{doc_prefix}-{item_id}")
            try:
                chunks_indexed = self._process_text_item(item, doc_prefix)
                report.indexed += 1
                report.chunks += chunks_indexed
                if not had_failure and modified_gmt:
                    safe_watermark = modified_gmt
            except Exception as e:
                logger.error(
                    f"WP sync [{state_key}] item id={item_id} failed: {describe_exception(e)}"
                )
                logger.debug("Full traceback:", exc_info=True)
                report.errors += 1
                had_failure = True

        if safe_watermark:
            self.state.set(state_key, safe_watermark)
        if had_failure:
            logger.warning(
                f"WP sync [{state_key}]: watermark held at {safe_watermark!r} "
                "due to earlier failures - successful items past the first failure "
                "will be retried on the next run."
            )

    def _process_text_item(self, item: dict, doc_prefix: str) -> int:
        """Convert one WP post/page/CPT into chunks and upsert."""
        item_id = item["id"]
        slug = item.get("slug") or str(item_id)
        title = self._html_to_text((item.get("title") or {}).get("rendered", ""))
        raw_html = (item.get("content") or {}).get("rendered", "")

        # Parse the body once and reuse the soup for both link extraction and
        # text extraction (was: two BeautifulSoup parses per post, expensive
        # on a 720-post sync).
        body_soup = None
        if raw_html:
            try:
                body_soup = BeautifulSoup(raw_html, "html.parser")
            except Exception as e:
                logger.warning(f"HTML parse failed for {doc_prefix}-{item_id}: {e}")
                body_soup = None

        # Discover PDF links in the body BEFORE we strip HTML to plain text.
        # We dedupe across the whole sync run via self._linked_pdfs.
        if settings.WORDPRESS_INGEST_LINKED_PDFS and body_soup is not None:
            self._collect_pdf_links_from_soup(
                body_soup,
                host_url=item.get("link") or settings.WORDPRESS_URL,
                host_doc_id=f"{doc_prefix}-{item_id}",
            )

        body = self._soup_to_text(body_soup) if body_soup is not None else ""
        body = body.strip()
        if not body:
            logger.info(f"  skip {doc_prefix}-{item_id} ({slug}): empty body")
            return 0

        combined = f"{title}\n\n{body}" if title else body

        doc_id = f"{doc_prefix}-{item_id}"
        # Doc-level citations parsed from the post title (e.g.
        # "AL-ANFAAL (Chapter 8) Verses 1-40" -> chapter_num=8, verse_range="1-40").
        chapter_num = extract_chapter_from_title(title) if title else None
        verse_range = extract_verse_range_from_title(title) if title else None
        page = ExtractedPage(
            text=combined,
            page_number=1,
            source=f"wp:{doc_prefix.removeprefix('wp-')}:{slug}",
            is_ocr=False,
            title=title or slug,
            url=item.get("link") or None,
            content_type=self._display_content_type(doc_prefix),
            chapter_num=chapter_num,
            verse_range=verse_range,
        )
        page.language = detect_language(page.text)

        chunks = self.chunker.chunk_pages([page], doc_id=doc_id)
        if not chunks:
            logger.info(f"  skip {doc_id}: produced 0 chunks")
            return 0

        # Atomic re-index: upsert new chunks first (overwrite by uuid5 ID),
        # then prune any leftover chunks with this doc_id whose chunk_id isn't
        # in the new set. If indexing fails, old content stays in Qdrant
        # rather than disappearing.
        total = self.indexer.index_chunks_replacing(chunks, doc_id=doc_id)
        logger.info(f"  indexed {doc_id} ({slug}): {total} chunks")
        return total

    # ----------------------- media (PDFs) -----------------------

    def _sync_media(
        self,
        wp: WordPressClient,
        modified_after: Optional[str],
        report: ContentTypeReport,
        touched: set[str],
    ) -> None:
        # See note on fail-stop watermark in _sync_text_route - same logic here.
        safe_watermark: Optional[str] = None
        had_failure = False

        for item in wp.list_media_pdfs(modified_after=modified_after):
            report.fetched += 1
            item_id = item.get("id")
            modified_gmt = item.get("modified_gmt") or item.get("modified") or ""
            source_url = item.get("source_url")
            if item_id is not None:
                touched.add(f"wp-media-{item_id}")
            if not source_url:
                logger.warning(f"WP media id={item_id} has no source_url; skipping")
                report.errors += 1
                had_failure = True
                continue

            try:
                pdf_bytes = wp.download_media(source_url)
                slug = item.get("slug") or str(item_id)
                wp_title = self._html_to_text((item.get("title") or {}).get("rendered", "")) or slug
                chunks_indexed = self._process_pdf_bytes(
                    pdf_bytes,
                    doc_id=f"wp-media-{item_id}",
                    source=f"wp:media:{slug}",
                    title=wp_title,
                    url=source_url,
                    tmp_prefix=f"wp-media-{item_id}-",
                )
                report.indexed += 1
                report.chunks += chunks_indexed
                if not had_failure and modified_gmt:
                    safe_watermark = modified_gmt
            except httpx.HTTPStatusError as e:
                # 404/410 on the file itself = WP DB row exists but the file is
                # gone from /wp-content/uploads. Common with old test uploads.
                # Treat as skip-permanent: we can never index it, retrying won't
                # help, and we don't want this to block the watermark or prune.
                status = e.response.status_code if e.response is not None else 0
                if status in (404, 410):
                    logger.warning(
                        f"WP media id={item_id}: source file missing on server "
                        f"(HTTP {status}, url={source_url}). Skipping permanently."
                    )
                    report.skipped += 1
                    if not had_failure and modified_gmt:
                        safe_watermark = modified_gmt
                else:
                    logger.error(f"WP media id={item_id} failed: {describe_exception(e)}")
                    logger.debug("Full traceback:", exc_info=True)
                    report.errors += 1
                    had_failure = True
            except Exception as e:
                logger.error(f"WP media id={item_id} failed: {describe_exception(e)}")
                logger.debug("Full traceback:", exc_info=True)
                report.errors += 1
                had_failure = True

        if safe_watermark:
            self.state.set("media", safe_watermark)
        if had_failure:
            logger.warning(
                f"WP sync [media]: watermark held at {safe_watermark!r} "
                "due to earlier failures - successful items past the first failure "
                "will be retried on the next run."
            )

    def _process_pdf_bytes(
        self,
        pdf_bytes: bytes,
        *,
        doc_id: str,
        source: str,
        title: str,
        url: str,
        tmp_prefix: str = "wp-pdf-",
    ) -> int:
        """
        Generic PDF ingestion: write bytes to a temp file, run the existing
        two-stage extractor, language-tag each page, chunk, and upsert under
        the given doc_id. Used by both WP media items and linked PDFs.
        """
        upload_dir = settings.upload_dir_path
        with tempfile.NamedTemporaryFile(
            dir=str(upload_dir),
            prefix=tmp_prefix,
            suffix=".pdf",
            delete=False,
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            pages = self.pdf_extractor.extract(str(tmp_path))
            # Doc-level volume parsed from the title or URL filename. Same for
            # every page of this PDF; per-page numbers are already set by the
            # extractor as page.page_number.
            volume = extract_volume(title) or extract_volume(url)
            for p in pages:
                p.source = source
                p.language = detect_language(p.text)
                p.title = title
                p.url = url
                p.content_type = "PDF"
                p.volume = volume

            chunks = self.chunker.chunk_pages(pages, doc_id=doc_id)
            if not chunks:
                logger.info(f"  skip {doc_id}: produced 0 chunks")
                return 0

            # Atomic re-index: see note in _process_text_item.
            total = self.indexer.index_chunks_replacing(chunks, doc_id=doc_id)
            logger.info(f"  indexed {doc_id}: {total} chunks")
            return total
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # ----------------------- helpers -----------------------

    def _delete_doc_chunks(self, doc_id: str) -> None:
        """Remove existing chunks for a doc_id before re-indexing (handles shrinks/edits)."""
        try:
            self.qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(
                    filter=Filter(must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ])
                ),
                wait=True,
            )
        except Exception as e:
            logger.warning(f"Pre-index delete for {doc_id} failed (continuing): {e}")

    def _resolve_content_types(self, requested: Optional[list[str]]) -> list[str]:
        configured: list[str] = ["posts", "pages", "media"]
        cpts = [s.strip() for s in (settings.WORDPRESS_CPT_ROUTES or "").split(",") if s.strip()]
        configured.extend(f"cpt:{slug}" for slug in cpts)

        if not requested:
            return configured

        normalized = [c.strip() for c in requested if c.strip()]
        # Accept "lectures" as shorthand for "cpt:lectures" if it's a configured CPT.
        result = []
        for c in normalized:
            if c in configured:
                result.append(c)
            elif f"cpt:{c}" in configured:
                result.append(f"cpt:{c}")
            else:
                logger.warning(f"Ignoring unknown content type: {c!r}")
        return result

    def _route_for(self, key: str) -> tuple[str, str]:
        """Map content-type key to (REST route, doc-id prefix)."""
        if key == "posts":
            return "wp/v2/posts", "wp-post"
        if key == "pages":
            return "wp/v2/pages", "wp-page"
        if key.startswith("cpt:"):
            slug = key.split(":", 1)[1]
            return f"wp/v2/{slug}", f"wp-{slug}"
        raise ValueError(f"Unknown content type key: {key}")

    @staticmethod
    def _doc_prefix_for_key(key: str) -> str:
        """The doc_id prefix used for a given content-type key (e.g. 'wp-post-')."""
        if key == "posts":
            return "wp-post-"
        if key == "pages":
            return "wp-page-"
        if key == "media":
            return "wp-media-"
        if key.startswith("cpt:"):
            return f"wp-{key.split(':', 1)[1]}-"
        raise ValueError(f"Unknown content type key: {key}")

    def _list_doc_ids_with_prefixes(self, prefixes: list[str]) -> set[str]:
        """
        Return the set of distinct doc_ids in the collection whose value
        starts with any of the given prefixes.

        Prefers Qdrant's facet API (qdrant-client >=1.10), which returns
        distinct values for an indexed payload key in one cheap call. Falls
        back to a full scroll for older qdrant-client versions.
        """
        seen = self._facet_doc_ids(prefixes)
        if seen is not None:
            return seen
        return self._scroll_doc_ids(prefixes)

    def _facet_doc_ids(self, prefixes: list[str]) -> Optional[set[str]]:
        """Fast path using payload facet. Returns None if facet API isn't
        available so the caller can fall back to scroll."""
        facet = getattr(self.qdrant, "facet", None)
        if not callable(facet):
            return None
        try:
            # Cap to a large limit; qdrant-client default is small (~100).
            response = facet(
                collection_name=COLLECTION_NAME,
                key="doc_id",
                limit=200_000,
            )
            hits = getattr(response, "hits", None) or []
            seen: set[str] = set()
            for hit in hits:
                value = getattr(hit, "value", None) or getattr(hit, "key", None)
                if not isinstance(value, str):
                    continue
                if any(value.startswith(pref) for pref in prefixes):
                    seen.add(value)
            logger.debug(
                f"_facet_doc_ids: {len(hits)} distinct doc_ids in collection, "
                f"{len(seen)} match prefixes {prefixes}"
            )
            return seen
        except Exception as e:
            logger.debug(f"facet API unavailable ({e}); falling back to scroll")
            return None

    def _scroll_doc_ids(self, prefixes: list[str]) -> set[str]:
        """Fallback path: read every chunk's doc_id payload and dedupe in
        Python. Cheap because the doc_id index lets Qdrant skip vectors."""
        seen: set[str] = set()
        next_offset = None
        page = 0
        while True:
            points, next_offset = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["doc_id"],
                with_vectors=False,
                offset=next_offset,
            )
            page += 1
            for p in points:
                doc_id = (p.payload or {}).get("doc_id")
                if not doc_id:
                    continue
                if any(doc_id.startswith(pref) for pref in prefixes):
                    seen.add(doc_id)
            if next_offset is None:
                break
        logger.debug(f"_scroll_doc_ids: scrolled {page} page(s), found {len(seen)} matches")
        return seen

    # ----------------------- linked-PDF discovery -----------------------

    def _collect_pdf_links(self, html: str, host_url: str, host_doc_id: str) -> None:
        """Parse HTML once and harvest PDF anchors. Thin wrapper for callers
        that only have raw HTML (e.g. _discover_pdfs_in_route)."""
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return
        self._collect_pdf_links_from_soup(soup, host_url, host_doc_id)

    def _collect_pdf_links_from_soup(self, soup, host_url: str, host_doc_id: str) -> None:
        """Find <a href> PDFs in an already-parsed body and add to the dedupe queue."""
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("#", "mailto:", "javascript:")):
                continue
            absolute = urljoin(host_url or settings.WORDPRESS_URL, href)
            normalized = self._normalize_pdf_url(absolute)
            if not normalized:
                continue
            if normalized in self._linked_pdfs:
                # Already queued by an earlier post - record the additional reference.
                self._linked_pdfs[normalized]["referenced_by"].append(host_doc_id)
                continue
            anchor_text = (a.get_text(" ", strip=True) or "")[:200] or None
            self._linked_pdfs[normalized] = {
                "url": normalized,
                "anchor_text": anchor_text,
                "first_seen_in": host_doc_id,
                "referenced_by": [host_doc_id],
            }

    def _normalize_pdf_url(self, url: str) -> Optional[str]:
        """
        Return a canonical PDF URL on an allowed host, or None to drop.
        - Must end in .pdf (case-insensitive), allowing for query strings
        - Must be on an allowed host (WP host + WORDPRESS_LINKED_PDF_HOSTS)
        - Forces https:// on the WP host so http+https variants dedupe
        - Strips fragment; preserves query
        """
        try:
            parts = urlsplit(url)
        except Exception:
            return None
        if not parts.scheme or not parts.netloc:
            return None
        if parts.scheme.lower() not in ("http", "https"):
            return None
        path_lower = parts.path.lower()
        if not path_lower.endswith(".pdf"):
            return None
        host = parts.netloc.lower()
        # Strip port for host comparison
        host_no_port = host.split(":", 1)[0]
        if host_no_port not in self._allowed_pdf_hosts:
            return None
        scheme = "https"  # normalize so http/https variants of the same path dedupe
        return urlunsplit((scheme, host, parts.path, parts.query, ""))

    @staticmethod
    def _linked_pdf_doc_id(url: str) -> str:
        """Stable doc_id for a linked PDF: wp-linkedpdf-<sha1[:16]>."""
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return f"wp-linkedpdf-{digest}"

    # Anchor texts that look like CTAs rather than titles - we'd rather fall
    # through to the filename stem than store these as the document title.
    _GENERIC_ANCHOR_TEXTS = frozenset({
        "download", "download pdf", "download here", "click here", "click",
        "here", "view", "view pdf", "read", "read more", "read here",
        "pdf", "link", "more", "open", "get pdf", "get it", "see",
    })

    @classmethod
    def _derive_pdf_title(cls, url: str, anchor_text: Optional[str]) -> str:
        """Prefer anchor text if meaningful; else derive from the filename stem."""
        if anchor_text:
            text = anchor_text.strip()
            normalized = text.lower()
            if (
                len(text) >= 5
                and normalized not in cls._GENERIC_ANCHOR_TEXTS
                and not normalized.endswith(".pdf")
                and not normalized.startswith(("http://", "https://"))
            ):
                return text
        filename = Path(urlsplit(url).path).name
        stem = Path(filename).stem if filename else ""
        title = stem.replace("-", " ").replace("_", " ").strip()
        return title or "Document"

    def _chunk_exists_for_doc(self, doc_id: str) -> bool:
        """Cheap existence check via the doc_id payload index."""
        try:
            result = self.qdrant.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]),
                exact=False,
            )
            return result.count > 0
        except Exception as e:
            logger.debug(f"existence check for {doc_id} failed: {e}")
            return False

    def _process_linked_pdfs(
        self,
        wp: WordPressClient,
        report: ContentTypeReport,
        touched: set[str],
        full_resync: bool,
        dry_run: bool,
    ) -> None:
        """
        Drain the dedupe queue. For each unique PDF URL:
          - If --dry-run: log it and continue (don't touch failed-list state).
          - Else if doc_id exists and not full_resync: mark touched, skip download.
          - Else: download, extract, chunk, index under wp-linkedpdf-<hash>.
        404/410 are treated as skip-permanent (file gone from server) - clear
        them from the failed-PDF list.
        Other failures are persisted in the failed-PDF list so the next run
        retries them regardless of watermark.
        """
        if not self._linked_pdfs:
            logger.info("No linked PDFs discovered.")
            if not dry_run:
                # Nothing to do, but also nothing failed - leave state alone.
                pass
            return

        logger.info(f"Linked PDFs: {len(self._linked_pdfs)} unique URL(s) to process")

        # Track URLs that hit retryable failures this run; persist at end.
        retry_next_run: set[str] = set(self.state.get_failed_pdfs())

        for url, info in sorted(self._linked_pdfs.items()):
            report.fetched += 1
            doc_id = self._linked_pdf_doc_id(url)
            touched.add(doc_id)

            if dry_run:
                refs = len(info.get("referenced_by") or [])
                logger.info(
                    f"  [dry-run] {url}  ->  {doc_id}  "
                    f"(refs={refs}, first_seen={info.get('first_seen_in')!r})"
                )
                continue

            if not full_resync and self._chunk_exists_for_doc(doc_id):
                logger.info(f"  skip {doc_id} (already indexed): {url}")
                report.skipped += 1
                retry_next_run.discard(url)
                continue

            try:
                pdf_bytes = wp.download_media(url)
                title = self._derive_pdf_title(url, info.get("anchor_text"))
                slug = self._derive_pdf_slug(url)
                chunks_indexed = self._process_pdf_bytes(
                    pdf_bytes,
                    doc_id=doc_id,
                    source=f"wp:linkedpdf:{slug}",
                    title=title,
                    url=url,
                    tmp_prefix=f"wp-linkedpdf-{doc_id[-16:]}-",
                )
                report.indexed += 1
                report.chunks += chunks_indexed
                retry_next_run.discard(url)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (404, 410):
                    logger.warning(
                        f"Linked PDF gone (HTTP {status}): {url}"
                    )
                    report.skipped += 1
                    retry_next_run.discard(url)  # never coming back, stop retrying
                else:
                    logger.error(f"Linked PDF failed {url}: {describe_exception(e)}")
                    report.errors += 1
                    retry_next_run.add(url)
            except Exception as e:
                logger.error(f"Linked PDF failed {url}: {describe_exception(e)}")
                logger.debug("Full traceback:", exc_info=True)
                report.errors += 1
                retry_next_run.add(url)

        if not dry_run:
            # Persist the new failed-PDF list, replacing the old one.
            previous = self.state.get_failed_pdfs()
            if retry_next_run != previous:
                self.state.set_failed_pdfs(retry_next_run)
                if retry_next_run:
                    logger.info(
                        f"Linked PDFs: {len(retry_next_run)} URL(s) will be retried "
                        "on the next run (persisted to state file)"
                    )
                else:
                    logger.info("Linked PDFs: failed-list cleared (all URLs succeeded or are gone)")

    @staticmethod
    def _derive_pdf_slug(url: str) -> str:
        """Slug-ish identifier for the linked PDF (used in `source` payload)."""
        path = urlsplit(url).path
        return path.lstrip("/").replace("/", ":") or "unknown"

    # ----------------------- existing helpers -----------------------

    @staticmethod
    def _display_content_type(doc_prefix: str) -> str:
        """Human-readable label shown in the chat UI ("Article", "Page", ...)."""
        if doc_prefix == "wp-post":
            return "Article"
        if doc_prefix == "wp-page":
            return "Page"
        # CPT: wp-<slug>  ->  Slug (title-cased, hyphens to spaces).
        slug = doc_prefix.removeprefix("wp-")
        return slug.replace("-", " ").replace("_", " ").title() if slug else "Document"

    @classmethod
    def _html_to_text(cls, html: str) -> str:
        """Parse HTML and convert to plain text with structural markers. Thin
        wrapper around _soup_to_text for callers that only have raw HTML."""
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return ""
        return cls._soup_to_text(soup)

    @staticmethod
    def _soup_to_text(soup) -> str:
        """
        Convert a parsed BeautifulSoup body to plain text with structural
        markers preserved as lightweight markdown:
          - <h1>..<h6> become "# Heading" / "## Heading" / etc.
          - <li>     becomes "- item"
          - <br>     becomes a newline
          - block-level elements get separated by blank lines

        Markdown markers help two downstream consumers:
          1. The semantic chunker - blank lines around headings/lists give it
             clean boundaries.
          2. The LLM at generation time - it sees that "## Foo" is a heading
             and can use it as a structural anchor in the answer.
        """
        if soup is None:
            return ""

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c.extract()

        # Headings - prefix with the right number of #s.
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                hashes = "#" * level
                inner = h.get_text(" ", strip=True)
                # Replace the whole heading with a synthetic paragraph that
                # carries the markdown prefix and trailing blank-line marker.
                h.replace_with(BeautifulSoup(
                    f"\n\n{hashes} {inner}\n\n", "html.parser"
                ))

        # List items - prefix with "- "
        for li in soup.find_all("li"):
            inner = li.get_text(" ", strip=True)
            li.replace_with(BeautifulSoup(
                f"\n- {inner}", "html.parser"
            ))

        # <br> -> newline
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Other block-level elements get blank-line separators.
        for block in soup.find_all(["p", "div", "blockquote"]):
            block.append("\n\n")

        text = unescape(soup.get_text())
        # Collapse runs of horizontal whitespace, but preserve newlines.
        text = re.sub(r"[ \t]+", " ", text)
        # Tidy excess blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# A small UUID helper kept here for callers that need to mint deterministic
# doc IDs themselves (e.g. ad-hoc admin scripts). Not used directly above.
def stable_doc_uuid(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

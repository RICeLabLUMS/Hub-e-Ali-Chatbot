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

import logging
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Comment
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from app.core.config import settings
from app.core.dependencies import (
    get_embedder,
    get_pdf_extractor,
    get_qdrant_client,
)
from app.services.ingestion.chunker import MultilingualChunker
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
                f"chunks={r.chunks} errors={r.errors}"
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

    # ----------------------- public API -----------------------

    def sync(
        self,
        content_types: Optional[list[str]] = None,
        since: Optional[str] = None,
        full_resync: bool = False,
        prune: bool = False,
    ) -> SyncReport:
        """
        Run one sync pass.

        content_types: subset of {"posts", "pages", "media", "cpt:<slug>"};
                       None means "everything configured".
        since: ISO-8601 GMT timestamp to override the state watermark for this run.
        full_resync: if True, ignore the state file entirely (re-ingest everything).
        prune: if True, after a successful run delete any wp-* doc_ids that
               exist in Qdrant but were not visited this run (i.e. items
               deleted from WordPress since they were last ingested).
               Implies full_resync - we can only safely identify orphans when
               we've visited every current WP item.
        """
        report = SyncReport()
        wanted = self._resolve_content_types(content_types)
        if not wanted:
            logger.warning("WordPress sync: no content types selected")
            return report

        if prune and not full_resync:
            logger.info("prune=True forces full_resync=True (orphan detection requires a complete pass)")
            full_resync = True

        user_display = settings.WORDPRESS_USERNAME or "<none>"
        auth_display = "app-password" if settings.WORDPRESS_APP_PASSWORD else "anonymous"
        logger.info(
            f"WordPress sync starting: url={settings.WORDPRESS_URL} "
            f"user={user_display} auth={auth_display} "
            f"content_types={wanted} full_resync={full_resync} prune={prune}"
        )

        # Capture pre-existing wp-* doc_ids (scoped to the prefixes we're syncing)
        # so we can identify orphans after the pass.
        prefixes_in_scope: list[str] = [self._doc_prefix_for_key(k) for k in wanted]
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
                logger.info(f"WP sync [{key}] starting (modified_after={watermark!r})")
                try:
                    if key == "media":
                        self._sync_media(wp, watermark, report.for_type(key), touched_doc_ids)
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
                if modified_gmt:
                    self.state.set(state_key, modified_gmt)
            except Exception as e:
                logger.error(
                    f"WP sync [{state_key}] item id={item_id} failed: {describe_exception(e)}"
                )
                logger.debug("Full traceback:", exc_info=True)
                report.errors += 1

    def _process_text_item(self, item: dict, doc_prefix: str) -> int:
        """Convert one WP post/page/CPT into chunks and upsert."""
        item_id = item["id"]
        slug = item.get("slug") or str(item_id)
        title = self._html_to_text((item.get("title") or {}).get("rendered", ""))
        body = self._html_to_text((item.get("content") or {}).get("rendered", ""))
        body = body.strip()
        if not body:
            logger.info(f"  skip {doc_prefix}-{item_id} ({slug}): empty body")
            return 0

        combined = f"{title}\n\n{body}" if title else body

        doc_id = f"{doc_prefix}-{item_id}"
        page = ExtractedPage(
            text=combined,
            page_number=1,
            source=f"wp:{doc_prefix.removeprefix('wp-')}:{slug}",
            is_ocr=False,
            title=title or slug,
            url=item.get("link") or None,
            content_type=self._display_content_type(doc_prefix),
        )
        page.language = detect_language(page.text)

        chunks = self.chunker.chunk_pages([page], doc_id=doc_id)
        if not chunks:
            logger.info(f"  skip {doc_id}: produced 0 chunks")
            return 0

        # Clear prior chunks for this doc so shorter edits don't leave stale
        # higher-indexed chunks behind (deterministic IDs only overwrite where
        # the new chunk_id matches an old one).
        self._delete_doc_chunks(doc_id)

        total = self.indexer.index_chunks(chunks, doc_id=doc_id)
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
                continue

            try:
                pdf_bytes = wp.download_media(source_url)
                chunks_indexed = self._process_pdf_bytes(
                    item_id,
                    item.get("slug") or str(item_id),
                    pdf_bytes,
                    title=self._html_to_text((item.get("title") or {}).get("rendered", ""))
                          or (item.get("slug") or str(item_id)),
                    url=source_url,
                )
                report.indexed += 1
                report.chunks += chunks_indexed
                if modified_gmt:
                    self.state.set("media", modified_gmt)
            except Exception as e:
                logger.error(f"WP media id={item_id} failed: {describe_exception(e)}")
                logger.debug("Full traceback:", exc_info=True)
                report.errors += 1

    def _process_pdf_bytes(
        self,
        item_id: int,
        slug: str,
        pdf_bytes: bytes,
        title: str,
        url: str,
    ) -> int:
        upload_dir = settings.upload_dir_path
        with tempfile.NamedTemporaryFile(
            dir=str(upload_dir),
            prefix=f"wp-media-{item_id}-",
            suffix=".pdf",
            delete=False,
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            doc_id = f"wp-media-{item_id}"
            pages = self.pdf_extractor.extract(str(tmp_path))
            for p in pages:
                # Source the PDF as wp:media:<slug> so it's visible in retrieval payloads
                p.source = f"wp:media:{slug}"
                p.language = detect_language(p.text)
                p.title = title
                p.url = url
                p.content_type = "PDF"

            chunks = self.chunker.chunk_pages(pages, doc_id=doc_id)
            if not chunks:
                logger.info(f"  skip {doc_id}: produced 0 chunks")
                return 0

            self._delete_doc_chunks(doc_id)
            total = self.indexer.index_chunks(chunks, doc_id=doc_id)
            logger.info(f"  indexed {doc_id} ({slug}): {total} chunks")
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
        starts with any of the given prefixes. Uses scroll because Qdrant's
        keyword index doesn't support prefix match - we read each point's
        doc_id and filter in Python. With the doc_id payload index this is
        cheap (no vector deserialization).
        """
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
        logger.debug(f"_list_doc_ids_with_prefixes: scrolled {page} page(s), found {len(seen)} matches")
        return seen

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

    @staticmethod
    def _html_to_text(html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c.extract()
        # Convert <br> to newlines, paragraph-ish blocks get blank-line separators.
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            block.append("\n\n")
        text = unescape(soup.get_text())
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# A small UUID helper kept here for callers that need to mint deterministic
# doc IDs themselves (e.g. ad-hoc admin scripts). Not used directly above.
def stable_doc_uuid(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

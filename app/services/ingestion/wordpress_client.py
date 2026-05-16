"""
Sync WordPress REST client.

Uses httpx.Client (sync) instead of AsyncClient because the anyio-backed
async TLS path on Windows is unreliable (SSLWantReadError + IndexError
on the internal deque during the handshake). The synchronous httpcore
transport goes through socket+ssl directly and is rock-solid.

The orchestrator runs the sync calls in a worker thread when invoked from
an asyncio context (APScheduler default executor).
"""

import logging
import socket
import time
from typing import Iterator, Optional
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PER_PAGE = 100
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds


def describe_exception(exc: BaseException) -> str:
    """
    Walk the __cause__ / __context__ chain and produce a single-line description.
    httpx wraps httpcore wraps ssl/OS errors, and the outer layers often have
    empty messages - the useful text is at the bottom of the chain.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).strip()
        cls = type(current).__name__
        parts.append(f"{cls}: {msg}" if msg else cls)
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


class WordPressClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        app_password: str,
        timeout: float = 60.0,
        user_agent: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        if not base_url:
            raise ValueError("WordPressClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        auth = (
            httpx.BasicAuth(username, app_password)
            if username and app_password
            else None
        )
        # Browser-like headers - some WAFs (Cloudflare, Wordfence) drop the
        # connection on non-browser User-Agents and reject responses to clients
        # that don't send the usual Accept-* set.
        # Note: 'br' (Brotli) intentionally omitted because httpx requires the
        # 'brotli' package to decode it, and the WP REST API will gladly serve
        # gzip/deflate instead - which httpx handles natively.
        headers = {
            "User-Agent": user_agent or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }
        self._client = httpx.Client(
            base_url=self.base_url,
            auth=auth,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            verify=verify_ssl,
        )
        # Separate auth-free client for downloading from non-WP hosts (e.g.,
        # CDNs in WORDPRESS_LINKED_PDF_HOSTS). Sending WP Basic Auth headers
        # to an external host would leak credentials.
        self._external_client = httpx.Client(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            verify=verify_ssl,
        )
        self._wp_host = (urlsplit(self.base_url).hostname or "").lower()
        self._log_dns()

    def _log_dns(self) -> None:
        """
        Resolve and log all A/AAAA records for the WP host. Useful when a
        load-balanced host has one unhealthy backend IP that keeps resetting
        connections (curl --resolve <ip> to confirm; or pin a healthy IP in
        the system hosts file as a workaround).
        """
        parsed = urlsplit(self.base_url)
        host = parsed.hostname
        if not host:
            return
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as e:
            logger.warning(f"DNS lookup failed for {host}: {describe_exception(e)}")
            return
        ips = sorted({ai[4][0] for ai in infos})
        logger.info(f"DNS {host} -> {', '.join(ips)} (port {port})")

    def close(self) -> None:
        self._client.close()
        self._external_client.close()

    def __enter__(self) -> "WordPressClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def ping(self) -> None:
        """
        Probe /wp-json/ and surface a clean diagnostic. Raises a RuntimeError
        with a descriptive message on any failure (DNS, TLS, 401, 404, 5xx, ...).
        """
        url = "/wp-json/"
        try:
            resp = self._client.get(url)
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach WordPress at {self.base_url}{url} - {describe_exception(e)}"
            ) from e

        if resp.status_code == 401:
            raise RuntimeError(
                f"WordPress {self.base_url}{url} returned 401 Unauthorized - "
                "check WORDPRESS_USERNAME / WORDPRESS_APP_PASSWORD "
                "(Application Password, not the WP login password)."
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"WordPress {self.base_url}{url} returned 404 - the REST API is not "
                "reachable here. Verify WORDPRESS_URL points at the site root and "
                "pretty permalinks (or ?rest_route=) are enabled."
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"WordPress {self.base_url}{url} returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )

        try:
            info = resp.json()
        except UnicodeDecodeError as e:
            # Almost certainly a content-encoding (Brotli) the client can't decode.
            ce = resp.headers.get("content-encoding", "<none>")
            raise RuntimeError(
                f"WordPress returned 200 but body could not be decoded "
                f"(content-encoding={ce!r}): {e}"
            ) from e
        except ValueError as e:
            ct = resp.headers.get("content-type", "<unknown>")
            raise RuntimeError(
                f"WordPress {self.base_url}{url} returned 200 but body is not JSON "
                f"(content-type={ct!r}, first 200 chars: {resp.text[:200]!r})"
            ) from e

        name = info.get("name", "<unknown>")
        namespaces = info.get("namespaces") or []
        logger.info(
            f"WP ping ok: site={name!r} namespaces={len(namespaces)} "
            f"({', '.join(namespaces[:5])}{'...' if len(namespaces) > 5 else ''})"
        )

    def list_content(
        self,
        route: str,
        modified_after: Optional[str] = None,
        extra_params: Optional[dict] = None,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> Iterator[dict]:
        """
        Yield items from a WP REST route, paginated and ordered by modified asc.

        `route` is the path under /wp-json - e.g. "wp/v2/posts", "wp/v2/pages",
        "wp/v2/lectures" (CPT), "wp/v2/media".
        """
        page = 1
        while True:
            params = {
                "per_page": per_page,
                "page": page,
                "orderby": "modified",
                "order": "asc",
                "status": "publish",
                "_embed": "false",
            }
            if modified_after:
                params["modified_after"] = modified_after
            if extra_params:
                params.update(extra_params)

            resp = self._get_with_retry(f"/wp-json/{route.lstrip('/')}", params)

            # WP returns 400/404 with structured JSON: { code, message, data: {...} }.
            # rest_post_invalid_page_number fires after the last page - clean stop.
            if 400 <= resp.status_code < 500:
                body = self._safe_json(resp)
                wp_code = (body or {}).get("code", "")
                wp_msg = (body or {}).get("message", "")
                if wp_code in ("rest_post_invalid_page_number", "rest_no_route"):
                    return
                raise httpx.HTTPStatusError(
                    f"WP {resp.status_code} on {resp.url}: "
                    f"code={wp_code!r} message={wp_msg!r}",
                    request=resp.request,
                    response=resp,
                )

            resp.raise_for_status()
            items = resp.json()
            if not items:
                return

            for item in items:
                yield item

            total_pages = int(resp.headers.get("X-WP-TotalPages", "1") or 1)
            if page >= total_pages:
                return
            page += 1

    def list_media_pdfs(
        self,
        modified_after: Optional[str] = None,
    ) -> Iterator[dict]:
        # WP REST `media_type` accepts: image|video|audio|text|application.
        # PDFs fall under "application"; we filter to application/pdf client-side
        # so the query stays valid across WP versions/plugins.
        # Attachments use post_status='inherit', not 'publish' - sending
        # status=publish to /wp/v2/media returns 400.
        for item in self.list_content(
            "wp/v2/media",
            modified_after=modified_after,
            extra_params={"media_type": "application", "status": "inherit"},
        ):
            if (item.get("mime_type") or "").lower() == "application/pdf":
                yield item

    def download_media(self, source_url: str) -> bytes:
        """
        Download a media file by absolute URL. Routes through the WP-auth
        client only if the URL is on the same host as base_url; otherwise
        uses the auth-free external client so WP credentials never leak to
        third-party hosts.
        """
        target_host = (urlsplit(source_url).hostname or "").lower()
        use_auth = bool(self._wp_host) and target_host == self._wp_host
        client = self._client if use_auth else self._external_client
        resp = self._get_with_retry(source_url, params=None, absolute=True, client=client)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Optional[dict]:
        """Best-effort JSON parse; returns None on any error."""
        try:
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _get_with_retry(
        self,
        url: str,
        params: Optional[dict],
        absolute: bool = False,
        client: Optional[httpx.Client] = None,
    ) -> httpx.Response:
        c = client if client is not None else self._client
        last_exc: Optional[Exception] = None
        resp: Optional[httpx.Response] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = c.get(url, params=params)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                resp = None
                logger.warning(
                    f"WP GET {url} failed [{describe_exception(e)}]; "
                    f"attempt {attempt}/{MAX_RETRIES}"
                )
            else:
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", BACKOFF_BASE))
                    logger.warning(f"WP GET {url} rate-limited; sleeping {retry_after}s")
                    time.sleep(retry_after)
                    continue
                if 500 <= resp.status_code < 600:
                    logger.warning(
                        f"WP GET {url} returned {resp.status_code}; attempt {attempt}/{MAX_RETRIES}"
                    )
                else:
                    return resp

            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

        if last_exc:
            raise last_exc
        # Fell through with last response being 5xx - return it; caller will raise_for_status.
        assert resp is not None
        return resp

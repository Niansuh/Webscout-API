import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from functools import cached_property
from itertools import cycle, islice
from threading import Event
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, Union, cast

import pyreqwest_impersonate as pri  # type: ignore

try:
    from lxml.etree import _Element
    from lxml.html import HTMLParser as LHTMLParser
    from lxml.html import document_fromstring

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

import re
from decimal import Decimal
from html import unescape
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Dict, List, Union
from urllib.parse import unquote
import orjson
import requests
import base64
from typing import List, Dict, Union
import json
import requests
import base64
from typing import List, Dict, Union


REGEX_STRIP_TAGS = re.compile("<.*?>")


def json_dumps(obj: Any) -> str:
    try:
        return orjson.dumps(obj).decode("utf-8")
    except Exception as ex:
        raise WebscoutE(f"{type(ex).__name__}: {ex}") from ex


def json_loads(obj: Union[str, bytes]) -> Any:
    try:
        return orjson.loads(obj)
    except Exception as ex:
        raise WebscoutE(f"{type(ex).__name__}: {ex}") from ex


def _extract_vqd(html_bytes: bytes, keywords: str) -> str:
    """Extract vqd from html bytes."""
    for c1, c1_len, c2 in (
        (b'vqd="', 5, b'"'),
        (b"vqd=", 4, b"&"),
        (b"vqd='", 5, b"'"),
    ):
        try:
            start = html_bytes.index(c1) + c1_len
            end = html_bytes.index(c2, start)
            return html_bytes[start:end].decode()
        except ValueError:
            pass
    raise WebscoutE(f"_extract_vqd() {keywords=} Could not extract vqd.")


def _text_extract_json(html_bytes: bytes, keywords: str) -> List[Dict[str, str]]:
    """text(backend="api") -> extract json from html."""
    try:
        start = html_bytes.index(b"DDG.pageLayout.load('d',") + 24
        end = html_bytes.index(b");DDG.duckbar.load(", start)
        data = html_bytes[start:end]
        result: List[Dict[str, str]] = json_loads(data)
        return result
    except Exception as ex:
        raise WebscoutE(f"_text_extract_json() {keywords=} {type(ex).__name__}: {ex}") from ex
    raise WebscoutE(f"_text_extract_json() {keywords=} return None")


def _normalize(raw_html: str) -> str:
    """Strip HTML tags from the raw_html string."""
    return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""


def _normalize_url(url: str) -> str:
    """Unquote URL and replace spaces with '+'."""
    return unquote(url.replace(" ", "+")) if url else ""


def _calculate_distance(lat1: Decimal, lon1: Decimal, lat2: Decimal, lon2: Decimal) -> float:
    """Calculate distance between two points in km. Haversine formula."""
    R = 6371.0087714  # Earth's radius in km
    rlat1, rlon1, rlat2, rlon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlon, dlat = rlon2 - rlon1, rlat2 - rlat1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

logger = logging.getLogger("webscout.WEBS")

class WebscoutE(Exception):
    """Base exception class for search."""


class RatelimitE(Exception):
    """Raised for rate limit exceeded errors during API requests."""


class TimeoutE(Exception):
    """Raised for timeout errors during API requests."""
    
class FailedToGenerateResponseError(Exception):
    
    """Provider failed to fetch response"""
class AllProvidersFailure(Exception):
    """None of the providers generated response successfully"""

    pass
class WEBS:
    """webscout class to get search results from duckduckgo.com."""

    _executor: ThreadPoolExecutor = ThreadPoolExecutor()

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
        proxies: Union[Dict[str, str], str, None] = None,  # deprecated
        timeout: Optional[int] = 10,
    ) -> None:
        """Initialize the WEBS object.

        Args:
            headers (dict, optional): Dictionary of headers for the HTTP client. Defaults to None.
            proxy (str, optional): proxy for the HTTP client, supports http/https/socks5 protocols.
                example: "http://user:pass@example.com:3128". Defaults to None.
            timeout (int, optional): Timeout value for the HTTP client. Defaults to 10.
        """
        self.proxy: Optional[str] = proxy
        assert self.proxy is None or isinstance(self.proxy, str), "proxy must be a str"
        if not proxy and proxies:
            warnings.warn("'proxies' is deprecated, use 'proxy' instead.", stacklevel=1)
            self.proxy = proxies.get("http") or proxies.get("https") if isinstance(proxies, dict) else proxies
        self.headers = headers if headers else {}
        self.headers["Referer"] = "https://duckduckgo.com/"
        self.client = pri.Client(
            headers=self.headers,
            proxy=self.proxy,
            timeout=timeout,
            cookie_store=True,
            referer=True,
            impersonate="chrome_124",
            follow_redirects=False,
            verify=False,
        )
        self._exception_event = Event()
        self._chat_messages: List[Dict[str, str]] = []
        self._chat_vqd: str = ""

    def __enter__(self) -> "WEBS":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_val: Optional[BaseException] = None,
        exc_tb: Optional[TracebackType] = None,
    ) -> None:
        pass

    @cached_property
    def parser(self) -> "LHTMLParser":
        """Get HTML parser."""
        return LHTMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True, collect_ids=False)

    def _get_url(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, str]] = None,
        content: Optional[bytes] = None,
        data: Optional[Union[Dict[str, str], bytes]] = None,
    ) -> bytes:
        if self._exception_event.is_set():
            raise WebscoutE("Exception occurred in previous call.")
        try:
            resp = self.client.request(method, url, params=params, content=content, data=data)
        except Exception as ex:
            self._exception_event.set()
            if "time" in str(ex).lower():
                raise TimeoutE(f"{url} {type(ex).__name__}: {ex}") from ex
            raise WebscoutE(f"{url} {type(ex).__name__}: {ex}") from ex
        logger.debug(f"_get_url() {resp.url} {resp.status_code} {len(resp.content)}")
        if resp.status_code == 200:
            return cast(bytes, resp.content)
        self._exception_event.set()
        if resp.status_code in (202, 301, 403):
            raise RatelimitE(f"{resp.url} {resp.status_code} Ratelimit")
        raise WebscoutE(f"{resp.url} return None. {params=} {content=} {data=}")

    def _get_vqd(self, keywords: str) -> str:
        """Get vqd value for a search query."""
        resp_content = self._get_url("POST", "https://duckduckgo.com", data={"q": keywords})
        return _extract_vqd(resp_content, keywords)

    def chat(self, keywords: str, model: str = "gpt-3.5") -> str:
        """Initiates a chat session with DuckDuckGo AI.

        Args:
            keywords (str): The initial message or question to send to the AI.
            model (str): The model to use: "gpt-3.5", "claude-3-haiku", "llama-3-70b", "mixtral-8x7b".
                Defaults to "gpt-3.5".

        Returns:
            str: The response from the AI.
        """
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "gpt-3.5": "gpt-3.5-turbo-0125",
            "llama-3-70b": "meta-llama/Llama-3-70b-chat-hf",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
        # vqd
        if not self._chat_vqd:
            resp = self.client.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"})
            self._chat_vqd = resp.headers.get("x-vqd-4", "")

        self._chat_messages.append({"role": "user", "content": keywords})

        json_data = {
            "model": models[model],
            "messages": self._chat_messages,
        }
        resp = self.client.post(
            "https://duckduckgo.com/duckchat/v1/chat", headers={"x-vqd-4": self._chat_vqd}, json=json_data
        )
        self._chat_vqd = resp.headers.get("x-vqd-4", "")

        messages = []
        for line in resp.text.replace("data: ", "").replace("[DONE]", "").split("\n\n"):
            x = line.strip()
            if x:
                j = json_loads(x)
                message = j.get("message", "")
                messages.append(message)
        result = "".join(messages)
        self._chat_messages.append({"role": "assistant", "content": result})
        return result

    def text(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        backend: str = "api",
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo text search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m, y. Defaults to None.
            backend: api, html, lite. Defaults to api.
                api - collect data from https://duckduckgo.com,
                html - collect data from https://html.duckduckgo.com,
                lite - collect data from https://lite.duckduckgo.com.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with search results, or None if there was an error.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        if LXML_AVAILABLE is False and backend != "api":
            backend = "api"
            warnings.warn("lxml is not installed. Using backend='api'.", stacklevel=2)

        if backend == "api":
            results = self._text_api(keywords, region, safesearch, timelimit, max_results)
        elif backend == "html":
            results = self._text_html(keywords, region, timelimit, max_results)
        elif backend == "lite":
            results = self._text_lite(keywords, region, timelimit, max_results)
        return results

    def _text_api(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo text search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m, y. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with search results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        vqd = self._get_vqd(keywords)

        payload = {
            "q": keywords,
            "kl": region,
            "l": region,
            "p": "",
            "s": "0",
            "df": "",
            "vqd": vqd,
            "bing_market": f"{region[3:]}-{region[:2].upper()}",
            "ex": "",
        }
        safesearch = safesearch.lower()
        if safesearch == "moderate":
            payload["ex"] = "-1"
        elif safesearch == "off":
            payload["ex"] = "-2"
        elif safesearch == "on":  # strict
            payload["p"] = "1"
        if timelimit:
            payload["df"] = timelimit

        cache = set()
        results: List[Dict[str, str]] = []

        def _text_api_page(s: int) -> List[Dict[str, str]]:
            payload["s"] = f"{s}"
            resp_content = self._get_url("GET", "https://links.duckduckgo.com/d.js", params=payload)
            page_data = _text_extract_json(resp_content, keywords)
            page_results = []
            for row in page_data:
                href = row.get("u", None)
                if href and href not in cache and href != f"http://www.google.com/search?q={keywords}":
                    cache.add(href)
                    body = _normalize(row["a"])
                    if body:
                        result = {
                            "title": _normalize(row["t"]),
                            "href": _normalize_url(href),
                            "body": body,
                        }
                        page_results.append(result)
            return page_results

        slist = [0]
        if max_results:
            max_results = min(max_results, 2023)
            slist.extend(range(23, max_results, 50))
        try:
            for r in self._executor.map(_text_api_page, slist):
                results.extend(r)
        except Exception as e:
            raise e

        return list(islice(results, max_results))

    def _text_html(
        self,
        keywords: str,
        region: str = "wt-wt",
        timelimit: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo text search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            timelimit: d, w, m, y. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with search results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        payload = {
            "q": keywords,
            "s": "0",
            "o": "json",
            "api": "d.js",
            "vqd": "",
            "kl": region,
            "bing_market": region,
        }
        if timelimit:
            payload["df"] = timelimit
        if max_results and max_results > 20:
            vqd = self._get_vqd(keywords)
            payload["vqd"] = vqd

        cache = set()
        results: List[Dict[str, str]] = []

        def _text_html_page(s: int) -> List[Dict[str, str]]:
            payload["s"] = f"{s}"
            resp_content = self._get_url("POST", "https://html.duckduckgo.com/html", data=payload)
            if b"No  results." in resp_content:
                return []

            page_results = []
            tree = document_fromstring(resp_content, self.parser)
            elements = tree.xpath("//div[h2]")
            if not isinstance(elements, List):
                return []
            for e in elements:
                if isinstance(e, _Element):
                    hrefxpath = e.xpath("./a/@href")
                    href = str(hrefxpath[0]) if isinstance(hrefxpath, List) else None
                    if (
                        href
                        and href not in cache
                        and not href.startswith(
                            ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                        )
                    ):
                        cache.add(href)
                        titlexpath = e.xpath("./h2/a/text()")
                        title = str(titlexpath[0]) if isinstance(titlexpath, List) else ""
                        bodyxpath = e.xpath("./a//text()")
                        body = "".join(str(x) for x in bodyxpath) if isinstance(bodyxpath, List) else ""
                        result = {
                            "title": _normalize(title),
                            "href": _normalize_url(href),
                            "body": _normalize(body),
                        }
                        page_results.append(result)
            return page_results

        slist = [0]
        if max_results:
            max_results = min(max_results, 2023)
            slist.extend(range(23, max_results, 50))
        try:
            for r in self._executor.map(_text_html_page, slist):
                results.extend(r)
        except Exception as e:
            raise e

        return list(islice(results, max_results))

    def _text_lite(
        self,
        keywords: str,
        region: str = "wt-wt",
        timelimit: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo text search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            timelimit: d, w, m, y. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with search results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        payload = {
            "q": keywords,
            "s": "0",
            "o": "json",
            "api": "d.js",
            "vqd": "",
            "kl": region,
            "bing_market": region,
        }
        if timelimit:
            payload["df"] = timelimit

        cache = set()
        results: List[Dict[str, str]] = []

        def _text_lite_page(s: int) -> List[Dict[str, str]]:
            payload["s"] = f"{s}"
            resp_content = self._get_url("POST", "https://lite.duckduckgo.com/lite/", data=payload)
            if b"No more results." in resp_content:
                return []

            page_results = []
            tree = document_fromstring(resp_content, self.parser)
            elements = tree.xpath("//table[last()]//tr")
            if not isinstance(elements, List):
                return []

            data = zip(cycle(range(1, 5)), elements)
            for i, e in data:
                if isinstance(e, _Element):
                    if i == 1:
                        hrefxpath = e.xpath(".//a//@href")
                        href = str(hrefxpath[0]) if hrefxpath and isinstance(hrefxpath, List) else None
                        if (
                            href is None
                            or href in cache
                            or href.startswith(
                                ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                            )
                        ):
                            [next(data, None) for _ in range(3)]  # skip block(i=1,2,3,4)
                        else:
                            cache.add(href)
                            titlexpath = e.xpath(".//a//text()")
                            title = str(titlexpath[0]) if isinstance(titlexpath, List) else ""
                    elif i == 2:
                        bodyxpath = e.xpath(".//td[@class='result-snippet']//text()")
                        body = "".join(str(x) for x in bodyxpath) if isinstance(bodyxpath, List) else ""
                        if href:
                            result = {
                                "title": _normalize(title),
                                "href": _normalize_url(href),
                                "body": _normalize(body),
                            }
                            page_results.append(result)
            return page_results

        slist = [0]
        if max_results:
            max_results = min(max_results, 2023)
            slist.extend(range(23, max_results, 50))
        try:
            for r in self._executor.map(_text_lite_page, slist):
                results.extend(r)
        except Exception as e:
            raise e

        return list(islice(results, max_results))

    def images(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        size: Optional[str] = None,
        color: Optional[str] = None,
        type_image: Optional[str] = None,
        layout: Optional[str] = None,
        license_image: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo images search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: Day, Week, Month, Year. Defaults to None.
            size: Small, Medium, Large, Wallpaper. Defaults to None.
            color: color, Monochrome, Red, Orange, Yellow, Green, Blue,
                Purple, Pink, Brown, Black, Gray, Teal, White. Defaults to None.
            type_image: photo, clipart, gif, transparent, line.
                Defaults to None.
            layout: Square, Tall, Wide. Defaults to None.
            license_image: any (All Creative Commons), Public (PublicDomain),
                Share (Free to Share and Use), ShareCommercially (Free to Share and Use Commercially),
                Modify (Free to Modify, Share, and Use), ModifyCommercially (Free to Modify, Share, and
                Use Commercially). Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with images search results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        vqd = self._get_vqd(keywords)

        safesearch_base = {"on": "1", "moderate": "1", "off": "-1"}
        timelimit = f"time:{timelimit}" if timelimit else ""
        size = f"size:{size}" if size else ""
        color = f"color:{color}" if color else ""
        type_image = f"type:{type_image}" if type_image else ""
        layout = f"layout:{layout}" if layout else ""
        license_image = f"license:{license_image}" if license_image else ""
        payload = {
            "l": region,
            "o": "json",
            "q": keywords,
            "vqd": vqd,
            "f": f"{timelimit},{size},{color},{type_image},{layout},{license_image}",
            "p": safesearch_base[safesearch.lower()],
        }

        cache = set()
        results: List[Dict[str, str]] = []

        def _images_page(s: int) -> List[Dict[str, str]]:
            payload["s"] = f"{s}"
            resp_content = self._get_url("GET", "https://duckduckgo.com/i.js", params=payload)
            resp_json = json_loads(resp_content)

            page_data = resp_json.get("results", [])
            page_results = []
            for row in page_data:
                image_url = row.get("image")
                if image_url and image_url not in cache:
                    cache.add(image_url)
                    result = {
                        "title": row["title"],
                        "image": _normalize_url(image_url),
                        "thumbnail": _normalize_url(row["thumbnail"]),
                        "url": _normalize_url(row["url"]),
                        "height": row["height"],
                        "width": row["width"],
                        "source": row["source"],
                    }
                    page_results.append(result)
            return page_results

        slist = [0]
        if max_results:
            max_results = min(max_results, 500)
            slist.extend(range(100, max_results, 100))
        try:
            for r in self._executor.map(_images_page, slist):
                results.extend(r)
        except Exception as e:
            raise e

        return list(islice(results, max_results))

    def videos(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        resolution: Optional[str] = None,
        duration: Optional[str] = None,
        license_videos: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo videos search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m. Defaults to None.
            resolution: high, standart. Defaults to None.
            duration: short, medium, long. Defaults to None.
            license_videos: creativeCommon, youtube. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with videos search results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        vqd = self._get_vqd(keywords)

        safesearch_base = {"on": "1", "moderate": "-1", "off": "-2"}
        timelimit = f"publishedAfter:{timelimit}" if timelimit else ""
        resolution = f"videoDefinition:{resolution}" if resolution else ""
        duration = f"videoDuration:{duration}" if duration else ""
        license_videos = f"videoLicense:{license_videos}" if license_videos else ""
        payload = {
            "l": region,
            "o": "json",
            "q": keywords,
            "vqd": vqd,
            "f": f"{timelimit},{resolution},{duration},{license_videos}",
            "p": safesearch_base[safesearch.lower()],
        }

        cache = set()
        results: List[Dict[str, str]] = []

        def _videos_page(s: int) -> List[Dict[str, str]]:
            payload["s"] = f"{s}"
            resp_content = self._get_url("GET", "https://duckduckgo.com/v.js", params=payload)
            resp_json = json_loads(resp_content)

            page_data = resp_json.get("results", [])
            page_results = []
            for row in page_data:
                if row["content"] not in cache:
                    cache.add(row["content"])
                    page_results.append(row)
            return page_results

        slist = [0]
        if max_results:
            max_results = min(max_results, 400)
            slist.extend(range(60, max_results, 60))
        try:
            for r in self._executor.map(_videos_page, slist):
                results.extend(r)
        except Exception as e:
            raise e

        return list(islice(results, max_results))

    def news(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo news search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with news search results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        vqd = self._get_vqd(keywords)

        safesearch_base = {"on": "1", "moderate": "-1", "off": "-2"}
        payload = {
            "l": region,
            "o": "json",
            "noamp": "1",
            "q": keywords,
            "vqd": vqd,
            "p": safesearch_base[safesearch.lower()],
        }
        if timelimit:
            payload["df"] = timelimit

        cache = set()
        results: List[Dict[str, str]] = []

        def _news_page(s: int) -> List[Dict[str, str]]:
            payload["s"] = f"{s}"
            resp_content = self._get_url("GET", "https://duckduckgo.com/news.js", params=payload)
            resp_json = json_loads(resp_content)
            page_data = resp_json.get("results", [])
            page_results = []
            for row in page_data:
                if row["url"] not in cache:
                    cache.add(row["url"])
                    image_url = row.get("image", None)
                    result = {
                        "date": datetime.fromtimestamp(row["date"], timezone.utc).isoformat(),
                        "title": row["title"],
                        "body": _normalize(row["excerpt"]),
                        "url": _normalize_url(row["url"]),
                        "image": _normalize_url(image_url),
                        "source": row["source"],
                    }
                    page_results.append(result)
            return page_results

        slist = [0]
        if max_results:
            max_results = min(max_results, 120)
            slist.extend(range(30, max_results, 30))
        try:
            for r in self._executor.map(_news_page, slist):
                results.extend(r)
        except Exception as e:
            raise e

        return list(islice(results, max_results))

    def answers(self, keywords: str) -> List[Dict[str, str]]:
        """DuckDuckGo instant answers. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query,

        Returns:
            List of dictionaries with instant answers results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        payload = {
            "q": f"what is {keywords}",
            "format": "json",
        }
        resp_content = self._get_url("GET", "https://api.duckduckgo.com/", params=payload)
        page_data = json_loads(resp_content)

        results = []
        answer = page_data.get("AbstractText")
        url = page_data.get("AbstractURL")
        if answer:
            results.append(
                {
                    "icon": None,
                    "text": answer,
                    "topic": None,
                    "url": url,
                }
            )

        # related
        payload = {
            "q": f"{keywords}",
            "format": "json",
        }
        resp_content = self._get_url("GET", "https://api.duckduckgo.com/", params=payload)
        resp_json = json_loads(resp_content)
        page_data = resp_json.get("RelatedTopics", [])

        for row in page_data:
            topic = row.get("Name")
            if not topic:
                icon = row["Icon"].get("URL")
                results.append(
                    {
                        "icon": f"https://duckduckgo.com{icon}" if icon else "",
                        "text": row["Text"],
                        "topic": None,
                        "url": row["FirstURL"],
                    }
                )
            else:
                for subrow in row["Topics"]:
                    icon = subrow["Icon"].get("URL")
                    results.append(
                        {
                            "icon": f"https://duckduckgo.com{icon}" if icon else "",
                            "text": subrow["Text"],
                            "topic": topic,
                            "url": subrow["FirstURL"],
                        }
                    )

        return results

    def suggestions(self, keywords: str, region: str = "wt-wt") -> List[Dict[str, str]]:
        """DuckDuckGo suggestions. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".

        Returns:
            List of dictionaries with suggestions results.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        payload = {
            "q": keywords,
            "kl": region,
        }
        resp_content = self._get_url("GET", "https://duckduckgo.com/ac/", params=payload)
        page_data = json_loads(resp_content)
        return [r for r in page_data]

    def maps(
        self,
        keywords: str,
        place: Optional[str] = None,
        street: Optional[str] = None,
        city: Optional[str] = None,
        county: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        postalcode: Optional[str] = None,
        latitude: Optional[str] = None,
        longitude: Optional[str] = None,
        radius: int = 0,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo maps search. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query
            place: if set, the other parameters are not used. Defaults to None.
            street: house number/street. Defaults to None.
            city: city of search. Defaults to None.
            county: county of search. Defaults to None.
            state: state of search. Defaults to None.
            country: country of search. Defaults to None.
            postalcode: postalcode of search. Defaults to None.
            latitude: geographic coordinate (north-south position). Defaults to None.
            longitude: geographic coordinate (east-west position); if latitude and
                longitude are set, the other parameters are not used. Defaults to None.
            radius: expand the search square by the distance in kilometers. Defaults to 0.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with maps search results, or None if there was an error.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        vqd = self._get_vqd(keywords)

        # if longitude and latitude are specified, skip the request about bbox to the nominatim api
        if latitude and longitude:
            lat_t = Decimal(latitude.replace(",", "."))
            lat_b = Decimal(latitude.replace(",", "."))
            lon_l = Decimal(longitude.replace(",", "."))
            lon_r = Decimal(longitude.replace(",", "."))
            if radius == 0:
                radius = 1
        # otherwise request about bbox to nominatim api
        else:
            if place:
                params = {
                    "q": place,
                    "polygon_geojson": "0",
                    "format": "jsonv2",
                }
            else:
                params = {
                    "polygon_geojson": "0",
                    "format": "jsonv2",
                }
                if street:
                    params["street"] = street
                if city:
                    params["city"] = city
                if county:
                    params["county"] = county
                if state:
                    params["state"] = state
                if country:
                    params["country"] = country
                if postalcode:
                    params["postalcode"] = postalcode
            # request nominatim api to get coordinates box
            resp_content = self._get_url(
                "GET",
                "https://nominatim.openstreetmap.org/search.php",
                params=params,
            )
            if resp_content == b"[]":
                raise WebscoutE("maps() Coordinates are not found, check function parameters.")
            resp_json = json_loads(resp_content)
            coordinates = resp_json[0]["boundingbox"]
            lat_t, lon_l = Decimal(coordinates[1]), Decimal(coordinates[2])
            lat_b, lon_r = Decimal(coordinates[0]), Decimal(coordinates[3])

        # if a radius is specified, expand the search square
        lat_t += Decimal(radius) * Decimal(0.008983)
        lat_b -= Decimal(radius) * Decimal(0.008983)
        lon_l -= Decimal(radius) * Decimal(0.008983)
        lon_r += Decimal(radius) * Decimal(0.008983)
        logger.debug(f"bbox coordinates\n{lat_t} {lon_l}\n{lat_b} {lon_r}")

        cache = set()
        results: List[Dict[str, str]] = []

        def _maps_page(
            bbox: Tuple[Decimal, Decimal, Decimal, Decimal],
        ) -> Optional[List[Dict[str, str]]]:
            if max_results and len(results) >= max_results:
                return None
            lat_t, lon_l, lat_b, lon_r = bbox
            params = {
                "q": keywords,
                "vqd": vqd,
                "tg": "maps_places",
                "rt": "D",
                "mkexp": "b",
                "wiki_info": "1",
                "is_requery": "1",
                "bbox_tl": f"{lat_t},{lon_l}",
                "bbox_br": f"{lat_b},{lon_r}",
                "strict_bbox": "1",
            }
            resp_content = self._get_url("GET", "https://duckduckgo.com/local.js", params=params)
            resp_json = json_loads(resp_content)
            page_data = resp_json.get("results", [])

            page_results = []
            for res in page_data:
                r_name = f'{res["name"]} {res["address"]}'
                if r_name in cache:
                    continue
                else:
                    cache.add(r_name)
                    result = {
                        "title": res["name"],
                        "address": res["address"],
                        "country_code": res["country_code"],
                        "url": _normalize_url(res["website"]),
                        "phone": res["phone"] or "",
                        "latitude": res["coordinates"]["latitude"],
                        "longitude": res["coordinates"]["longitude"],
                        "source": _normalize_url(res["url"]),
                        "image": x.get("image", "") if (x := res["embed"]) else "",
                        "desc": x.get("description", "") if (x := res["embed"]) else "",
                        "hours": res["hours"] or "",
                        "category": res["ddg_category"] or "",
                        "facebook": f"www.facebook.com/profile.php?id={x}" if (x := res["facebook_id"]) else "",
                        "instagram": f"https://www.instagram.com/{x}" if (x := res["instagram_id"]) else "",
                        "twitter": f"https://twitter.com/{x}" if (x := res["twitter_id"]) else "",
                    }
                    page_results.append(result)
            return page_results

        # search squares (bboxes)
        start_bbox = (lat_t, lon_l, lat_b, lon_r)
        work_bboxes = [start_bbox]
        while work_bboxes:
            queue_bboxes = []  # for next iteration, at the end of the iteration work_bboxes = queue_bboxes
            tasks = []
            for bbox in work_bboxes:
                tasks.append(bbox)
                # if distance between coordinates > 1, divide the square into 4 parts and save them in queue_bboxes
                if _calculate_distance(lat_t, lon_l, lat_b, lon_r) > 1:
                    lat_t, lon_l, lat_b, lon_r = bbox
                    lat_middle = (lat_t + lat_b) / 2
                    lon_middle = (lon_l + lon_r) / 2
                    bbox1 = (lat_t, lon_l, lat_middle, lon_middle)
                    bbox2 = (lat_t, lon_middle, lat_middle, lon_r)
                    bbox3 = (lat_middle, lon_l, lat_b, lon_middle)
                    bbox4 = (lat_middle, lon_middle, lat_b, lon_r)
                    queue_bboxes.extend([bbox1, bbox2, bbox3, bbox4])

            # gather tasks using asyncio.wait_for and timeout
            work_bboxes_results = []
            try:
                for r in self._executor.map(_maps_page, tasks):
                    if r:
                        work_bboxes_results.extend(r)
            except Exception as e:
                raise e

            for x in work_bboxes_results:
                if isinstance(x, list):
                    results.extend(x)
                elif isinstance(x, dict):
                    results.append(x)

            work_bboxes = queue_bboxes
            if not max_results or len(results) >= max_results or len(work_bboxes_results) == 0:
                break

        return list(islice(results, max_results))

    def translate(
        self, keywords: Union[List[str], str], from_: Optional[str] = None, to: str = "en"
    ) -> List[Dict[str, str]]:
        """DuckDuckGo translate.

        Args:
            keywords: string or list of strings to translate.
            from_: translate from (defaults automatically). Defaults to None.
            to: what language to translate. Defaults to "en".

        Returns:
            List od dictionaries with translated keywords.

        Raises:
            WebscoutE: Base exception for webscout errors.
            RatelimitE: Inherits from WebscoutE, raised for exceeding API request rate limits.
            TimeoutE: Inherits from WebscoutE, raised for API request timeouts.
        """
        assert keywords, "keywords is mandatory"

        vqd = self._get_vqd("translate")

        payload = {
            "vqd": vqd,
            "query": "translate",
            "to": to,
        }
        if from_:
            payload["from"] = from_

        def _translate_keyword(keyword: str) -> Dict[str, str]:
            resp_content = self._get_url(
                "POST",
                "https://duckduckgo.com/translation.js",
                params=payload,
                content=keyword.encode(),
            )
            page_data: Dict[str, str] = json_loads(resp_content)
            page_data["original"] = keyword
            return page_data

        if isinstance(keywords, str):
            keywords = [keywords]

        results = []
        try:
            for r in self._executor.map(_translate_keyword, keywords):
                results.append(r)
        except Exception as e:
            raise e

        return results
import requests
import http.cookiejar as cookiejar
import sys
import json
from xml.etree import ElementTree
import re
from requests import HTTPError
import html.parser

html_parser = html.parser.HTMLParser()
import html

def unescape(string):
    return html.unescape(string)
WATCH_URL = 'https://www.youtube.com/watch?v={video_id}'

class TranscriptRetrievalError(Exception):
    """
    Base class for exceptions raised when a transcript cannot be retrieved.
    """
    ERROR_MESSAGE = '\nCould not retrieve a transcript for the video {video_url}!'
    CAUSE_MESSAGE_INTRO = ' This is most likely caused by:\n\n{cause}'
    CAUSE_MESSAGE = ''
    GITHUB_REFERRAL = (
        '\n\nIf you are sure that the described cause is not responsible for this error '
        'and that a transcript should be retrievable, please create an issue at '
        'https://github.com/OE-LUCIFER/Webscout/issues. '
        'Please add which version of webscout you are using '
        'and provide the information needed to replicate the error. '
    )

    def __init__(self, video_id):
        self.video_id = video_id
        super(TranscriptRetrievalError, self).__init__(self._build_error_message())

    def _build_error_message(self):
        cause = self.cause
        error_message = self.ERROR_MESSAGE.format(video_url=WATCH_URL.format(video_id=self.video_id))

        if cause:
            error_message += self.CAUSE_MESSAGE_INTRO.format(cause=cause) + self.GITHUB_REFERRAL

        return error_message

    @property
    def cause(self):
        return self.CAUSE_MESSAGE

class YouTubeRequestFailedError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'Request to YouTube failed: {reason}'

    def __init__(self, video_id, http_error):
        self.reason = str(http_error)
        super(YouTubeRequestFailedError, self).__init__(video_id)

    @property
    def cause(self):
        return self.CAUSE_MESSAGE.format(reason=self.reason)

class VideoUnavailableError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'The video is no longer available'

class InvalidVideoIdError(TranscriptRetrievalError):
    CAUSE_MESSAGE = (
        'You provided an invalid video id. Make sure you are using the video id and NOT the url!\n\n'
        'Do NOT run: `YouTubeTranscriptApi.get_transcript("https://www.youtube.com/watch?v=1234")`\n'
        'Instead run: `YouTubeTranscriptApi.get_transcript("1234")`'
    )

class TooManyRequestsError(TranscriptRetrievalError):
    CAUSE_MESSAGE = (
        'YouTube is receiving too many requests from this IP and now requires solving a captcha to continue. '
        'One of the following things can be done to work around this:\n\
        - Manually solve the captcha in a browser and export the cookie. '
        'Read here how to use that cookie with '
        'youtube-transcript-api: https://github.com/jdepoix/youtube-transcript-api#cookies\n\
        - Use a different IP address\n\
        - Wait until the ban on your IP has been lifted'
    )

class TranscriptsDisabledError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'Subtitles are disabled for this video'

class NoTranscriptAvailableError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'No transcripts are available for this video'

class NotTranslatableError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'The requested language is not translatable'

class TranslationLanguageNotAvailableError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'The requested translation language is not available'

class CookiePathInvalidError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'The provided cookie file was unable to be loaded'

class CookiesInvalidError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'The cookies provided are not valid (may have expired)'

class FailedToCreateConsentCookieError(TranscriptRetrievalError):
    CAUSE_MESSAGE = 'Failed to automatically give consent to saving cookies'

class NoTranscriptFoundError(TranscriptRetrievalError):
    CAUSE_MESSAGE = (
        'No transcripts were found for any of the requested language codes: {requested_language_codes}\n\n'
        '{transcript_data}'
    )

    def __init__(self, video_id, requested_language_codes, transcript_data):
        self._requested_language_codes = requested_language_codes
        self._transcript_data = transcript_data
        super(NoTranscriptFoundError, self).__init__(video_id)

    @property
    def cause(self):
        return self.CAUSE_MESSAGE.format(
            requested_language_codes=self._requested_language_codes,
            transcript_data=str(self._transcript_data),
        )



def _raise_http_errors(response, video_id):
    try:
        response.raise_for_status()
        return response
    except HTTPError as error:
        raise YouTubeRequestFailedError(error, video_id)


class TranscriptListFetcher(object):
    def __init__(self, http_client):
        self._http_client = http_client

    def fetch(self, video_id):
        return TranscriptList.build(
            self._http_client,
            video_id,
            self._extract_captions_json(self._fetch_video_html(video_id), video_id),
        )

    def _extract_captions_json(self, html, video_id):
        splitted_html = html.split('"captions":')

        if len(splitted_html) <= 1:
            if video_id.startswith('http://') or video_id.startswith('https://'):
                raise InvalidVideoIdError(video_id)
            if 'class="g-recaptcha"' in html:
                raise TooManyRequestsError(video_id)
            if '"playabilityStatus":' not in html:
                raise VideoUnavailableError(video_id)

            raise TranscriptsDisabledError(video_id)

        captions_json = json.loads(
            splitted_html[1].split(',"videoDetails')[0].replace('\n', '')
        ).get('playerCaptionsTracklistRenderer')
        if captions_json is None:
            raise TranscriptsDisabledError(video_id)

        if 'captionTracks' not in captions_json:
            raise TranscriptsDisabledError(video_id)

        return captions_json

    def _create_consent_cookie(self, html, video_id):
        match = re.search('name="v" value="(.*?)"', html)
        if match is None:
            raise FailedToCreateConsentCookieError(video_id)
        self._http_client.cookies.set('CONSENT', 'YES+' + match.group(1), domain='.youtube.com')

    def _fetch_video_html(self, video_id):
        html = self._fetch_html(video_id)
        if 'action="https://consent.youtube.com/s"' in html:
            self._create_consent_cookie(html, video_id)
            html = self._fetch_html(video_id)
            if 'action="https://consent.youtube.com/s"' in html:
                raise FailedToCreateConsentCookieError(video_id)
        return html

    def _fetch_html(self, video_id):
        response = self._http_client.get(WATCH_URL.format(video_id=video_id), headers={'Accept-Language': 'en-US'})
        return unescape(_raise_http_errors(response, video_id).text)


class TranscriptList(object):
    """
    This object represents a list of transcripts. It can be iterated over to list all transcripts which are available
    for a given YouTube video. Also it provides functionality to search for a transcript in a given language.
    """

    def __init__(self, video_id, manually_created_transcripts, generated_transcripts, translation_languages):
        """
        The constructor is only for internal use. Use the static build method instead.

        :param video_id: the id of the video this TranscriptList is for
        :type video_id: str
        :param manually_created_transcripts: dict mapping language codes to the manually created transcripts
        :type manually_created_transcripts: dict[str, Transcript]
        :param generated_transcripts: dict mapping language codes to the generated transcripts
        :type generated_transcripts: dict[str, Transcript]
        :param translation_languages: list of languages which can be used for translatable languages
        :type translation_languages: list[dict[str, str]]
        """
        self.video_id = video_id
        self._manually_created_transcripts = manually_created_transcripts
        self._generated_transcripts = generated_transcripts
        self._translation_languages = translation_languages

    @staticmethod
    def build(http_client, video_id, captions_json):
        """
        Factory method for TranscriptList.

        :param http_client: http client which is used to make the transcript retrieving http calls
        :type http_client: requests.Session
        :param video_id: the id of the video this TranscriptList is for
        :type video_id: str
        :param captions_json: the JSON parsed from the YouTube pages static HTML
        :type captions_json: dict
        :return: the created TranscriptList
        :rtype TranscriptList:
        """
        translation_languages = [
            {
                'language': translation_language['languageName']['simpleText'],
                'language_code': translation_language['languageCode'],
            } for translation_language in captions_json.get('translationLanguages', [])
        ]

        manually_created_transcripts = {}
        generated_transcripts = {}

        for caption in captions_json['captionTracks']:
            if caption.get('kind', '') == 'asr':
                transcript_dict = generated_transcripts
            else:
                transcript_dict = manually_created_transcripts

            transcript_dict[caption['languageCode']] = Transcript(
                http_client,
                video_id,
                caption['baseUrl'],
                caption['name']['simpleText'],
                caption['languageCode'],
                caption.get('kind', '') == 'asr',
                translation_languages if caption.get('isTranslatable', False) else [],
            )

        return TranscriptList(
            video_id,
            manually_created_transcripts,
            generated_transcripts,
            translation_languages,
        )

    def __iter__(self):
        return iter(list(self._manually_created_transcripts.values()) + list(self._generated_transcripts.values()))

    def find_transcript(self, language_codes):
        """
        Finds a transcript for a given language code. Manually created transcripts are returned first and only if none
        are found, generated transcripts are used. If you only want generated transcripts use
        `find_manually_created_transcript` instead.

        :param language_codes: A list of language codes in a descending priority. For example, if this is set to
        ['de', 'en'] it will first try to fetch the german transcript (de) and then fetch the english transcript (en) if
        it fails to do so.
        :type languages: list[str]
        :return: the found Transcript
        :rtype Transcript:
        :raises: NoTranscriptFound
        """
        return self._find_transcript(language_codes, [self._manually_created_transcripts, self._generated_transcripts])

    def find_generated_transcript(self, language_codes):
        """
        Finds an automatically generated transcript for a given language code.

        :param language_codes: A list of language codes in a descending priority. For example, if this is set to
        ['de', 'en'] it will first try to fetch the german transcript (de) and then fetch the english transcript (en) if
        it fails to do so.
        :type languages: list[str]
        :return: the found Transcript
        :rtype Transcript:
        :raises: NoTranscriptFound
        """
        return self._find_transcript(language_codes, [self._generated_transcripts])

    def find_manually_created_transcript(self, language_codes):
        """
        Finds a manually created transcript for a given language code.

        :param language_codes: A list of language codes in a descending priority. For example, if this is set to
        ['de', 'en'] it will first try to fetch the german transcript (de) and then fetch the english transcript (en) if
        it fails to do so.
        :type languages: list[str]
        :return: the found Transcript
        :rtype Transcript:
        :raises: NoTranscriptFound
        """
        return self._find_transcript(language_codes, [self._manually_created_transcripts])

    def _find_transcript(self, language_codes, transcript_dicts):
        for language_code in language_codes:
            for transcript_dict in transcript_dicts:
                if language_code in transcript_dict:
                    return transcript_dict[language_code]

        raise NoTranscriptFoundError(
            self.video_id,
            language_codes,
            self
        )

    def __str__(self):
        return (
            'For this video ({video_id}) transcripts are available in the following languages:\n\n'
            '(MANUALLY CREATED)\n'
            '{available_manually_created_transcript_languages}\n\n'
            '(GENERATED)\n'
            '{available_generated_transcripts}\n\n'
            '(TRANSLATION LANGUAGES)\n'
            '{available_translation_languages}'
        ).format(
            video_id=self.video_id,
            available_manually_created_transcript_languages=self._get_language_description(
                str(transcript) for transcript in self._manually_created_transcripts.values()
            ),
            available_generated_transcripts=self._get_language_description(
                str(transcript) for transcript in self._generated_transcripts.values()
            ),
            available_translation_languages=self._get_language_description(
                '{language_code} ("{language}")'.format(
                    language=translation_language['language'],
                    language_code=translation_language['language_code'],
                ) for translation_language in self._translation_languages
            )
        )

    def _get_language_description(self, transcript_strings):
        description = '\n'.join(' - {transcript}'.format(transcript=transcript) for transcript in transcript_strings)
        return description if description else 'None'


class Transcript(object):
    def __init__(self, http_client, video_id, url, language, language_code, is_generated, translation_languages):
        """
        You probably don't want to initialize this directly. Usually you'll access Transcript objects using a
        TranscriptList.

        :param http_client: http client which is used to make the transcript retrieving http calls
        :type http_client: requests.Session
        :param video_id: the id of the video this TranscriptList is for
        :type video_id: str
        :param url: the url which needs to be called to fetch the transcript
        :param language: the name of the language this transcript uses
        :param language_code:
        :param is_generated:
        :param translation_languages:
        """
        self._http_client = http_client
        self.video_id = video_id
        self._url = url
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated
        self.translation_languages = translation_languages
        self._translation_languages_dict = {
            translation_language['language_code']: translation_language['language']
            for translation_language in translation_languages
        }

    def fetch(self, preserve_formatting=False):
        """
        Loads the actual transcript data.
        :param preserve_formatting: whether to keep select HTML text formatting
        :type preserve_formatting: bool
        :return: a list of dictionaries containing the 'text', 'start' and 'duration' keys
        :rtype [{'text': str, 'start': float, 'end': float}]:
        """
        response = self._http_client.get(self._url, headers={'Accept-Language': 'en-US'})
        return _TranscriptParser(preserve_formatting=preserve_formatting).parse(
            _raise_http_errors(response, self.video_id).text,
        )

    def __str__(self):
        return '{language_code} ("{language}"){translation_description}'.format(
            language=self.language,
            language_code=self.language_code,
            translation_description='[TRANSLATABLE]' if self.is_translatable else ''
        )

    @property
    def is_translatable(self):
        return len(self.translation_languages) > 0

    def translate(self, language_code):
        if not self.is_translatable:
            raise NotTranslatableError(self.video_id)

        if language_code not in self._translation_languages_dict:
            raise TranslationLanguageNotAvailableError(self.video_id)

        return Transcript(
            self._http_client,
            self.video_id,
            '{url}&tlang={language_code}'.format(url=self._url, language_code=language_code),
            self._translation_languages_dict[language_code],
            language_code,
            True,
            [],
        )


class _TranscriptParser(object):
    _FORMATTING_TAGS = [
        'strong',  # important
        'em',  # emphasized
        'b',  # bold
        'i',  # italic
        'mark',  # marked
        'small',  # smaller
        'del',  # deleted
        'ins',  # inserted
        'sub',  # subscript
        'sup',  # superscript
    ]

    def __init__(self, preserve_formatting=False):
        self._html_regex = self._get_html_regex(preserve_formatting)

    def _get_html_regex(self, preserve_formatting):
        if preserve_formatting:
            formats_regex = '|'.join(self._FORMATTING_TAGS)
            formats_regex = r'<\/?(?!\/?(' + formats_regex + r')\b).*?\b>'
            html_regex = re.compile(formats_regex, re.IGNORECASE)
        else:
            html_regex = re.compile(r'<[^>]*>', re.IGNORECASE)
        return html_regex

    def parse(self, plain_data):
        return [
            {
                'text': re.sub(self._html_regex, '', unescape(xml_element.text)),
                'start': float(xml_element.attrib['start']),
                'duration': float(xml_element.attrib.get('dur', '0.0')),
            }
            for xml_element in ElementTree.fromstring(plain_data)
            if xml_element.text is not None
        ]

WATCH_URL = 'https://www.youtube.com/watch?v={video_id}'

class transcriber(object):
    @classmethod
    def list_transcripts(cls, video_id, proxies=None, cookies=None):
        with requests.Session() as http_client:
            if cookies:
                http_client.cookies = cls._load_cookies(cookies, video_id)
            http_client.proxies = proxies if proxies else {}
            return TranscriptListFetcher(http_client).fetch(video_id)

    @classmethod
    def get_transcripts(cls, video_ids, languages=('en',), continue_after_error=False, proxies=None,
                        cookies=None, preserve_formatting=False):

        assert isinstance(video_ids, list), "`video_ids` must be a list of strings"

        data = {}
        unretrievable_videos = []

        for video_id in video_ids:
            try:
                data[video_id] = cls.get_transcript(video_id, languages, proxies, cookies, preserve_formatting)
            except Exception as exception:
                if not continue_after_error:
                    raise exception

                unretrievable_videos.append(video_id)

        return data, unretrievable_videos

    @classmethod
    def get_transcript(cls, video_id, languages=('en',), proxies=None, cookies=None, preserve_formatting=False):
        assert isinstance(video_id, str), "`video_id` must be a string"
        return cls.list_transcripts(video_id, proxies, cookies).find_transcript(languages).fetch(preserve_formatting=preserve_formatting)

    @classmethod
    def _load_cookies(cls, cookies, video_id):
        try:
            cookie_jar = cookiejar.MozillaCookieJar()
            cookie_jar.load(cookies)
            if not cookie_jar:
                raise CookiesInvalidError(video_id)
            return cookie_jar
        except:
            raise CookiePathInvalidError(video_id)

class LLM:
    def __init__(self, model: str, system_message: str = "You are a Helpful AI."):
        self.model = model
        self.conversation_history = [{"role": "system", "content": system_message}]

    def chat(self, messages: List[Dict[str, str]]) -> Union[str, None]:
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Pragma': 'no-cache',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        data = json.dumps(
            {
                'model': self.model,
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 16000,
                'stop': [],
                'stream': False #dont change it
            }, separators=(',', ':')
        )
        try:
            result = requests.post(url=url, data=data, headers=headers)
            return result.json()['choices'][0]['message']['content']
        except:
            return None

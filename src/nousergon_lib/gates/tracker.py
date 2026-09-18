"""The rolling-issue adapter: comment, create, rewrite a body — never close.

Lifted from `crucible/crucible/tracker.py` on its second adoption
(`shared-code-policy`, commissioned by `alpha-engine-config-I10951`). What
moved is the WIRE: request construction, credential resolution, the
find-or-create by title, and the three mutating calls. What did not move is
any caller's vocabulary — the repository, the environment-variable names and
the issue title are all constructor arguments or call arguments, so nothing
here names crucible.

**It may comment, create, and rewrite a body. It may never close.** Closing
or reopening an issue is a human's authority (`principles.md` §3.2). A
machine comment is a RECORD, not a closure. The mutating requests this module
can construct are a ``POST`` to an issue's ``/comments``, a ``POST`` creating
an issue, and a ``PATCH`` on ``/issues/{n}`` whose payload is the LITERAL
``{"body": ...}`` — never a caller-supplied dict, so it cannot carry
``state`` and cannot close or reopen anything regardless of what a caller
passes. There is no close method, and its absence is the control.

**Two open issues with the same title is a loud error, never a pick.**
Posting to whichever one a race or a manual duplicate left behind is a full
update nobody can find from the headline that links it.

**The credential is operator-granted and its absence is never green.** Every
call raises :class:`TrackerError` when no credential is configured, naming
both variables — a report that silently skipped its tracker post while its
manifest read ``ok`` is the accountability gap the whole instrument exists to
close.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "API_ROOT",
    "COMMENT_PAGE_CEILING",
    "COMMENT_PAGE_SIZE",
    "HTTP_TIMEOUT_S",
    "ISSUE_STATES",
    "SEARCH_API_ROOT",
    "TRACKER_APP_PERMISSIONS",
    "IssueRead",
    "Opener",
    "Tracker",
    "TrackerConfig",
    "TrackerCredentialError",
    "TrackerError",
]

#: GitHub's REST root. A module-level constant so a test can point the adapter
#: at nothing at all, and so the one hostname this package talks to outside
#: AWS is greppable in one line.
API_ROOT = "https://api.github.com"

#: GitHub's search endpoint, off the repo-scoped ``/repos/{repo}/...`` shape
#: every other call here uses — :meth:`Tracker.find_issue_by_title` is the one
#: caller.
SEARCH_API_ROOT = f"{API_ROOT}/search/issues"

#: The narrowing requested at mint time. The installation holds more; the
#: token this adapter uses holds exactly what its calls need.
TRACKER_APP_PERMISSIONS: dict[str, str] = {"issues": "write"}

#: Seconds. A daily report blocked forever on a hung socket is an absence page
#: on a working producer.
HTTP_TIMEOUT_S = 20

#: GitHub's closed vocabulary for an issue. Anything else is REFUSED rather
#: than passed through: a state a consumer has no rendering for would reach a
#: board as a row it cannot classify, and an unclassifiable row renders as
#: whatever the default branch happens to be.
ISSUE_STATES: tuple[str, ...] = ("open", "closed")

#: GitHub's maximum page size for a comment listing.
COMMENT_PAGE_SIZE = 100

#: How many pages :meth:`Tracker.comment_bodies` will follow before it
#: REFUSES. A listing that would need more raises rather than truncating —
#: see that method for why a short listing is the dangerous direction.
COMMENT_PAGE_CEILING = 10

#: A GitHub API request, as a caller may substitute it. ``(status, body)``.
#: Injected so every test exercises the real request construction — headers,
#: method, URL and payload — without a socket.
Opener = Callable[[urllib.request.Request], "tuple[int, bytes]"]


class TrackerError(RuntimeError):
    """A tracker call that could not be performed.

    Raised, never swallowed. A record filed to the store while the tracker was
    never told is precisely the two-instruments-disagreeing defect this
    mechanism exists to remove.
    """


class TrackerCredentialError(TrackerError):
    """The App-minted credential was CONFIGURED and could not be produced.

    Distinct from "no credential at all": a prefix that is set and an SSM read
    or a GitHub mint that then fails is a statement about this identity's
    grant or the App's health, and it is reported with its cause rather than
    rendered as the same absence an unconfigured laptop shows.
    """


@dataclass(frozen=True)
class TrackerConfig:
    """Which repository this adapter writes to, and where its credential is.

    The variable NAMES are configuration, not constants, because two systems
    now use this adapter against the same tracker repository with different
    credentials, and a shared variable name would make one system's grant
    silently serve the other's job.
    """

    repo: str
    token_var: str
    app_ssm_prefix_var: str
    #: The exact operator step that grants this adapter its credential, in the
    #: caller's own vocabulary (which stack, which role, which repo variable).
    #: Appended verbatim to every credential-absence message, because an
    #: operator step recorded in an issue and nowhere else is
    #: `alpha-engine-config-I1906` — closed as *fixed* on a PR whose command
    #: was never run. Empty is legal and means the caller has nothing shorter
    #: to say than the two variable names already in the message.
    grant_hint: str = ""


def _default_opener(request: urllib.request.Request) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        # A 403/404/422 is an ANSWER, not a transport failure: it carries a
        # body naming what GitHub refused, and the caller renders that. Only a
        # URLError (an OSError) propagates, and each caller catches it by name.
        return int(exc.code), exc.read()


@dataclass(frozen=True)
class IssueRead:
    """One tracker issue's state, or the reason it could not be read.

    ``state`` is one of :data:`ISSUE_STATES` when the read succeeded and
    ``None`` otherwise; exactly one of ``state``/``problem`` is set.

    ``access_problem`` is the field that earns this class its existence. A
    fault in OUR access — no credential, a 403, a transport failure — is a
    statement about this identity's grant, NOT about the issue. A consumer
    renders it ``UNMEASURABLE``; folding it into "the issue is not closed"
    would report an IAM gap as a met-or-unmet verdict, which is a wrong answer
    wearing a measurement's clothes.
    """

    state: str | None
    problem: str | None
    access_problem: bool = False

    @property
    def closed(self) -> bool:
        """True only when the tracker was READ and says closed."""
        return self.state == "closed"


class Tracker:
    """One repository's rolling issue, as the ONE adapter that touches it."""

    def __init__(self, config: TrackerConfig, *, opener: Opener | None = None) -> None:
        """``opener`` is the injectable transport; production passes nothing."""
        self.config = config
        self._opener = opener or _default_opener

    # -- credential ---------------------------------------------------------

    def credential(self) -> str | None:
        """The tracker credential, or ``None`` when none is granted.

        Resolution, in order: ``config.token_var`` in the environment; then a
        short-lived installation token minted from the fleet GitHub App when
        ``config.app_ssm_prefix_var`` is set. It is NEVER defaulted to the
        Actions token: ``GITHUB_TOKEN`` is scoped to the repository the
        workflow runs in, so falling back to it would turn "no grant" into a
        404 that reads like a deleted issue.

        Raises :class:`TrackerCredentialError` when the prefix is set and the
        mint fails — configured-and-broken is not the same fact as absent.
        """
        value = (os.environ.get(self.config.token_var) or "").strip()
        if value:
            return value
        prefix = (os.environ.get(self.config.app_ssm_prefix_var) or "").strip()
        if not prefix:
            return None
        return self._mint_from_app(prefix)

    def _mint_from_app(self, prefix: str) -> str:
        """A short-lived installation token narrowed to this adapter's calls."""
        from nousergon_lib.github_app import (  # noqa: PLC0415 - lazy: boto3 + SSM
            GitHubAppTokenError,
            installation_token,
        )

        try:
            return installation_token(ssm_prefix=prefix, permissions=dict(TRACKER_APP_PERMISSIONS))
        except GitHubAppTokenError as exc:
            raise TrackerCredentialError(
                f"{self.config.app_ssm_prefix_var}={prefix!r} is set but no installation "
                f"token could be minted from the App credentials there: {exc}. A statement "
                "about this identity's SSM grant or the App, not an absent credential."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised with the cause named
            raise TrackerCredentialError(
                f"{self.config.app_ssm_prefix_var}={prefix!r} is set but minting raised {type(exc).__name__}: {exc}"
            ) from exc

    def _granted(self, doing: str) -> str:
        granted = self.credential()
        if granted is None:
            raise TrackerError(
                f"no tracker credential (${self.config.token_var} and "
                f"${self.config.app_ssm_prefix_var} unset), so {doing} on "
                f"{self.config.repo} cannot be performed. It is not filed anywhere else "
                "either — a record in one place and not the other is the "
                "two-instruments-disagreeing defect this adapter removes."
                + (f" Grant it with: {self.config.grant_hint}" if self.config.grant_hint else "")
            )
        return granted

    # -- transport ----------------------------------------------------------

    def _send(
        self,
        url: str,
        *,
        token: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, bytes]:
        """The one request builder every call goes through.

        Split from :meth:`_request` so a caller addressing something other
        than ``/repos/{repo}/...`` — the search API — still gets the same
        headers and the same injectable opener, rather than a second,
        divergent request construction.
        """
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        return self._opener(request)

    def _request(
        self,
        path: str,
        *,
        token: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, bytes]:
        return self._send(
            f"{API_ROOT}/repos/{self.config.repo}{path}",
            token=token,
            method=method,
            payload=payload,
        )

    @staticmethod
    def _decode(body: bytes, *, doing: str) -> Any:
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TrackerError(f"{doing} did not answer with JSON: {exc}") from exc

    # -- reads --------------------------------------------------------------

    def find_issue_by_title(self, title: str) -> int | None:
        """The number of the OPEN issue titled exactly ``title``, or ``None``.

        RAISES :class:`TrackerError` when MORE than one carries it: a second
        open issue with this title is a loud failure, never a pick — posting
        to whichever one a race or a manual duplicate left behind is a full
        update nobody can find from the headline that links it.

        GitHub's search API does PHRASE matching over the whole document, not
        an exact-field match, so every candidate it returns is re-checked
        against ``title`` byte for byte (and against ``state == "open"``,
        which the query already asks for) before it counts — a substring or
        fuzzy match would silently pick a differently named issue.
        """
        repo = self.config.repo
        doing = f"searching {repo} for an issue titled {title!r}"
        token = self._granted(doing)
        query = f'repo:{repo} is:issue is:open in:title "{title}"'
        url = f"{SEARCH_API_ROOT}?q={urllib.parse.quote(query)}"
        try:
            status, body = self._send(url, token=token)
        except OSError as exc:
            raise TrackerError(f"{doing} failed at the transport: {exc}") from exc
        if status != 200:
            raise TrackerError(f"GitHub answered {status} {doing}: {body.decode('utf-8', 'replace')[:200]}")
        document = self._decode(body, doing=doing)
        items = document.get("items") if isinstance(document, dict) else None
        if not isinstance(items, list):
            raise TrackerError(f"{doing} answered with no `items` list: {document!r}")
        numbers = sorted(
            {
                item["number"]
                for item in items
                if isinstance(item, dict)
                and item.get("title") == title
                and item.get("state") == "open"
                and isinstance(item.get("number"), int)
            }
        )
        if not numbers:
            return None
        if len(numbers) > 1:
            raise TrackerError(
                f"{repo} carries {len(numbers)} open issues titled {title!r} ({numbers}); "
                "refusing to pick one — a second copy of the rolling issue is a defect to "
                "fix by hand, not by choosing"
            )
        return numbers[0]

    def read_issue(self, issue: int) -> IssueRead:
        """Whether ``issue`` is open or closed. NEVER raises.

        The lenient face, opposite :meth:`comment_bodies`, and deliberately
        so: its callers render one row of a board per issue, and a board that
        died because one optional read was denied tells nobody anything. Every
        fault becomes an :class:`IssueRead` carrying ``access_problem=True``,
        which a consumer renders as UNMEASURABLE rather than as a verdict.

        A ``state`` outside :data:`ISSUE_STATES` is a PROBLEM, not a value
        passed through — see that constant.
        """
        repo = self.config.repo
        try:
            granted = self.credential()
        except TrackerCredentialError as exc:
            return IssueRead(None, str(exc), access_problem=True)
        if granted is None:
            # Compact on purpose: rendered once per board row inside a wire
            # budget, where a longer sentence here pushes another row out of
            # the message. The grant itself is carried on the row's own
            # "what to do when this is red" text, not repeated here.
            return IssueRead(
                None,
                f"no tracker credential (${self.config.app_ssm_prefix_var} and "
                f"${self.config.token_var} unset): could not ask {repo} whether the issue "
                "is open or closed",
                access_problem=True,
            )
        try:
            status, body = self._request(f"/issues/{issue}", token=granted)
        except OSError as exc:
            return IssueRead(None, f"reading {repo}#{issue} failed at the transport: {exc}", access_problem=True)
        if status != 200:
            return IssueRead(
                None,
                f"GitHub answered {status} for {repo}#{issue}: {body.decode('utf-8', 'replace')[:200]}",
                access_problem=True,
            )
        try:
            document = self._decode(body, doing=f"reading {repo}#{issue}")
        except TrackerError as exc:
            return IssueRead(None, str(exc), access_problem=True)
        state = document.get("state") if isinstance(document, dict) else None
        if state not in ISSUE_STATES:
            return IssueRead(
                None,
                f"{repo}#{issue} reports state {state!r}, which is not one of {ISSUE_STATES}",
                access_problem=True,
            )
        return IssueRead(str(state), None)

    def comment_bodies(self, issue: int) -> list[str]:
        """Every comment body on ``issue``, oldest first. RAISES on any fault.

        The strict face, because its caller is a WRITER: it exists so a record
        is not posted twice, and a listing that silently came back short would
        post a duplicate rather than skip one. The lenient reading that suits
        :meth:`read_issue` is the wrong default here — one of these two
        callers is harmed by a guess in each direction, which is why there are
        two methods and not one with a flag.

        Pagination is followed to :data:`COMMENT_PAGE_CEILING` and a listing
        needing more RAISES rather than truncating: an issue carrying that
        many comments is a fact worth failing on, not one worth guessing past.
        """
        repo = self.config.repo
        doing = f"listing comments on {repo}#{issue}"
        token = self._granted(doing)
        bodies: list[str] = []
        for page in range(1, COMMENT_PAGE_CEILING + 1):
            path = f"/issues/{issue}/comments?per_page={COMMENT_PAGE_SIZE}&page={page}"
            try:
                status, body = self._request(path, token=token)
            except OSError as exc:
                raise TrackerError(f"{doing} failed at the transport: {exc}") from exc
            if status != 200:
                raise TrackerError(f"GitHub answered {status} {doing}: {body.decode('utf-8', 'replace')[:200]}")
            document = self._decode(body, doing=doing)
            if not isinstance(document, list):
                raise TrackerError(f"{doing} answered with a {type(document).__name__}, not a list")
            bodies.extend(str(item.get("body") or "") for item in document if isinstance(item, dict))
            if len(document) < COMMENT_PAGE_SIZE:
                return bodies
        raise TrackerError(
            f"{repo}#{issue} carries more than {COMMENT_PAGE_CEILING * COMMENT_PAGE_SIZE} comments; "
            "this reader stops rather than deciding from a truncated listing whether the "
            "record is already posted"
        )

    # -- writes -------------------------------------------------------------

    def create_issue(self, *, title: str, body: str) -> int:
        """Create an issue titled ``title``. Returns its number.

        One of the three mutating requests this adapter can construct — an
        issue CREATE, never a close, reopen, label or assign. Reached exactly
        once per rolling issue's lifetime.
        """
        repo = self.config.repo
        doing = f"creating the issue {title!r} in {repo}"
        token = self._granted(doing)
        if not title.strip():
            raise TrackerError(f"refusing to create an issue with an empty title in {repo}")
        try:
            status, raw = self._request("/issues", token=token, method="POST", payload={"title": title, "body": body})
        except OSError as exc:
            raise TrackerError(f"{doing} failed at the transport: {exc}") from exc
        if status != 201:
            raise TrackerError(f"GitHub answered {status} {doing}: {raw.decode('utf-8', 'replace')[:200]}")
        document = self._decode(raw, doing=doing)
        number = document.get("number") if isinstance(document, dict) else None
        if not isinstance(number, int):
            raise TrackerError(
                f"the issue {title!r} was created in {repo} but the answer carries no usable `number`: {document!r}"
            )
        return number

    def find_or_create_issue(self, *, title: str, body: str) -> int:
        """The rolling issue's number, creating it once if it does not exist.

        ``body`` is the issue's INITIAL body only. It is not re-posted on a
        day the issue already exists: every delivery after the first is one
        more comment, and the body is rewritten separately by
        :meth:`update_issue_body` with a regenerated index.
        """
        number = self.find_issue_by_title(title)
        if number is not None:
            return number
        return self.create_issue(title=title, body=body)

    def post_comment(self, issue: int, body: str) -> str:
        """Post ``body`` as a comment on ``issue``. Returns its permalink.

        The permalink is the return value because the headline that goes to
        the operator is a POINTER at this comment — a comment posted whose
        location cannot be named is a full update nobody can reach.
        """
        repo = self.config.repo
        doing = f"posting a comment to {repo}#{issue}"
        token = self._granted(doing)
        if not body.strip():
            raise TrackerError(
                f"refusing to post an empty comment to {repo}#{issue}: an empty record is "
                "indistinguishable from no record and harder to notice."
            )
        try:
            status, raw = self._request(f"/issues/{issue}/comments", token=token, method="POST", payload={"body": body})
        except OSError as exc:
            raise TrackerError(f"{doing} failed at the transport: {exc}") from exc
        if status != 201:
            raise TrackerError(f"GitHub answered {status} {doing}: {raw.decode('utf-8', 'replace')[:200]}")
        document = self._decode(raw, doing=doing)
        url = document.get("html_url") if isinstance(document, dict) else None
        if not isinstance(url, str) or not url:
            raise TrackerError(
                f"the comment on {repo}#{issue} was accepted but carries no html_url, so the "
                "record cannot name where it landed"
            )
        return url

    def update_issue_body(self, issue: int, body: str) -> None:
        """Replace ``issue``'s BODY. Never its state, title, labels or assignees.

        **The payload is a LITERAL ``{"body": body}``, never a caller-supplied
        dict.** ``PATCH /issues/{n}`` is the one GitHub request that COULD
        close or reopen an issue, by carrying a ``state`` key — this method
        accepts no argument that could ever reach that key, so it cannot be
        made to close anything regardless of what a caller passes.
        """
        repo = self.config.repo
        doing = f"rewriting the body of {repo}#{issue}"
        token = self._granted(doing)
        try:
            status, raw = self._request(f"/issues/{issue}", token=token, method="PATCH", payload={"body": body})
        except OSError as exc:
            raise TrackerError(f"{doing} failed at the transport: {exc}") from exc
        if status != 200:
            raise TrackerError(f"GitHub answered {status} {doing}: {raw.decode('utf-8', 'replace')[:200]}")

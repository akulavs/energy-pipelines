import collections.abc
import contextlib
import functools
import http
import logging
import types
from typing import Annotated, Any, Self, cast

import annotated_types
import httpx
import pydantic
import tenacity
import whenever

import common.models
import common.types


__all__ = (
    "AsyncHTTPClient",
    "BaseHTTPClientConfiguration",
    "HTTPErrorStatus",
    "HTTPRequestRetryConfiguration",
)


LOGGER = logging.getLogger(__name__)


type HTTPErrorStatus = Annotated[http.HTTPStatus, annotated_types.Gt(399)]


class HTTPRequestRetryConfiguration(common.models.FrozenModel):
    """
    Parameters that control retry behavior for failed requests.
    """

    max_timeout: common.types.PositiveTimeDelta = pydantic.Field(
        default=whenever.seconds(30),
        description="max time to wait for a single request, including retries",
    )
    per_request_timeout: common.types.PositiveTimeDelta = pydantic.Field(
        default=whenever.seconds(5),
        description="timeout per request",
    )
    max_retries: pydantic.PositiveInt = pydantic.Field(
        default=3,
        description="maximum number of retries for failed requests",
    )
    use_exponential_backoff: bool = pydantic.Field(
        default=True,
        description="whether to use exponential backoff, with jitter, between retries",
    )
    initial_wait: common.types.NonNegativeTimeDelta = pydantic.Field(
        default=whenever.seconds(1),
        description="initial wait time before retrying a failed request",
    )
    jitter_upper_bound: common.types.NonNegativeTimeDelta = pydantic.Field(
        default=whenever.seconds(1),
        description="upper bound for random jitter added to wait times between retries",
    )
    retry_on_statuses: frozenset[HTTPErrorStatus] = pydantic.Field(
        default=frozenset(
            {
                http.HTTPStatus(408),
                http.HTTPStatus(429),
                http.HTTPStatus(500),
                http.HTTPStatus(502),
                http.HTTPStatus(503),
                http.HTTPStatus(504),
            }
        ),
        description="HTTP statuses that should trigger a retry",
    )

    @pydantic.field_validator(
        "per_request_timeout", "initial_wait", "jitter_upper_bound", mode="after"
    )
    @classmethod
    def _check_waits_timeouts_consistent(
        cls,
        value: whenever.TimeDelta,
        info: pydantic.ValidationInfo,
    ) -> whenever.TimeDelta:
        """
        Ensure that each of these field values is < `max_timeout`.
        """
        if (
            max_timeout := info.data.get("max_timeout")
        ) is None:  # max_timeout didn't pass validation
            return value

        if value > max_timeout:
            msg = f"{info.field_name} ({value}) must be < max_timeout ({max_timeout})"
            raise ValueError(msg)

        return value


class BaseHTTPClientConfiguration(common.models.FrozenModel):
    """
    Configuration for HTTP clients.
    """

    base_url: pydantic.AnyHttpUrl | None = pydantic.Field(
        default=None, description="base URL to use for requests"
    )
    headers: common.types.Mapping[
        common.types.NonEmptyStr, common.types.NonEmptyStr
    ] = pydantic.Field(
        default_factory=lambda: types.MappingProxyType({}),
        description="headers to include in all requests",
    )
    enable_http2: bool = pydantic.Field(
        default=True, description="whether to enable HTTP/2 support"
    )
    follow_redirects: bool = pydantic.Field(
        default=False, description="whether to follow redirects"
    )
    max_connections: pydantic.PositiveInt | None = pydantic.Field(
        default=16,
        description="max number of allowable connections, or None for unlimited",
    )
    retry_configuration: HTTPRequestRetryConfiguration = pydantic.Field(
        default_factory=HTTPRequestRetryConfiguration,
        description="default configuration for retrying failed requests",
    )


class AsyncHTTPClient(
    contextlib.AbstractAsyncContextManager, common.models.StrictModel, extra="forbid"
):
    """
    Wrapper around `httpx.AsyncClient` with configurable retry logic.

    `request()` always calls `response.raise_for_status()`. By default, it will retry transient errors,
    using the instance-level configuration, though `request()` can be changed on a per-request basis.
    """

    configuration: BaseHTTPClientConfiguration = pydantic.Field(
        default_factory=BaseHTTPClientConfiguration,
        description="HTTP client configuration",
        frozen=True,
    )
    _client: httpx.AsyncClient = pydantic.PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ARG002
        client_config = self._marshal_client_config()
        self._client = httpx.AsyncClient(**client_config)

    def model_copy(
        self,
        *,
        update: collections.abc.Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """
        Create a copy or initialize a new instance.

        If `update`s are passed or `deep` is `True`, a new instance, wrapping a new `httpx.AsyncClient`, is initialized.
        """
        if (updates := dict(update or {})) or deep is True:
            # cannot deepcopy an AsyncClient, so create a new instance
            return self.model_validate(self.model_dump(mode="python") | updates)

        return super().model_copy(update=update, deep=deep)

    @pydantic.validate_call
    async def request(
        self,
        method: http.HTTPMethod,
        url: str | pydantic.AnyHttpUrl,
        *,
        retry_configuration: HTTPRequestRetryConfiguration | None = None,
        **request_kwargs: Any,
    ) -> httpx.Response:
        """
        Wrapper around `httpx.AsyncClient.request()` that allows customizing retry behavior.

        If retries are exhausted, the originally encountered exception is raised (not a RetryError).

        Args:
            `method`: the HTTP method to use for the request
            `url`: the URL to request
            `retry_configuration`: retry behavior specific to this request, if any; if `None`, the instance-level
                configuration is used
            `request_kwargs`: additional arguments to forward to `httpx.AsyncClient.request()`
        """
        retryer = self._get_retry_wrapper(retry_configuration)
        return await retryer(
            self._client.request, method.name, str(url), **request_kwargs
        )

    async def get(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.GET, url, **kwargs)

    async def head(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.HEAD, url, **kwargs)

    async def options(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.OPTIONS, url, **kwargs)

    async def post(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.POST, url, **kwargs)

    async def put(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.PUT, url, **kwargs)

    async def patch(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.PATCH, url, **kwargs)

    async def delete(
        self, url: str | pydantic.AnyHttpUrl, **kwargs: Any
    ) -> httpx.Response:
        return await self.request(http.HTTPMethod.DELETE, url, **kwargs)

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return await self._client.__aexit__(exc_type, exc_val, exc_tb)

    def _marshal_client_config(self) -> dict[str, Any]:
        """
        Marshal configuration for an `httpx.AsyncClient`.
        """
        limits = httpx.Limits(max_connections=self.configuration.max_connections)
        timeouts = httpx.Timeout(
            self.configuration.retry_configuration.per_request_timeout.total("seconds")
        )

        return {
            "base_url": str(self.configuration.base_url)
            if self.configuration.base_url
            else "",
            "headers": self.configuration.headers,
            "http2": self.configuration.enable_http2,
            "follow_redirects": self.configuration.follow_redirects,
            "limits": limits,
            "timeout": timeouts,
            # register this event hook so we always raise on 4xx/5xx responses
            # the retry wrapper depends on this behavior, since it's triggered by exceptions (which it then inspects)
            "event_hooks": {"response": [self._raise_on_4xx_5xx]},
            "trust_env": False,
        }

    @staticmethod
    async def _raise_on_4xx_5xx(response: httpx.Response) -> None:
        """
        `httpx` requires that event hooks for async clients be async functions.
        """
        response.raise_for_status()

    def _get_retry_wrapper(
        self,
        retry_configuration: HTTPRequestRetryConfiguration | None = None,
    ) -> tenacity.AsyncRetrying:
        """
        Get a properly configured `tenacity` retry wrapper. This method is called by `request()`,
        so that retry behavior can be customized per-request. (An alternative would be to implement retry behavior via
        a custom `httpx.AsyncBaseTransport`.)
        """
        if retry_configuration is None:
            config = self.configuration.retry_configuration
        else:
            config = self.configuration.retry_configuration.model_copy(
                update=retry_configuration.model_dump()
            )

        max_attempts = config.max_retries + 1

        wait_strategies: list[tenacity.wait.wait_base] = [
            cast(tenacity.wait.wait_base, self._read_retry_after_header)
        ]
        if config.use_exponential_backoff is True:
            wait_strategies.append(
                tenacity.wait_exponential_jitter(
                    initial=config.initial_wait.total("seconds"),
                    max=config.max_timeout.total("seconds"),
                    jitter=config.jitter_upper_bound.total("seconds"),
                )
            )

        is_transient_error = functools.partial(
            self._is_transient_error,
            retry_on_statuses=config.retry_on_statuses,
        )

        return tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception(is_transient_error)
            | tenacity.retry_if_exception_type(httpx.TimeoutException),
            wait=tenacity.wait_combine(*wait_strategies),
            before_sleep=tenacity.before_sleep_log(LOGGER, logging.INFO),  # type: ignore[arg-type]
            stop=tenacity.stop_after_delay(config.max_timeout.to_stdlib())
            | tenacity.stop_after_attempt(max_attempts),
            reraise=True,
        )

    @staticmethod
    def _is_transient_error(
        exception: BaseException,
        retry_on_statuses: collections.abc.Set[HTTPErrorStatus],
    ) -> bool:
        match exception:
            case httpx.HTTPStatusError():
                return exception.response.status_code in retry_on_statuses
            case _:
                return False

    @staticmethod
    def _read_retry_after_header(retry_state: tenacity.RetryCallState) -> float:
        if retry_state.outcome is None or retry_state.outcome.failed is False:
            return 0

        exc = retry_state.outcome.exception()
        if not isinstance(exc, httpx.HTTPStatusError):
            return 0

        retry_after = exc.response.headers.get("retry-after", 0)

        with contextlib.suppress(ValueError):
            return float(retry_after)

        try:
            wait_until = whenever.Instant.parse_rfc2822(retry_after)
        except ValueError:
            msg = f"could not parse Retry-After header value: {retry_after}"
            LOGGER.warning(msg)
            return 0
        else:
            delay = wait_until - whenever.Instant.now()
            return max(delay.in_seconds(), 0)

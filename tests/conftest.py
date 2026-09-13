import collections.abc

import prefect.testing.utilities
import pytest
import pytest_mock


@pytest.fixture(name="prefect_test_harness", scope="session", autouse=True)
def fixture_prefect_test_harness() -> collections.abc.Generator[None]:
    """An ephemeral Prefect API for the whole session, so flows and tasks run for
    real instead of every test mocking `prefect.get_run_logger()`."""
    with prefect.testing.utilities.prefect_test_harness():
        yield


@pytest.fixture(name="mock_get_run_logger")
def fixture_mock_get_run_logger(mocker: pytest_mock.MockerFixture) -> None:
    """For tests that call a task or flow body directly, outside any run context."""
    mocker.patch("prefect.get_run_logger", autospec=True)

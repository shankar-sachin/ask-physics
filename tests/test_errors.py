import pytest

from askphysics import errors


@pytest.mark.parametrize(
    "exc",
    [
        errors.ConfigError,
        errors.DataValidationError,
        errors.LLMError,
        errors.LLMResponseFormatError,
        errors.OutOfScopeError,
        errors.RetrievalEmptyError,
        errors.PlanValidationError,
        errors.SolverError,
        errors.UnitError,
        errors.UnitParseError,
        errors.UnitMismatchError,
        errors.SanityCheckError,
    ],
)
def test_every_error_derives_from_base(exc: type[Exception]) -> None:
    assert issubclass(exc, errors.AskPhysicsError)


def test_unit_and_llm_subtrees() -> None:
    assert issubclass(errors.UnitMismatchError, errors.UnitError)
    assert issubclass(errors.UnitParseError, errors.UnitError)
    assert issubclass(errors.LLMResponseFormatError, errors.LLMError)


def test_data_validation_error_lists_every_problem() -> None:
    exc = errors.DataValidationError(["bad unit", "missing source"])
    assert exc.problems == ["bad unit", "missing source"]
    assert "2 data validation problem(s)" in str(exc)
    assert "bad unit" in str(exc)
    assert "missing source" in str(exc)

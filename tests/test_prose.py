from askphysics.prose import FALLBACK_REFUSAL, readable, refusal_reason, usable_redirect

ROUGHLY = (
    "Category error: anxiety is a feeling, not a device that draws power does a device that "
    "draws power does it answerable" + " roughly" * 20
)


def test_word_salad_is_not_readable() -> None:
    assert not readable(ROUGHLY)
    assert not readable("reasoning")
    assert readable(
        "Using [coulomb_law], the charge is 3 C. This assumes point charges; charges at rest."
    )
    assert readable("From [orbital_speed] (Speed of a circular orbit), v = 3 m/s.")


def test_refusal_reasons_must_be_about_the_question() -> None:
    q = "what is ten divided by three"
    assert refusal_reason(q, ROUGHLY) == FALLBACK_REFUSAL
    transplanted = "Category error: anxiety is a feeling, not a device that draws power."
    assert refusal_reason(q, transplanted) == FALLBACK_REFUSAL
    blue = "Category error: a color is a property of light, not an object with mass."
    assert refusal_reason("How much does the color blue weigh?", blue) == blue
    generic = "Not a physics question; it is history."
    assert refusal_reason("Who was the first ruler of Lima?", generic) == generic


def test_redirects_must_be_questions() -> None:
    assert usable_redirect("reasoning") is None
    assert usable_redirect(None) is None
    assert usable_redirect("How much power does a phone charger draw") is None
    ok = "How much power does a phone charger draw?"
    assert usable_redirect(f"  {ok} ") == ok

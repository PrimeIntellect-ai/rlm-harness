import rlm


def test_dummy_sync():
    assert hasattr(rlm, "run")


async def test_dummy_async():
    assert True

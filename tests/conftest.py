import pytest


@pytest.fixture(scope="session")
def renderer():
    """One offscreen 3D renderer for the whole test run (Panda3D allows only one)."""
    try:
        from mr147.vision.scene import Renderer

        return Renderer(640, 360, seed=0)
    except Exception as e:  # no graphics available
        pytest.skip(f"cannot open an offscreen renderer: {e}")

"""O dashboard sobe sem erro (Streamlit AppTest, sem rede)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_dashboard_renderiza_sem_erro():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 1

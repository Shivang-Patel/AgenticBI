"""
Pytest fixtures for AgenticBI Dash application.
"""

import pytest
from app.main import app


@pytest.fixture
def dash_app():
    """Return the Dash application instance."""
    return app


@pytest.fixture
def test_client(dash_app):
    """Create a test client for the Dash app."""
    return dash_app.server.test_client()


@pytest.fixture
def sample_store_data():
    """Return sample store data for testing callbacks."""
    return {
        "charts": [],
        "messages": [
            {
                "role": "user",
                "content": "Show me sales by region",
                "chart": None,
            }
        ],
        "chart_counter": 0,
        "pinned": [],
        "deleted": [],
    }


@pytest.fixture
def sample_workspace_info():
    """Return sample workspace metadata."""
    return {
        "workspace_id": "test-workspace-001",
        "workspace_name": "Test Workspace",
        "tables": ["sales", "customers", "products"],
    }

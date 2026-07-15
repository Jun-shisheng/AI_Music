import json
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app
from app.services.deepseek_arrange import _validate_arrangement


VALID_RESPONSE = {
    "title": "星空下的约定",
    "language": "zh",
    "lyrics": "[前奏]\n\n[主歌]\n夜空下的星光闪烁\n照亮了我们的约定\n\n[副歌]\n在星空下许下的诺言\n永远不会改变\n",
    "style": {
        "genres": ["流行", "国风"],
        "moods": ["治愈"],
        "instruments": ["钢琴", "弦乐"],
        "vocals": ["女声"],
        "tempo": "中速",
    },
    "style_prompt": "Chinese pop, guofeng, healing, female vocal, piano, strings, mid tempo",
}


class TestValidateArrangement:
    def test_valid_arrangement_passes(self):
        _validate_arrangement(VALID_RESPONSE)

    def test_missing_top_level_fields_raises(self):
        with pytest.raises(ValueError, match="Missing required fields"):
            _validate_arrangement({"title": "test"})

    def test_missing_style_fields_raises(self):
        bad = {**VALID_RESPONSE, "style": {"genres": ["pop"]}}
        with pytest.raises(ValueError, match="Missing style fields"):
            _validate_arrangement(bad)


class TestArrangeEndpoint:
    @pytest.mark.asyncio
    async def test_arrange_returns_valid_json(self):
        transport = ASGITransport(app=app)
        with patch("app.routers.arrange.arrange") as mock_arrange:
            mock_arrange.return_value = VALID_RESPONSE
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/arrange", json={
                    "prompt": "一首关于星空的治愈系歌曲",
                })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["title"] == "星空下的约定"

    @pytest.mark.asyncio
    async def test_arrange_empty_requirement_rejected(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/arrange", json={"prompt": ""})
        assert resp.status_code == 422

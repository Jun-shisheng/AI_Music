import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.deepseek_arrange import arrange

router = APIRouter(prefix="/arrange", tags=["arrange"])


class ArrangeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="歌曲要求描述")
    lyrics: str = Field(default="", description="可选：用户提供的歌词")


@router.post("")
async def create_arrangement(req: ArrangeRequest):
    try:
        result = await arrange(
            requirement=req.prompt,
            lyrics=req.lyrics,
            language="zh",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"DeepSeek returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arrangement failed: {e}")

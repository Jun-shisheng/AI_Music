import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.deepseek_arrange import arrange

router = APIRouter(prefix="/arrange", tags=["arrange"])


class ArrangeRequest(BaseModel):
    requirement: str = Field(..., min_length=1, description="歌曲要求描述")
    lyrics: str = Field(default="", description="可选：用户提供的歌词")
    language: str = Field(default="zh", pattern="^(zh|en)$")


class ArrangeResponse(BaseModel):
    success: bool
    result: dict | None = None
    error: str | None = None


@router.post("", response_model=ArrangeResponse)
async def create_arrangement(req: ArrangeRequest):
    try:
        result = await arrange(
            requirement=req.requirement,
            lyrics=req.lyrics,
            language=req.language,
        )
        return ArrangeResponse(success=True, result=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"DeepSeek returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arrangement failed: {e}")

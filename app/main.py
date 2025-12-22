import json
from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from . import crud, inference, models, schemas
from .auth import create_access_token, get_current_user, verify_password
from .db import Base, engine, get_db
from .export_utils import analyses_to_csv, analyses_to_pdf
from .settings import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def problem_response(
    request: Request, status_code: int, title: str, detail: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": str(request.url.path),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return problem_response(
        request=request,
        status_code=exc.status_code,
        title=str(exc.status_code),
        detail=detail,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return problem_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        title="Validation error",
        detail=str(exc),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return problem_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="Internal Server Error",
        detail=str(exc),
    )


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    # lazy-load model so first request is fast
    inference.get_model_components()


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/auth/register", response_model=schemas.TokenResponse)
def register(
    payload: schemas.RegisterRequest, db: Session = Depends(get_db)
) -> schemas.TokenResponse:
    existing = crud.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    user = crud.create_user(db, payload.email, payload.password)
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return schemas.TokenResponse(access_token=token)


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(
    payload: schemas.LoginRequest, db: Session = Depends(get_db)
) -> schemas.TokenResponse:
    user = crud.get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return schemas.TokenResponse(access_token=token)


@app.get("/auth/me", response_model=schemas.UserOut)
def me(current_user: models.User = Depends(get_current_user)) -> schemas.UserOut:
    return current_user


def _text_preview(text: str, length: int = 120) -> str:
    return text[:length] + ("..." if len(text) > length else "")


def _parse_segments(segments_json: str) -> List[schemas.Segment]:
    try:
        segments_data = json.loads(segments_json or "[]")
        return [schemas.Segment(**s) for s in segments_data]
    except Exception:
        return []


@app.post(
    "/analyze",
    response_model=schemas.AnalyzeResponse,
    summary="Analyze text for AI generation",
    description=(
        "Returns an analysis result where label 1 means AI-generated and "
        "label 0 means Human-written."
    ),
    response_description="Analysis result (label 1 = AI-generated, label 0 = Human-written).",
)
def analyze(
    payload: schemas.AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required",
        )

    result = inference.analyze_text(payload.text)
    analysis = crud.create_analysis(
        db=db,
        user_id=current_user.id,
        text=payload.text,
        label_pred=result["label"],
        prob_ai=result["prob_ai"],
        model_name=result.get("model_name", "model"),
        segments=result["segments"],
    )
    return schemas.AnalyzeResponse(
        label=analysis.label_pred,
        prob_ai=analysis.prob_ai,
        segments=[schemas.Segment(**seg) for seg in result["segments"]],
        threshold=settings.AI_THRESHOLD,
    )


@app.get("/analyses", response_model=List[schemas.AnalysisPreviewOut])
def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    records = crud.get_analyses(db, user_id=current_user.id, limit=limit)
    output = []
    for r in records:
        output.append(
            schemas.AnalysisPreviewOut(
                id=r.id,
                created_at=r.created_at or datetime.utcnow(),
                label_pred=r.label_pred,
                prob_ai=r.prob_ai,
                text_preview=_text_preview(r.text),
                segments=_parse_segments(r.segments_json),
            )
        )
    return output


@app.get("/analyses/{analysis_id}", response_model=schemas.AnalysisOut)
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    record = crud.get_analysis_by_id(
        db, user_id=current_user.id, analysis_id=analysis_id
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )
    return schemas.AnalysisOut(
        id=record.id,
        created_at=record.created_at or datetime.utcnow(),
        label_pred=record.label_pred,
        prob_ai=record.prob_ai,
        text=record.text,
        segments=_parse_segments(record.segments_json),
    )


@app.get("/stats", response_model=schemas.StatsOut)
def stats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    data = crud.get_stats(db, user_id=current_user.id)
    return schemas.StatsOut(**data)


def _all_user_analyses(db: Session, user_id: int) -> List[models.Analysis]:
    return (
        db.query(models.Analysis)
        .filter(models.Analysis.user_id == user_id)
        .order_by(models.Analysis.created_at.desc())
        .all()
    )


@app.get("/export/csv")
def export_csv(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analyses = _all_user_analyses(db, current_user.id)
    csv_text = analyses_to_csv(analyses)
    headers = {"Content-Disposition": 'attachment; filename="analyses.csv"'}
    return Response(content=csv_text, media_type="text/csv", headers=headers)


@app.get("/export/pdf")
def export_pdf(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analyses = _all_user_analyses(db, current_user.id)
    pdf_bytes = analyses_to_pdf(analyses)
    headers = {"Content-Disposition": 'attachment; filename="analyses.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

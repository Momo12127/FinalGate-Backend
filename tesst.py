import os
import uuid
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from concurrent.futures import ThreadPoolExecutor
import time
import re

# ========== Executor for synchronous tasks ==========
executor = ThreadPoolExecutor(max_workers=50)

# ========== FastAPI App ==========
app = FastAPI(
    title="Finalgate AI Platform",
    description="Product Quality Analysis + Chat AI",
    version="2.8.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Supabase Setup ==========
SUPABASE_URL = "https://soxwifnrwqkbfpvzdfkl.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNveHdpZm5yd3FrYmZwdnpkZmtsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ0NTkyNzMsImV4cCI6MjA4MDAzNTI3M30.44Jzm3XP35KPMJlE7YCZ9Yp95Y0bPJX2cCIJ2ogmYxw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
STORAGE_BUCKET = "product-images"

# ========== NVIDIA AI Config ==========
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = "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"
VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

SYSTEM_PROMPT = """
You are Finalgate AI, the intelligent assistant for the Finalgate web platform.

Finalgate is a web-based system that allows users to fully control a production conveyor system remotely using AI:

1. Users can upload a Standard Reference (image + description).
2. Users can set the motor speed for the conveyor belt.
3. The system automatically sorts products based on comparison vs the Standard Reference.
4. Finalgate AI provides real-time feedback: number of accepted and rejected products.
5. Users can create accounts with different permission levels.

Always explain clearly, provide guidance, and help users operate and monitor their production line efficiently.
"""

# ========== Pydantic Models ==========
class QuickAnalysisResponse(BaseModel):
    decision: str
    reason: str
    image_url: str
    standard_id: str = Field(..., description="Standard ID as string")
    standard_description: str
    standard_product_id: str

class StandardResponse(BaseModel):
    message: str
    product_id: str
    image_url: str

# ========== Helper Functions ==========
DECISION_PATTERNS = [
    re.compile(r'(?:DECISION|decision|Decision).*?([A-Z]+)', re.IGNORECASE | re.DOTALL),
    re.compile(r'(ACCEPTED|REJECTED)', re.IGNORECASE),
]

def parse_perfect_accuracy(text: str) -> tuple:
    text_upper = text.upper()
    for pattern in DECISION_PATTERNS:
        match = pattern.search(text)
        if match:
            decision = match.group(1).strip().upper()
            if decision in ["ACCEPTED", "PASS", "GOOD", "OK", "PERFECT", "MATCH", "IDENTICAL"]:
                return "ACCEPTED", "Perfect match - identical to standard"
            elif decision in ["REJECTED", "FAIL", "BAD", "DEFECT"]:
                return "REJECTED", "Quality defect detected"

    if any(word in text_upper for word in ["IDENTICAL", "PERFECT", "MATCH", "SAME"]):
        return "ACCEPTED", "Images identical"
    if any(word in text_upper for word in ["DIFFERENT", "DEFECT", "WRONG", "ISSUE", "MISMATCH"]):
        return "REJECTED", "Does not match standard"

    return "ACCEPTED", "No defects found"

def upload_image_sync(file_content: bytes, file_path: str) -> bool:
    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            file_path,
            file_content,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        return True
    except Exception as e:
        print(f"Upload failed for {file_path}: {e}")
        return False

async def fast_upload_to_supabase(file: UploadFile, folder: str, wait_seconds: float = 0.0) -> str:
    file_content = await file.read()
    timestamp = str(int(time.time()))
    safe_filename = f"{folder}/{uuid.uuid4().hex}_{timestamp}.jpg"

    success = await asyncio.get_event_loop().run_in_executor(
        executor, upload_image_sync, file_content, safe_filename
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to upload image to Supabase Storage")

    public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(safe_filename)
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
    return public_url

async def analyze_perfect_accuracy(standard_url: str, product_url: str, description: str) -> tuple:
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": """Strict quality control AI.
Compare PRODUCT image vs REFERENCE standard image.
- Identical → DECISION: ACCEPTED
- Any difference → DECISION: REJECTED
Exact format: DECISION: ACCEPTED or REJECTED, REASON: One short sentence."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"Reference standard: {description}"},
                {"type": "image_url", "image_url": {"url": standard_url}},
                {"type": "image_url", "image_url": {"url": product_url}}
            ]}
        ],
        "max_tokens": 120,
        "temperature": 0.0
    }
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
        async with session.post(NVIDIA_API_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise HTTPException(status_code=502, detail=f"AI service error: {resp.status} {error_text}")
            result = await resp.json()
            content = result["choices"][0]["message"]["content"].strip()
            return parse_perfect_accuracy(content)

async def get_active_standard() -> Optional[Dict]:
    def fetch():
        try:
            result = supabase.table("standard_reference").select("*").eq("is_active", True).limit(1).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"DB error: {e}")
            return None
    return await asyncio.get_event_loop().run_in_executor(executor, fetch)

async def log_result_background(data: Dict):
    try:
        await asyncio.get_event_loop().run_in_executor(
            executor, lambda: supabase.table("product_quality_logs").insert([data]).execute()
        )
    except Exception as e:
        print(f"Logging failed: {e}")

# ========== Endpoints ==========
@app.get("/")
async def root():
    return {"message": "🚀 Quality API v2.8 + Chat AI Ready", "status": "✅ Ready"}

@app.post("/quick-analyze/", response_model=QuickAnalysisResponse)
async def quick_analyze(product_image: UploadFile = File(...)):
    standard = await get_active_standard()
    if not standard:
        raise HTTPException(status_code=404, detail="❌ No ACTIVE standard set. Please select one first.")

    product_url = await fast_upload_to_supabase(product_image, "quick_analysis", wait_seconds=2.0)
    decision, reason = await analyze_perfect_accuracy(standard["image_url"], product_url, standard["description"])

    log_data = {
        "product_id": standard["product_id"],
        "batch_number": f"QUICK-{int(time.time())}",
        "decision": decision,
        "reason": reason,
        "image_url": product_url,
        "standard_id": str(standard.get("id")),
        "created_at": datetime.now().isoformat()
    }
    asyncio.create_task(log_result_background(log_data))

    return QuickAnalysisResponse(
        decision=decision,
        reason=reason,
        image_url=product_url,
        standard_id=str(standard.get("id")),
        standard_description=standard["description"],
        standard_product_id=standard["product_id"]
    )

@app.post("/chat-ai/")
async def chat_ai(request: Request):
    data = await request.json()
    user_message = data.get("message")
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(NVIDIA_API_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise HTTPException(status_code=502, detail=f"NVIDIA AI Error: {resp.status} {error_text}")
            result = await resp.json()
            reply = result["choices"][0]["message"]["content"].strip()
            return {"reply": reply}

@app.post("/set-standard/", response_model=StandardResponse)
async def set_standard(
    image: UploadFile = File(...),
    product_id: str = Form(...),
    description: str = Form(...)
):
    image_url = await fast_upload_to_supabase(image, f"standards/{product_id}", wait_seconds=0.0)
    data = {
        "product_id": product_id,
        "description": description,
        "image_url": image_url,
        "is_active": False,
        "created_at": datetime.now().isoformat()
    }
    await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: supabase.table("standard_reference").upsert(data, on_conflict="product_id").execute()
    )
    return StandardResponse(message="✅ Standard created/updated", product_id=product_id, image_url=image_url)

# ========== Run ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
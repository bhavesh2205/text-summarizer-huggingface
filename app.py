from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize FastAPI app
app = FastAPI(title="Text Summarizer", description="API for text summarization")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load summarization model
# Using a smaller, lighter-weight model to avoid memory issues
import os

# Set cache directory to D: to avoid C: drive space issues
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create cache directory if it doesn't exist
cache_dir = os.path.expandvars("D:/huggingface_cache")
os.makedirs(cache_dir, exist_ok=True)

model_name = "sshleifer/distilbart-cnn-6-6"  # Smaller model variant
tokenizer = None
model = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")


# Request model
class TextRequest(BaseModel):
    text: str


# Response model
class SummaryResponse(BaseModel):
    original_text: str
    summary: str


@app.get("/")
def read_root():
    return {"message": "Text Summarizer API. Use POST /summarize to summarize text."}


@app.post("/summarize", response_model=SummaryResponse)
def summarize_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Server initialization failed."
        )

    try:
        inputs = tokenizer(
            request.text, max_length=1024, truncation=True, return_tensors="pt"
        )
        summary_ids = model.generate(
            inputs["input_ids"], max_length=70, min_length=35, do_sample=False
        )
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

        return SummaryResponse(original_text=request.text, summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

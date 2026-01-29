
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import base64
import os

from database import SessionLocal, Conversation, Message, User
from auth import router as auth_router, get_current_user, get_db
from rag_service import chain, parse_terraform_code, validate_terraform_code, generate_diagram_image
from langchain_core.messages import AIMessage, HumanMessage

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In prod, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")

# Pydantic Schemas for Requests
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    files: Optional[dict] = None
    diagram_b64: Optional[str] = None
    conversation_id: int
    validation_status: str

# Endpoints

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Terraform Architect API is running. Please verify that you are accessing the frontend at http://localhost"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    
    # 1. Get or Create Conversation
    if request.conversation_id:
        conversation = db.query(Conversation).filter(Conversation.id == request.conversation_id, Conversation.user_id == current_user.id).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(user_id=current_user.id, title=request.message[:30])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 2. Reconstruct History
    # We grab last 10 messages for context
    past_messages = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.id).all()[-10:]
    chat_history = []
    for msg in past_messages:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        else:
            chat_history.append(AIMessage(content=msg.content))

    # 3. Invoke RAG Logic (Loop for Validation)
    MAX_RETRIES = 3
    current_input = request.message
    final_answer = ""
    validation_status = "Skipped" 
    files = {}
    
    # Save User Message to DB
    user_msg_db = Message(conversation_id=conversation.id, role="user", content=request.message)
    db.add(user_msg_db)
    db.commit()

    # Generation Loop
    for attempt in range(MAX_RETRIES):
        hist_for_run = chat_history # In a real advanced loop we'd append temp history
        
        # Invoke Chain
        ai_res = chain.invoke({"chat_history": hist_for_run, "input": current_input})
        answer = ai_res["answer"]
        
        files = parse_terraform_code(answer)
        
        if not files:
            final_answer = answer
            validation_status = "Text Only"
            break
            
        # Validate
        is_valid, val_msg = validate_terraform_code(files)
        
        if is_valid:
            final_answer = answer + f"\n\n✅ Code Validated Successfully."
            validation_status = "Success"
            break
        else:
            # Failed
            if attempt == MAX_RETRIES - 1:
                final_answer = answer + f"\n\n❌ Validation Failed after retries."
                validation_status = "Failed"
            else:
                current_input = (
                    f"The previous Terraform code failed validation.\n"
                    f"Errors:\n{val_msg}\n"
                    f"Please fix the code."
                )
    
    # 4. Generate Diagram (if files exist and valid)
    diagram_b64 = None
    if files and validation_status == "Success":
        # Combine files (simplified)
        full_code = "\n".join(files.values())
        img_file = generate_diagram_image(full_code)
        if img_file:
            with open(img_file, "rb") as image_file:
                diagram_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(img_file)

    # 5. Save AI Message to DB
    ai_msg_db = Message(conversation_id=conversation.id, role="assistant", content=final_answer)
    db.add(ai_msg_db)
    db.commit()

    return ChatResponse(
        response=final_answer,
        files=files,
        diagram_b64=diagram_b64,
        conversation_id=conversation.id,
        validation_status=validation_status
    )

@app.get("/conversations")
def get_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convs = db.query(Conversation).filter(Conversation.user_id == current_user.id).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in convs]

@app.get("/conversations/{conv_id}")
def get_messages(conv_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation.messages

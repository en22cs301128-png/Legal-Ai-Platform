🚀 Overview

LegalAI Platform is a modern full-stack legal management and AI-powered research application designed to streamline legal workflows, automate document analysis, and provide intelligent legal assistance through Retrieval-Augmented Generation (RAG) and autonomous AI agents.

The platform combines:

Legal Case Management
AI Legal Research Agent
Chat with Legal Documents (RAG)
Analytics Dashboard
DevOps-ready deployment pipeline
🖥️ Frontend
HTML & ERB (Embedded Ruby)

The user interface is built using standard Ruby on Rails Views (.html.erb files), enabling seamless integration of dynamic backend data into the frontend structure.

Vanilla CSS

The entire application is styled using custom Vanilla CSS without external frameworks like TailwindCSS or Bootstrap.

Key highlights:

Modern Dark Mode UI
Glassmorphism Design
CSS Variables (:root)
Smooth Micro Animations
Responsive Layouts
Premium Dashboard Aesthetics
Vanilla JavaScript

Lightweight custom JavaScript is used to implement:

Interactive Chat Window
File Upload Simulations
Settings Tab Switching
Dynamic Calendar Events
Sidebar Navigation Effects
Chart.js & Lucide Icons
Chart.js powers the analytics and reporting dashboards.
Lucide Icons provide modern SVG-based iconography across the platform.
⚙️ Backend Architecture

The backend is divided into two independent services communicating together.

1️⃣ Primary Web Backend (Case Management)
Ruby on Rails 7

Acts as the primary web application server handling:

Routing
Authentication
Page Rendering
Legal Case Management
Business Logic
Devise Authentication

Used for:

User Login
Registration
Session Management
Secure Authentication
SQLite3

Stores:

User Accounts
Legal Cases
Metadata
Application Settings
2️⃣ AI Microservice (Research & RAG Engine)
FastAPI (Python)

A high-performance Python backend service dedicated to AI operations.

LangChain & HuggingFace

Used for:

Autonomous Legal Research
Prompt Management
Embedding Generation
AI Agent Workflows
ChromaDB

Vector database used to:

Store document embeddings
Perform semantic similarity search
Power RAG-based chat features
LLM Providers

Supports dynamic switching between:

OpenAI
Groq (Llama 3)

via environment variables.

🤖 AI Features
📄 Chat with Legal Documents (RAG)

Users can:

Upload legal documents
Generate embeddings
Ask questions about uploaded files
Receive contextual AI-generated responses
🔍 Autonomous Legal Research Agent

The AI agent can:

Accept legal queries
Perform intelligent research
Summarize findings
Generate legal insights
📊 Features
Dashboard Analytics
Case Management
Legal Research Agent
Document Upload & Processing
RAG Chat Interface
User Authentication
Calendar & Scheduling
Reports & Charts
Settings Management

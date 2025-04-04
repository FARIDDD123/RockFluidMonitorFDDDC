# Setup Guide

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- PostgreSQL (optional)

## Backend Setup

1. Clone the repository
   ```bash
   git clone https://github.com/FARIDDD123/RockFluidMonitorFDDDC.git
   cd RockFluidMonitorFDDDC
   ```

2. Set up a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend
   ```bash
   cd backend
   python run.py
   ```
   
   The API will be available at http://127.0.0.1:5000

## Frontend Setup

1. Install dependencies
   ```bash
   cd frontend
   npm install
   ```

2. Run the development server
   ```bash
   npm start
   ```
   
   The frontend will be available at http://localhost:3000 
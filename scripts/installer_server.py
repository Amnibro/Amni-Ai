import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
import uvicorn
from amni.serve import model_installer
app=FastAPI(title='Adam Model Installer (standalone)')
model_installer.mount(app)
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=7790,log_level='warning')

import sys,asyncio
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from amni.serve import serve_policy as sp
app=FastAPI()
@app.get('/healthz')
def healthz():return {'ok':True}
@app.get('/api/ping')
def ping():return {'pong':True}
@app.post('/chat')
async def chat(request:Request):return {'ok':True}
@app.post('/chat/stream')
async def chat_stream(request:Request):
    async def g():
        yield 'event: token\ndata: "hi"\n\n';await asyncio.sleep(0)
    return StreamingResponse(g(),media_type='text/event-stream')
@app.post('/skills/{name}')
async def skill(name:str):return {'skill':name}
print('POLICY',sp.install(app,cors=True,token=sp.load_token()),flush=True)
uvicorn.run(app,host='127.0.0.1',port=int(sys.argv[1]),log_level='info')

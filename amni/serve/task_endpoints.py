"""Task HTTP endpoints for the /jarvis floating task tray.
GET  /tasks                — { active[], recent[], stats }
GET  /tasks/<id>           — one task detail
POST /tasks/<id>/cancel    — set cancel flag (worker polls + breaks gracefully)"""
def mount(app,agent):
    from fastapi import HTTPException
    reg=getattr(agent,'task_registry',None)
    @app.get('/tasks')
    def list_tasks():
        if reg is None:return {'active':[],'recent':[],'stats':{'active':0}}
        return {'active':reg.list_active(),'recent':reg.list_recent(limit=10),'stats':reg.stats()}
    @app.get('/tasks/{task_id}')
    def get_task(task_id:str):
        if reg is None:raise HTTPException(404,'task registry not initialized')
        t=reg.get(task_id)
        if t is None:raise HTTPException(404,f'task {task_id} not found')
        return t
    @app.post('/tasks/{task_id}/cancel')
    def cancel_task(task_id:str):
        if reg is None:raise HTTPException(404,'task registry not initialized')
        ok=reg.request_cancel(task_id)
        if not ok:raise HTTPException(404,f'task {task_id} not active')
        return {'task_id':task_id,'cancel_requested':True}

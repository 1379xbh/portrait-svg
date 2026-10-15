"""Local photo -> image model -> vision language model -> SVG + JSON API."""
import asyncio
import json
import mimetypes
import os
from pathlib import Path
import secrets
import time
from typing import Literal
from uuid import uuid4, UUID
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pipeline
from svg_io import svg_and_json, svg_metrics

ROOT = Path(__file__).resolve().parent
# Windows registry entries sometimes map .js to text/plain, blocking ES modules.
mimetypes.init()
mimetypes.add_type('text/javascript','.js')
mimetypes.add_type('text/css','.css')
mimetypes.add_type('image/svg+xml','.svg')
load_dotenv(ROOT/'.env')
OUTPUT = ROOT/'outputs'
app = FastAPI(title='AI Portrait SVG API',version='1.0.0')
# One costly request at a time in this single-worker local service.
busy = asyncio.Lock()


def auth(authorization: str = Header(default='')):
    token = os.getenv('SERVICE_API_KEY','')
    if token and not secrets.compare_digest(authorization, 'Bearer '+token):
        raise HTTPException(401,'Invalid API token')


@app.get('/health')
def health():
    return {'status':'ok'}


@app.get('/v1/ui-config')
def ui_config():
    configured={}
    for mode in ('local','full','lineart'):
        try:
            pipeline.check_settings(mode)
            configured[mode]=True
        except ValueError:
            configured[mode]=False
    return {'configured':configured,'auth_required':bool(os.getenv('SERVICE_API_KEY',''))}


@app.get('/v1/example/{filename}',dependencies=[Depends(auth)])
def example(filename: Literal['lineart.png','portrait.svg','portrait.json']):
    if not (ROOT/'examples'/filename).is_file(): raise HTTPException(404,'No bundled examples')
    return FileResponse(ROOT/'examples'/filename,filename=filename,
                        headers={'X-Content-Type-Options':'nosniff'})


@app.post('/v1/portraits',dependencies=[Depends(auth)])
async def portrait(image: UploadFile=File(...),mode: Literal['full','lineart','local']=Form('full'),
                   local_detail: Literal['faithful','clean','minimal','contour']=Form('clean'),
                   remove_background: bool=Form(True),simplify_details: bool=Form(True),
                   detail_budget: int=Form(24,ge=0,le=200),
                   fit_error: float=Form(.9,ge=.1,le=5)):
    """full: image API then local SVG; lineart: local SVG only; local: offline photo pipeline. All export JSON."""
    try:
        pipeline.check_settings(mode)
    except ValueError as exc:
        raise HTTPException(503,str(exc)) from exc
    if busy.locked():
        raise HTTPException(429,'Another portrait is processing; try after it finishes')
    async with busy:
        data = await image.read(12*1024*1024+1)
        await image.close()
        if len(data)>12*1024*1024:
            raise HTTPException(413,'Image exceeds 12 MB')
        try:
            photo = pipeline.normalized_png(data)
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        job_id = uuid4().hex
        folder = OUTPUT/job_id
        folder.mkdir(parents=True)
        (folder/'input.png').write_bytes(photo)
        timings={}
        started=time.perf_counter()
        checkpoint=started
        stage='local' if mode=='local' else ('image' if mode=='full' else 'svg')
        try:
            if mode=='local':
                svg=await pipeline.generate_local(folder/'input.png',folder,local_detail,remove_background,detail_budget,fit_error)
                timings['local_seconds']=round(time.perf_counter()-checkpoint,3)
            else:
                if mode=='full':
                    await pipeline.generate_lineart(folder/'input.png',folder/'lineart.png')
                    timings['image_seconds']=round(time.perf_counter()-checkpoint,3)
                else:
                    (folder/'lineart.png').write_bytes(photo)
                stage='local-vectorize'
                checkpoint=time.perf_counter()
                svg=await pipeline.vectorize_lineart(folder/'lineart.png',folder,simplify_details)
                timings['vectorize_seconds']=round(time.perf_counter()-checkpoint,3)
            stage='svg-json'
            checkpoint=time.perf_counter()
            svg, document=svg_and_json(svg)
            (folder/'portrait.svg').write_text(svg,encoding='utf-8')
            (folder/'portrait.json').write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
            timings['json_seconds']=round(time.perf_counter()-checkpoint,3)
            timings['total_seconds']=round(time.perf_counter()-started,3)
            (folder/'timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
        except Exception as exc:
            (folder/'diagnostic.json').write_text(json.dumps({'stage':stage,'error_type':type(exc).__name__,
                'elapsed_seconds':round(time.perf_counter()-started,3),'timings':timings},indent=2),encoding='utf-8')
            # Preserve lineart for retry via mode=lineart without paying to regenerate it.
            code=504 if isinstance(exc,(asyncio.TimeoutError,TimeoutError)) else 502
            hint='local dependencies, model files or face detection' if mode=='local' else 'provider/model settings or generated SVG validity'
            raise HTTPException(code,{'message':f'{stage} stage failed; check {hint}',
                'job_id':job_id,'lineart_url':f'/v1/portraits/{job_id}/lineart.png' if (folder/'lineart.png').exists() else None}) from exc
        return {'id':job_id,'mode':mode,'svg':svg,'document':document,'timings':timings,
                'metrics':svg_metrics(svg),
                'files':{name:f'/v1/portraits/{job_id}/{name}' for name in ('lineart.png','portrait.svg','portrait.json')}}


@app.get('/v1/portraits/{job_id}/{filename}',dependencies=[Depends(auth)])
def artifact(job_id: str,filename: Literal['lineart.png','portrait.svg','portrait.json']):
    try:
        if UUID(job_id).hex != job_id: raise ValueError()
    except ValueError:
        raise HTTPException(404,'Not found')
    path=OUTPUT/job_id/filename
    if not path.is_file(): raise HTTPException(404,'Not found')
    return FileResponse(path,filename=filename,headers={'X-Content-Type-Options':'nosniff','Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'"})


class SVGInput(BaseModel):
    svg: str=Field(max_length=1_000_000)


@app.post('/v1/svg/to-json',dependencies=[Depends(auth)])
def to_json(body: SVGInput):
    try:
        _, document=svg_and_json(body.svg)
        return document
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc


# Serve only the compiled frontend directory, never project files or credentials.
WEB=ROOT/'web'/'dist'
if WEB.is_dir():
    app.mount('/',StaticFiles(directory=WEB,html=True),name='web')

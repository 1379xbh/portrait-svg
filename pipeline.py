import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
import httpx
from PIL import Image, ImageOps
from openai import AsyncOpenAI
from svg_io import svg_and_json

ROOT = Path(__file__).resolve().parent


def settings(kind):
    return {
        'api_key': os.getenv(f'{kind}_API_KEY') or os.getenv('OPENAI_API_KEY',''),
        'base_url': os.getenv(f'{kind}_BASE_URL') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
        'model': os.getenv(f'{kind}_MODEL',''),
        'provider': os.getenv(f'{kind}_PROVIDER','openai'),
    }


def check_settings(mode):
    if not local_python().is_file():
        raise ValueError('Install local_pipeline/.venv or set LOCAL_PYTHON')
    if mode == 'local':
        if not local_python().is_file():
            raise ValueError('Install local_pipeline/.venv or set LOCAL_PYTHON')
        for name in ('models/informative/style2.pth','models/face_landmarker.task','models/selfie_multiclass_256x256.tflite'):
            if not (ROOT/'local_pipeline'/name).is_file():
                raise ValueError(f'Local model missing: {name}')
        return
    if mode=='lineart':
        return
    for kind in ('IMAGE',):
        conf = settings(kind)
        if conf['provider'] not in ('openai','siliconflow','qianwen'):
            raise ValueError(f'Unsupported {kind}_PROVIDER')
        if not conf['api_key'] or not conf['model']:
            raise ValueError(f'Configure {kind}_MODEL and {kind}_API_KEY (or OPENAI_API_KEY)')


async def vectorize_lineart(source,folder,simplify_details=True):
    """Local boundary tracing; no language model, preserves filled strokes and holes."""
    output=folder/'traced.svg'
    command=[str(local_python()),str(ROOT/'local_pipeline/vectorize_ink.py'),str(source),str(output)]
    if simplify_details:command.append('--simplify')
    proc=await asyncio.create_subprocess_exec(*command,
            stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.PIPE)
    try:
        _,_=await asyncio.wait_for(proc.communicate(),float(os.getenv('LOCAL_TIMEOUT_SECONDS','180')))
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError('Local vectorization failed; check LOCAL_PYTHON dependencies')
    return output.read_text(encoding='utf-8')


def local_python():
    configured = os.getenv('LOCAL_PYTHON','').strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else ROOT/path
    return ROOT/'local_pipeline'/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


async def generate_local(source, folder, detail='clean', remove_background=True, detail_budget=24, fit_error=.9):
    """Offline subprocess isolates Torch/MediaPipe from the API environment."""
    command=[str(local_python()),str(ROOT/'local_pipeline/informative.py'),str(source),
             '-o',str(folder/'local_result.svg'),'--device','cpu','--style','style2',
             '--detail',detail,'--size','800','--line-width','1.1',
             '--detail-budget',str(detail_budget),'--fit-error',str(fit_error)]
    if remove_background:
        command.append('--remove-background')
    proc=await asyncio.create_subprocess_exec(*command,stdout=asyncio.subprocess.DEVNULL,
                                             stderr=asyncio.subprocess.PIPE,cwd=str(ROOT/'local_pipeline'))
    try:
        _,err=await asyncio.wait_for(proc.communicate(),float(os.getenv('LOCAL_TIMEOUT_SECONDS','180')))
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError('Local pipeline failed; check dependencies or face detection')
    (folder/'lineart.png').write_bytes((folder/'local_result_model.png').read_bytes())
    return (folder/'local_result.svg').read_text(encoding='utf-8')


def normalized_png(data):
    try:
        with Image.open(BytesIO(data)) as im:
            if im.format not in ('JPEG','PNG','WEBP') or im.width*im.height > 24_000_000:
                raise ValueError('Use JPEG/PNG/WebP up to 24 megapixels')
            im = ImageOps.exif_transpose(im).convert('RGBA')
            im.thumbnail((1600,1600),Image.Resampling.LANCZOS)
            bg = Image.new('RGBA',im.size,'white')
            bg.alpha_composite(im)
            out = BytesIO()
            bg.convert('RGB').save(out,format='PNG')
            return out.getvalue()
    except Exception as exc:
        raise ValueError('Invalid image: JPEG/PNG/WebP, at most 24 megapixels') from exc


async def generate_lineart(source, output):
    """Use the unmodified bundled imagegen CLI for the explicit API image stage."""
    conf = settings('IMAGE')
    if conf['provider']=='qianwen':
        await generate_qianwen_lineart(source,output,conf)
        return
    if conf['provider']=='siliconflow':
        await generate_siliconflow_lineart(source,output,conf)
        return
    env = dict(os.environ, OPENAI_API_KEY=conf['api_key'], OPENAI_BASE_URL=conf['base_url'])
    command = [sys.executable, str(ROOT/'vendor/image_gen.py'), 'edit',
               '--image', str(source), '--out', str(output), '--model', conf['model'],
               '--prompt-file', str(ROOT/'prompts/lineart.txt'), '--no-augment',
               '--quality', os.getenv('IMAGE_QUALITY','medium'), '--size','auto', '--output-format','png']
    proc = await asyncio.create_subprocess_exec(*command, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(),float(os.getenv('MODEL_TIMEOUT_SECONDS','300')))
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    if proc.returncode or not output.is_file():
        # Provider stderr may contain credentials/URLs. Do not forward it to clients.
        raise RuntimeError('Image API failed; check model access, provider configuration and balance')
    output.write_bytes(normalized_png(output.read_bytes()))


async def generate_siliconflow_lineart(source,output,conf):
    payload={'model':conf['model'],
             'prompt':(ROOT/'prompts/lineart.txt').read_text(encoding='utf-8'),
             'image':'data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode(),
             'num_inference_steps':20}
    timeout=float(os.getenv('MODEL_TIMEOUT_SECONDS','300'))
    async with httpx.AsyncClient(timeout=timeout,follow_redirects=False) as client:
        # Qwen Image Edit does not accept image_size. Keep provider watermark default.
        response=await client.post(conf['base_url'].rstrip('/')+'/images/generations',
                    headers={'Authorization':'Bearer '+conf['api_key']},json=payload)
        response.raise_for_status()
        images=response.json().get('images',[])
        if not images or not images[0].get('url'):
            raise RuntimeError('Image API returned no image URL')
        await download_generated_image(client,images[0]['url'],output,
            ('aliyuncs.com','siliconflow.cn','siliconflow.com'))


async def generate_qianwen_lineart(source,output,conf):
    """Qwen Image 3.0 OpenAI-compatible JSON image editing (not multipart edits)."""
    image=source.read_bytes()
    if len(image)>10*1024*1024:
        raise ValueError('Qianwen input image exceeds 10 MB')
    payload={'model':conf['model'],
             'prompt':(ROOT/'prompts/lineart.txt').read_text(encoding='utf-8'),
             'image':'data:image/png;base64,'+base64.b64encode(image).decode(),
             'n':1,'size':'auto','prompt_extend':False}
    timeout=float(os.getenv('MODEL_TIMEOUT_SECONDS','600'))
    async with httpx.AsyncClient(timeout=timeout,follow_redirects=False) as client:
        response=await client.post(conf['base_url'].rstrip('/')+'/images/generations',
            headers={'Authorization':'Bearer '+conf['api_key']},json=payload)
        response.raise_for_status()
        images=response.json().get('data',[])
        if not images or not images[0].get('url'):
            raise RuntimeError('Qianwen image API returned no image URL')
        await download_generated_image(client,images[0]['url'],output,('aliyuncs.com',))


async def download_generated_image(client,url,output,allowed_domains):
    parsed=urlparse(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError('Invalid generated image URL')
    if not any(parsed.hostname.endswith('.'+domain) or parsed.hostname==domain
               for domain in allowed_domains):
        raise RuntimeError('Unexpected image download host')
    # Authorization is attached only to the generation POST, never storage GETs.
    data=bytearray()
    async with client.stream('GET',url) as download:
        download.raise_for_status()
        async for chunk in download.aiter_bytes():
            data.extend(chunk)
            if len(data)>32*1024*1024:
                raise RuntimeError('Generated image exceeds 32 MB')
    output.write_bytes(normalized_png(bytes(data)))


async def draw_svg(artwork, photo=None):
    conf = settings('LANGUAGE')
    content = [{'type':'input_text','text':(ROOT/'prompts/svg.txt').read_text(encoding='utf-8')}]
    for data in ([artwork,photo] if photo is not None else [artwork]):
        content.append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(data).decode(),'detail':'high'})
    async with AsyncOpenAI(api_key=conf['api_key'],base_url=conf['base_url'],
                          timeout=float(os.getenv('MODEL_TIMEOUT_SECONDS','300')),max_retries=0) as client:
        if conf['provider']=='siliconflow':
            chat_content=[{'type':'text','text':content[0]['text']}]
            for item in content[1:]:
                chat_content.append({'type':'image_url','image_url':{'url':item['image_url'],'detail':'high'}})
            response=await client.chat.completions.create(model=conf['model'],
                        messages=[{'role':'user','content':chat_content}],max_tokens=14000)
            if not response.choices or response.choices[0].finish_reason!='stop':
                raise RuntimeError('Language model did not finish SVG generation')
            return svg_and_json(response.choices[0].message.content or '')
        response = await client.responses.create(model=conf['model'],
                    input=[{'role':'user','content':content}],max_output_tokens=14000,store=False)
    if response.status != 'completed':
        raise RuntimeError('Language model did not finish SVG generation')
    return svg_and_json(response.output_text)

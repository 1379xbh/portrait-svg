import asyncio
from io import BytesIO
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import app as service
import pipeline
from svg_io import svg_and_json

SVG='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><g stroke="#111" fill="none"><path d="M10 10 C20 30 40 30 60 10"/></g></svg>'


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.output=patch.object(service,'OUTPUT',Path(self.temp.name)); self.output.start()
        self.env=patch.dict(os.environ,{'OPENAI_API_KEY':'mock-key','LANGUAGE_MODEL':'mock-vision',
            'IMAGE_MODEL':'gpt-image-2','SERVICE_API_KEY':'local-test-token','LOCAL_PYTHON':sys.executable},clear=True); self.env.start()
        self.client=TestClient(service.app)
        self.headers={'Authorization':'Bearer local-test-token'}
        buff=BytesIO(); Image.new('RGB',(40,50),'white').save(buff,format='PNG'); self.png=buff.getvalue()

    def tearDown(self):
        self.client.close(); self.env.stop(); self.output.stop(); self.temp.cleanup()

    def post(self,mode='full',data=None):
        return self.client.post('/v1/portraits',headers=self.headers,data={'mode':mode},files={'image':('portrait.png',data or self.png,'image/png')})

    def test_json_preserves_paths_and_inherited_styles(self):
        r=self.client.post('/v1/svg/to-json',headers=self.headers,json={'svg':SVG})
        self.assertEqual(r.status_code,200)
        group=r.json()['svg']['children'][0]
        self.assertEqual(group['attributes']['stroke'],'#111')
        self.assertEqual(group['children'][0]['attributes']['d'],'M10 10 C20 30 40 30 60 10')

    def test_unsafe_and_broken_svg_rejected(self):
        for bad in [SVG.replace('<g ','<script/><g '),SVG.replace('stroke="#111"','onclick="evil()"'),
                    SVG.replace('#111','url(https://evil.test)'), SVG.replace('M10 10 C20 30 40 30 60 10','nonsense'),
                    '<!DOCTYPE svg [<!ENTITY x "abc">]>'+SVG]:
            with self.subTest(bad=bad):
                self.assertEqual(self.client.post('/v1/svg/to-json',headers=self.headers,json={'svg':bad}).status_code,400)

    def test_auth(self):
        self.assertEqual(self.client.post('/v1/svg/to-json',json={'svg':SVG}).status_code,401)

    def test_web_and_config_no_secret_exposure(self):
        response=self.client.get('/')
        self.assertEqual(response.status_code,200)
        self.assertIn('<div id="root">',response.text)
        import re
        script=re.search(r'src="([^"]+\.js)"',response.text)[1]
        self.assertIn('javascript',self.client.get(script).headers['content-type'])
        config=self.client.get('/v1/ui-config')
        self.assertEqual(config.status_code,200)
        self.assertTrue(config.json()['auth_required'])
        self.assertNotIn('mock-key',config.text)
        self.assertEqual(self.client.get('/.env').status_code,404)
        self.assertEqual(self.client.get('/v1/example/portrait.svg').status_code,401)
        self.assertEqual(self.client.get('/v1/example/portrait.svg',headers=self.headers).status_code,404)

    def test_full_pipeline_and_download(self):
        async def image_stage(source,output): output.write_bytes(self.png)
        with patch.object(pipeline,'generate_lineart',side_effect=image_stage) as im, patch.object(pipeline,'vectorize_lineart',new_callable=AsyncMock,return_value=SVG) as vector, patch.object(pipeline,'draw_svg',new_callable=AsyncMock) as lang:
            response=self.post()
            self.assertEqual(response.status_code,200,response.text)
            im.assert_called_once(); vector.assert_awaited_once(); lang.assert_not_awaited()
            self.assertIs(vector.await_args.args[2],True)
            result=response.json()
            self.assertEqual(self.client.get(result['files']['portrait.svg'],headers=self.headers).status_code,200)
            self.assertEqual(self.client.get(result['files']['portrait.json'],headers=self.headers).json(),result['document'])
            self.assertFalse('mock-key' in response.text)

    def test_lineart_skips_image_model(self):
        with patch.object(pipeline,'generate_lineart',new_callable=AsyncMock) as im, patch.object(pipeline,'vectorize_lineart',new_callable=AsyncMock,return_value=SVG), patch.object(pipeline,'draw_svg',new_callable=AsyncMock) as lang, patch.dict(os.environ,{'OPENAI_API_KEY':'','LANGUAGE_MODEL':''}):
            self.assertEqual(self.post('lineart').status_code,200)
            im.assert_not_awaited(); lang.assert_not_awaited()

    def test_bad_upload_and_missing_config(self):
        self.assertEqual(self.post(data=b'not an image').status_code,400)
        with patch.dict(os.environ,{'IMAGE_MODEL':''}): self.assertEqual(self.post().status_code,503)

    def test_detail_simplification_can_be_disabled(self):
        with patch.object(pipeline,'vectorize_lineart',new_callable=AsyncMock,return_value=SVG) as vector:
            response=self.client.post('/v1/portraits',headers=self.headers,
                data={'mode':'lineart','simplify_details':'false'},
                files={'image':('portrait.png',self.png,'image/png')})
            self.assertEqual(response.status_code,200,response.text)
            self.assertIs(vector.await_args.args[2],False)

    def test_failed_vectorization_preserves_artwork(self):
        with patch.object(pipeline,'vectorize_lineart',new_callable=AsyncMock,side_effect=RuntimeError('secret upstream detail')):
            response=self.post('lineart')
            self.assertEqual(response.status_code,502)
            self.assertNotIn('secret upstream detail',response.text)
            link=response.json()['detail']['lineart_url']
            self.assertEqual(self.client.get(link,headers=self.headers).status_code,200)

    @unittest.skip("Image-free distribution does not bundle examples")
    def test_existing_example_is_valid(self):
        text=(Path(__file__).parent/'examples/portrait.svg').read_text(encoding='utf-8')
        svg,doc=svg_and_json(text)
        self.assertIn('viewBox',doc['svg']['attributes'])

    def test_local_without_keys_returns_same_json_contract(self):
        async def local_stage(source,folder,detail,remove_background,detail_budget,fit_error):
            (folder/'lineart.png').write_bytes(self.png)
            return SVG
        with patch.dict(os.environ,{'OPENAI_API_KEY':'','LANGUAGE_MODEL':'','LOCAL_PYTHON':sys.executable}), \
             patch.object(pipeline,'generate_local',side_effect=local_stage) as local, \
             patch.object(pipeline,'draw_svg',new_callable=AsyncMock) as language, \
             patch.object(pipeline,'generate_lineart',new_callable=AsyncMock) as image:
            response=self.client.post('/v1/portraits',headers=self.headers,
                data={'mode':'local','local_detail':'clean','remove_background':'false'},
                files={'image':('portrait.png',self.png,'image/png')})
            self.assertEqual(response.status_code,200,response.text)
            result=response.json()
            self.assertEqual(result['mode'],'local')
            self.assertEqual(result['document'],svg_and_json(result['svg'])[1])
            self.assertEqual(local.call_args.args[2:],('clean',False,24,.9))
            image.assert_not_awaited(); language.assert_not_awaited()

    def test_local_clip_paths_survive_json(self):
        svg=SVG.replace('<g stroke=', '<defs><clipPath id="eye"><path d="M0 0L10 0L10 10Z"/></clipPath></defs><g clip-path="url(#eye)" data-part="eye" stroke=')
        _,document=svg_and_json(svg)
        self.assertEqual(document['svg']['children'][1]['attributes']['clip-path'],'url(#eye)')
        for bad in (svg.replace('url(#eye)','url(https://evil.test/eye)'),svg.replace('url(#eye)','url(#missing)')):
            with self.assertRaises(ValueError): svg_and_json(bad)

    def test_budget_parameters_and_default_style(self):
        with patch.object(pipeline, 'generate_local', new_callable=AsyncMock, return_value=SVG) as local:
            response = self.client.post('/v1/portraits', headers=self.headers,
                data={'mode': 'local', 'detail_budget': '7', 'fit_error': '0.5'},
                files={'image': ('portrait.png', self.png, 'image/png')})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(local.await_args.args[2:], ('clean', True, 7, .5))
            self.assertEqual(response.json()['metrics']['visible_strokes'], 1)
            for field, value in [('detail_budget', '-1'), ('detail_budget', '201'),
                                 ('fit_error', '0'), ('fit_error', 'nan')]:
                bad = self.client.post('/v1/portraits', headers=self.headers,
                    data={'mode': 'local', field: value},
                    files={'image': ('portrait.png', self.png, 'image/png')})
                self.assertEqual(bad.status_code, 422)

    def test_missing_local_interpreter_and_invalid_mode(self):
        with patch.dict(os.environ,{'LOCAL_PYTHON':str(Path(self.temp.name)/'missing.exe')}):
            self.assertEqual(self.post('local').status_code,503)
        self.assertEqual(self.post('unknown').status_code,422)

    def test_language_request_contains_images_and_requires_completed(self):
        from types import SimpleNamespace
        mocked=AsyncMock()
        mocked.__aenter__.return_value=mocked
        mocked.responses.create.return_value=SimpleNamespace(status='completed',output_text=SVG)
        with patch.object(pipeline,'AsyncOpenAI',return_value=mocked):
            asyncio.run(pipeline.draw_svg(self.png,self.png))
        args=mocked.responses.create.call_args.kwargs
        self.assertEqual(len(args['input'][0]['content']),3)
        self.assertFalse(args['store'])

    def test_siliconflow_image_contract_and_no_key_on_download(self):
        import httpx
        requests=[]
        def handle(request):
            requests.append(request)
            if request.method=='POST':
                import json
                payload=json.loads(request.content)
                self.assertEqual(payload['model'],'Qwen/Qwen-Image-Edit-2509')
                self.assertTrue(payload['image'].startswith('data:image/png;base64,'))
                self.assertNotIn('image_size',payload)
                return httpx.Response(200,json={'images':[{'url':'https://test.oss-cn-shanghai.aliyuncs.com/result.png'}]})
            self.assertNotIn('authorization',request.headers)
            return httpx.Response(200,content=self.png)
        source=Path(self.temp.name)/'input.png'; source.write_bytes(self.png)
        output=Path(self.temp.name)/'output.png'
        real_client=httpx.AsyncClient
        with patch.dict(os.environ,{'IMAGE_PROVIDER':'siliconflow','IMAGE_MODEL':'Qwen/Qwen-Image-Edit-2509','OPENAI_BASE_URL':'https://api.siliconflow.cn/v1'}), \
             patch.object(pipeline.httpx,'AsyncClient',side_effect=lambda **kw:real_client(transport=httpx.MockTransport(handle),**kw)):
            asyncio.run(pipeline.generate_lineart(source,output))
        self.assertEqual(len(requests),2)
        self.assertEqual(requests[0].url.path,'/v1/images/generations')
        with Image.open(output) as result_image:
            self.assertEqual(result_image.size,(40,50))

    def test_siliconflow_vision_uses_chat_and_rejects_truncation(self):
        from types import SimpleNamespace
        mocked=AsyncMock(); mocked.__aenter__.return_value=mocked
        choice=SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=SVG))
        mocked.chat.completions.create.return_value=SimpleNamespace(choices=[choice])
        with patch.dict(os.environ,{'LANGUAGE_PROVIDER':'siliconflow'}),patch.object(pipeline,'AsyncOpenAI',return_value=mocked):
            svg,_=asyncio.run(pipeline.draw_svg(self.png,self.png))
            self.assertIn('<svg',svg)
            args=mocked.chat.completions.create.call_args.kwargs
            self.assertEqual(args['messages'][0]['content'][1]['type'],'image_url')
            mocked.responses.create.assert_not_awaited()
            choice.finish_reason='length'
            with self.assertRaises(RuntimeError):asyncio.run(pipeline.draw_svg(self.png))

    def test_qianwen_image_contract_and_storage_has_no_key(self):
        import httpx
        import json
        requests = []
        def handle(request):
            requests.append(request)
            if request.method == 'POST':
                payload = json.loads(request.content)
                self.assertEqual(payload['model'], 'qwen-image-3.0-pro')
                self.assertEqual(payload['n'], 1)
                self.assertEqual(payload['size'], 'auto')
                self.assertFalse(payload['prompt_extend'])
                self.assertTrue(payload['image'].startswith('data:image/png;base64,'))
                self.assertNotIn('num_inference_steps', payload)
                self.assertEqual(request.headers['authorization'], 'Bearer mock-key')
                return httpx.Response(200, json={'data': [{'url': 'https://dashscope-result.oss-cn-shanghai.aliyuncs.com/result.png'}]})
            self.assertNotIn('authorization', request.headers)
            return httpx.Response(200, content=self.png)
        source = Path(self.temp.name) / 'input.png'; source.write_bytes(self.png)
        output = Path(self.temp.name) / 'output.png'
        real_client = httpx.AsyncClient
        with patch.dict(os.environ, {'IMAGE_PROVIDER': 'qianwen', 'IMAGE_MODEL': 'qwen-image-3.0-pro',
             'OPENAI_BASE_URL': 'https://maas.qianwenaiapi.com/compatible-mode/v1'}), \
             patch.object(pipeline.httpx, 'AsyncClient', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)):
            pipeline.check_settings('full')
            asyncio.run(pipeline.generate_lineart(source, output))
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].url.path, '/compatible-mode/v1/images/generations')
        with Image.open(output) as im:
            self.assertEqual(im.size, (40, 50))

    def test_qianwen_rejects_unexpected_storage_host(self):
        import httpx
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={'data': [{'url': 'https://evil.test/result.png'}]})
        source = Path(self.temp.name) / 'input.png'; source.write_bytes(self.png)
        real_client = httpx.AsyncClient
        with patch.object(pipeline.httpx, 'AsyncClient', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected image download host'):
                asyncio.run(pipeline.generate_qianwen_lineart(source, Path(self.temp.name)/'output.png',
                    {'base_url': 'https://maas.qianwenaiapi.com/compatible-mode/v1',
                     'model': 'qwen-image-3.0-pro', 'api_key': 'mock-key'}))
        self.assertEqual(len(requests), 1)


if __name__=='__main__': unittest.main()

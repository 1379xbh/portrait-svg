# 几笔：照片 / 线稿 → SVG → JSON

React + FastAPI 人像线稿工作台，支持纯本地处理、AI 生成线稿、已有线稿矢量化及 SVG/JSON 导出。包含离线模型权重及其来源说明；实际密钥、照片、运行结果和虚拟环境不提交到仓库。

## Linux 快速启动

建议使用 Python 3.12，为 API 和本地算法分别创建环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.12 -m venv local_pipeline/.venv
local_pipeline/.venv/bin/python -m pip install -r local_pipeline/requirements-cpu.txt
cp .env.example .env
bash start.sh
```

打开 http://127.0.0.1:8000/ 。`.env` 中 `LOCAL_PYTHON` 留空即可使用项目内的本地环境。纯本地和已有线稿模式不需要远程 API 密钥；AI 绘制需填写对应平台密钥。CPU 依赖清单最初面向 Windows 整理，其他平台安装后应运行测试并验证模型推理。

Windows 启动与各模式详细说明见下文。当前配置示例使用千问AI平台，硅基流动与 OpenAI 接入仍保留。

## 当前流程

所有网页入口都在本地生成 SVG，不再调用语言模型绘制 SVG。

| mode | 网页名称 | 流程 | 远程调用 |
|---|---|---|---|
| full | AI 绘制 | 照片 → 图像模型线稿 → 本地矢量化 → SVG → JSON | 仅图像生成 |
| lineart | 已有线稿 | 黑白线稿 → 本地矢量化 → SVG → JSON | 无 |
| local | 纯本地 | 照片 → CPU 分割/线稿模型 → 补线/拟合 → SVG → JSON | 无 |

本地矢量化描摹黑线边界，以平滑曲线保留线宽与内部孔洞，输出封闭填充路径，不是单线描边。它会过滤微小孤立斑点，但不会理解并重画五官，也不会自动删除所有多余发丝。

## 启动网页

在本目录运行 `start.ps1`，然后打开 http://127.0.0.1:8000/ 。API 文档：http://127.0.0.1:8000/docs 。
预编译网页位于 web/dist，无需启动 Node 服务。默认只监听本机、单 worker，一次处理一张图。

首次安装 API 环境（Python 3.12）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

本地算法在独立 Python 环境中执行。本机 `.env` 的 LOCAL_PYTHON 已指向之前可用的环境；复制到其他电脑时，可安装随包的 CPU 依赖：

```powershell
py -3.12 -m venv local_pipeline/.venv
.\local_pipeline\.venv\Scripts\python.exe -m pip install -r local_pipeline/requirements-cpu.txt
```

安装后 LOCAL_PYTHON 留空，或设为该环境的绝对解释器路径。已有线稿的矢量化使用 OpenCV、NumPy、SciPy 和 Pillow，不运行 Torch 模型；纯本地照片流程还会使用附带权重。实际测试复用原环境并强制 CPU，未新建安装 CPU-only 环境。

## 模型配置

硅基流动已配置在本机 .env，密钥不存入前端或示例文件。

```dotenv
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_API_KEY=在本地填写密钥
IMAGE_PROVIDER=siliconflow
IMAGE_MODEL=Qwen/Qwen-Image-Edit-2509
```

当前网页流程无需 LANGUAGE_MODEL / LANGUAGE_API_KEY；旧配置和 draw_svg 函数仅保留作兼容参考，路由不会调用它。
OpenAI 图像服务可设置 IMAGE_PROVIDER=openai，填写对应地址、密钥和模型名，使用附带原版 vendor/image_gen.py。
图像生成才会上传照片、产生服务商费用；lineart/local 不调用远程模型。
硅基流动使用 /images/generations，保留平台默认生成标识；下载返回图片时不发送 API Key。允许的图片下载域为 aliyuncs.com、siliconflow.cn、siliconflow.com 及其子域。

SERVICE_API_KEY 为可选的本地 API Bearer 令牌，可在网页「服务连接」填写。不要在网页填写图像模型密钥。

## API 示例

```powershell
# 照片 → AI 出图 → 本地 SVG + JSON
curl.exe --max-time 540 -X POST http://127.0.0.1:8000/v1/portraits -F "image=@D:/photos/portrait.jpg" -F "mode=full" -o result.json

# 已有线稿 → 本地 SVG + JSON，无远程费用
curl.exe --max-time 240 -X POST http://127.0.0.1:8000/v1/portraits -F "image=@D:/photos/lineart.png" -F "mode=lineart" -o result.json

# 照片 → 全部本地处理
curl.exe --max-time 240 -X POST http://127.0.0.1:8000/v1/portraits -F "image=@D:/photos/portrait.jpg" -F "mode=local" -F "local_detail=clean" -F "remove_background=true" -o result.json
```

mode=local 支持 local_detail=clean/contour/faithful/minimal，remove_background=true/false。戴眼镜时 contour 仍可能丢镜框，建议 clean 对照。
支持 JPG/PNG/WebP，最大 12 MB / 2400 万像素，发送处理前转正、合成白底、长边缩至 1600。

响应包含 id、mode、svg、document（SVG 元素树）、files（下载地址）和 timings（阶段耗时）。
所有模式都输出 portrait.svg 和 portrait.json；local_result.json 为本地算法指标，不是矢量结构。
outputs/<id>/ 保留图片、SVG、JSON、timings.json；失败时保存不含密钥的 diagnostic.json。
失败响应包含已生成线稿地址，下载后可通过 mode=lineart 转换，不必重复生成图片。

已有 SVG 可 POST /v1/svg/to-json，请求体为 {"svg":"<svg ...>...</svg>"}，或离线运行：

```powershell
.\.venv\Scripts\python.exe export_svg_json.py D:/vectors/portrait.svg -o outputs/portrait.json
```

JSON 保留 tag、attributes、children 和 text，遵循分组样式继承。SVG 支持基本绘图和内部 clipPath，不支持位图、脚本、外链或 CSS。

## 文件及验证

- app.py：网页托管、API、输出和计时。
- pipeline.py：图像模型/本地处理调度。
- local_pipeline/vectorize_ink.py：边界描摹、去小点与曲线拟合。
- svg_io.py / export_svg_json.py：SVG 验证和 JSON 导出。
- local_pipeline/models：离线照片处理的附带权重。
- prompts/lineart.txt：AI 线稿要求。
- web/src：React 前端；修改后在 web 目录执行 npm ci、npm run build。
- 本整理版不附带照片、示例 SVG/JSON、运行结果、虚拟环境或缓存。

测试：`.\.venv\Scripts\python.exe -m unittest -v test_api`。
实际已有 AI 线稿（880×1184）本地转 SVG 用时约 2.64 秒，含解释器启动；图片生成此前为 26.6 秒。二者是分次测量，不是整条流程的一次计时。效果更接近原始线稿，细节可能多于语言模型重绘。

## 默认细节简化

full / lineart 新增 `simplify_details=true`（默认），在本地矢量化前用人脸关键点限定区域：简化眉毛轮廓、删除短碎发、合并中央衣物的细小封闭纹理；保留眼镜、主要五官和主体边界。不再只是逐条描摹所有黑线。设为 false 可保留原线稿细节。此选项不改变 local 照片模式。
需要附带的 MediaPipe 人脸关键点模型。未检测到人脸时保守保留原图；复杂姿势、遮挡或衣物可能不适合该启发式处理，可关闭比较。该算法不等同于人工理解每条线，也不保证对任意图片都能正确去纹理。


## 线条预算与贴合度优化

网页和 API 的本地默认风格统一为 `clean`。

- `detail_budget=24`：本地流程允许保留的次要线条上限，范围 0–200。五官、下颌、主体边界与结构连接受到保护，不计入该上限，因此它不是全部笔画数量上限。`faithful` 不使用预算。
- `fit_error=0.9`：本地曲线拟合误差，范围 0.1–5 像素，基于最长边不超过 800 的处理图像。越小越贴合；越大允许更多平滑。五官采用更严格误差，轮廓模式部分结构仍保持关键点约束。
- `clean/minimal` 在原有清理后，按线长和图像支持强度排序保留次要线条；未检测到人脸时保守保留，避免误删辨识特征。
- `contour` 使用有图像依据的下颌边缘校正；未检测到人脸或检测到疑似闭合镜框时回退到 `clean`。镜框检测是启发式规则，不能保证识别所有眼镜。检测到镜框时 `clean/minimal` 保护眼部与鼻梁连接区域中的原线稿。
- 预算和拟合参数目前仅作用于 `local`，`full/lineart` 继续使用原有填充边界描摹。

响应新增 `metrics`：`visible_paths`、`visible_strokes`（可见子路径与基本形状数量）、`curve_segments`、`segments`、`path_length`（仅可见 path 的长度，单位为 SVG 坐标）、`svg_bytes`。统计排除 defs/clipPath 和白色背景。填充边界子路径仍不等同于人工笔画；这些数据用于比较复杂度，不是人物相似度评分。

回归测试：`python -m unittest -v test_api test_geometry`。新增合成图测试涵盖预算、结构保护、镜框证据、无人脸回退及几何统计；真实照片的视觉质量仍需固定照片集对照验证。本次未调用远程图像服务。


## 千问AI平台接入

支持千问AI平台 Qwen Image 3.0 的 JSON 图像编辑接口，配置示例（不含密钥）：

```dotenv
OPENAI_API_KEY=你的千问AI平台密钥
OPENAI_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
IMAGE_PROVIDER=qianwen
IMAGE_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
IMAGE_MODEL=qwen-image-3.0-pro
MODEL_TIMEOUT_SECONDS=600
```

`full` 调用 `/images/generations`，传入参考图片 Base64 与线稿提示词，每次只生成一张，关闭提示词改写以保留线稿要求。解析 `data[].url`，从允许的阿里云存储域下载图像；下载不携带 API 密钥，再在本机转换 SVG。

修改 `.env` 后重启服务。模型列表认证通过只证明密钥有效及模型可见，实际图像生成还取决于账户额度、模型权限和输入图像条件。

接口文档：https://platform.qianwenai.com/docs/api-reference/image-generation/qwen-text-to-image-openai

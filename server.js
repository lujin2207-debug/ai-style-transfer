require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3000;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }
});

app.use(express.json({ limit: '80mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const getRequestApiKey = (req) => {
  const headerKey = req.headers['x-user-api-key'];
  const bodyKey = req.body?.apiKey;
  return (Array.isArray(headerKey) ? headerKey[0] : headerKey || bodyKey || '').trim();
};

const buildStyleTransferPrompt = (extra) => `你将收到两张图片。

第一张图片是画风参考图，只学习它的画风。
第二张图片是内容参考图，必须保持第二张图片的主体、人物身份、姿势、服装、构图、背景布局和物体关系。

请将第二张图片重新绘制成第一张图片的画风。

要求：
1. 只迁移第一张图片的画风，包括线条、上色方式、阴影、高光、皮肤质感、五官渲染、色彩氛围、镜头语言和细节处理方式。
2. 不要复制第一张图片的人物、服装、发型、饰品、姿势和构图。
3. 第二张图片的内容优先级最高，不能换主体、不能改变主要角色数量、不能改变姿势和物体关系。
4. 风格相似度优先于文字描述，不要只生成泛化的卡通/插画风。
5. 输出完整图片，不要文字、水印、边框。

用户补充要求：
${extra || '无'}`;

const buildEnhancePrompt = () => `你将收到一张图片作为参考。

请在完全保持原图内容不变的前提下增强画质。

要求：
1. 保持原图的主体、构图、姿势、服装、画风、颜色方向和背景布局不变。
2. 提升清晰度、边缘质量、细节层次、纹理表现和整体精致度。
3. 修复轻微模糊、压缩痕迹、噪点、锯齿和低分辨率观感。
4. 不要重新设计角色，不要改变五官、道具、文字位置、主体数量和画面比例。
5. 输出完整高清图片，不要添加水印、边框、额外文字。`;

const buildLayerPrompt = (layerType) => {
  const prompts = {
    subject: `你将收到一张图片作为参考。

请从原图中拆分出“主体图层”。

要求：
1. 只保留主要角色、主要物体和前景核心元素。
2. 移除背景、环境、地面和无关装饰。
3. 保持主体的画风、颜色、细节、边缘和姿势不变。
4. 如果支持透明背景，请输出透明 PNG；如果不支持透明，请使用纯白或纯黑背景，主体边缘要干净。
5. 不要添加新元素、文字、水印或边框。`,
    background: `你将收到一张图片作为参考。

请从原图中拆分出“背景图层”。

要求：
1. 移除主要角色、主要物体和前景核心元素。
2. 尽量补全被主体遮挡的背景区域，让背景看起来自然完整。
3. 保持原图背景的画风、色彩、光影和氛围。
4. 不要保留人物、角色、道具主体、文字、水印或边框。
5. 输出完整背景图片。`,
    lineart: `你将收到一张图片作为参考。

请从原图中拆分出“线稿/轮廓图层”。

要求：
1. 提取主要角色、物体和关键细节的清晰轮廓线。
2. 使用干净的黑色或深色线条，背景保持白色或透明。
3. 保留重要五官、服装、道具和外轮廓，不要生成灰度厚涂或完整上色图。
4. 不要添加新元素、文字、水印或边框。
5. 输出适合后期叠加使用的清晰线稿图。`
  };

  return prompts[layerType] || prompts.subject;
};

const fileToInlineData = (file) => {
  return {
    mimeType: file.mimetype || 'image/png',
    data: file.buffer.toString('base64')
  };
};

const base64ToInlineData = (dataUrl) => {
  const mimeMatch = dataUrl.match(/^data:(image\/[\w.+-]+);base64,/);
  const mime = mimeMatch?.[1] || 'image/png';
  const base64Data = dataUrl.replace(/^data:image\/[\w.+-]+;base64,/, '');
  return { mimeType: mime, data: base64Data };
};

const imageInputToInlineData = async (image) => {
  if (image.startsWith('data:image/')) return base64ToInlineData(image);

  if (/^https?:\/\//i.test(image)) {
    const response = await fetch(image);
    if (!response.ok) throw new Error(`无法读取远程图片：${response.status}`);

    const arrayBuffer = await response.arrayBuffer();
    const mimeType = response.headers.get('content-type') || 'image/png';
    return {
      mimeType,
      data: Buffer.from(arrayBuffer).toString('base64')
    };
  }

  throw new Error('图片格式不支持，请使用 base64 图片或可访问的图片 URL');
};

const inlineDataToDataURL = (inlineData) => {
  return `data:${inlineData.mimeType || 'image/png'};base64,${inlineData.data}`;
};

const getGeminiEndpoint = (baseURL, model) => {
  const geminiBaseURL = process.env.GEMINI_BASE_URL || baseURL.replace(/\/v1\/?$/, '/v1beta');
  return `${geminiBaseURL.replace(/\/$/, '')}/models/${encodeURIComponent(model)}:generateContent`;
};

const extractGeminiImage = (payload) => {
  const parts = payload?.candidates?.[0]?.content?.parts || [];
  const imagePart = parts.find((part) => part.inlineData?.data || part.inline_data?.data);
  const inlineData = imagePart?.inlineData || imagePart?.inline_data;
  if (!inlineData?.data) return null;

  const mimeType = inlineData.mimeType || inlineData.mime_type || 'image/png';
  return `data:${mimeType};base64,${inlineData.data}`;
};

const extractImageFromText = (text) => {
  if (!text) return null;

  const dataUrlMatch = text.match(/data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/);
  if (dataUrlMatch) return dataUrlMatch[0];

  const urlMatch = text.match(/https?:\/\/[^\s)"']+\.(?:png|jpe?g|webp)(?:\?[^\s)"']*)?/i);
  if (urlMatch) return urlMatch[0];

  return null;
};

const extractOpenAICompatibleImage = (payload) => {
  const choice = payload?.choices?.[0];
  const message = choice?.message || {};

  if (message.image_url?.url) return message.image_url.url;
  if (message.image?.url) return message.image.url;
  if (message.image?.b64_json) return `data:image/png;base64,${message.image.b64_json}`;

  if (Array.isArray(message.images) && message.images[0]) {
    const image = message.images[0];
    if (typeof image === 'string') return image;
    if (image.url) return image.url;
    if (image.image_url?.url) return image.image_url.url;
    if (image.b64_json) return `data:image/png;base64,${image.b64_json}`;
  }

  if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (part.type === 'image_url' && part.image_url?.url) return part.image_url.url;
      if (part.type === 'output_image' && part.image_url?.url) return part.image_url.url;
      if (part.inlineData?.data) return `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
      if (part.text) {
        const image = extractImageFromText(part.text);
        if (image) return image;
      }
    }
  }

  return extractImageFromText(typeof message.content === 'string' ? message.content : '');
};

const extractImagesGenerationImage = (payload) => {
  const first = payload?.data?.[0] || payload?.images?.[0];
  if (!first) return null;

  if (typeof first === 'string') return first;
  if (first.url) return first.url;
  if (first.image_url?.url) return first.image_url.url;
  if (first.b64_json) return `data:image/png;base64,${first.b64_json}`;
  if (first.base64) return `data:image/png;base64,${first.base64}`;
  if (first.inlineData?.data) return `data:${first.inlineData.mimeType || 'image/png'};base64,${first.inlineData.data}`;
  if (first.inline_data?.data) return `data:${first.inline_data.mime_type || 'image/png'};base64,${first.inline_data.data}`;

  return extractImageFromText(JSON.stringify(first));
};

const callOpenAICompatibleMultiImageEdit = async ({ apiKey, baseURL, model, prompt, styleInlineData, photoInlineData }) => {
  const endpoint = `${baseURL.replace(/\/$/, '')}/chat/completions`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: inlineDataToDataURL(styleInlineData) } },
            { type: 'image_url', image_url: { url: inlineDataToDataURL(photoInlineData) } }
          ]
        }
      ],
      modalities: ['text', 'image']
    })
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.message || `OpenAI兼容接口请求失败：${response.status}`;
    throw new Error(message);
  }

  const image = extractOpenAICompatibleImage(payload);
  if (!image) {
    const error = new Error('OpenAI兼容接口未返回图片数据');
    error.code = 'NO_IMAGE_FROM_CHAT';
    throw error;
  }

  return image;
};

const callImagesGenerationWithReferences = async ({ apiKey, baseURL, model, prompt, styleInlineData, photoInlineData }) => {
  const endpoint = `${baseURL.replace(/\/$/, '')}/images/generations`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      prompt,
      n: 1,
      resolution: process.env.IMAGE_RESOLUTION || '1K',
      size: process.env.IMAGE_SIZE || '1024x1024',
      response_format: 'b64_json',
      reference_images: [
        inlineDataToDataURL(styleInlineData),
        inlineDataToDataURL(photoInlineData)
      ]
    })
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.message || `images/generations接口请求失败：${response.status}`;
    throw new Error(message);
  }

  const image = extractImagesGenerationImage(payload);
  if (!image) {
    throw new Error('images/generations接口未返回图片数据，请确认该平台是否支持 reference_images 参数');
  }

  return image;
};

const callGeminiMultiImageEdit = async ({ apiKey, baseURL, model, prompt, styleInlineData, photoInlineData }) => {
  const endpoint = getGeminiEndpoint(baseURL, model);
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'x-goog-api-key': apiKey
    },
    body: JSON.stringify({
      contents: [
        {
          role: 'user',
          parts: [
            { text: prompt },
            { inlineData: styleInlineData },
            { inlineData: photoInlineData }
          ]
        }
      ],
      generationConfig: {
        responseModalities: ['IMAGE']
      }
    })
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.message || `Gemini接口请求失败：${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }

  const image = extractGeminiImage(payload);
  if (!image) {
    const text = payload?.candidates?.[0]?.content?.parts?.map((part) => part.text).filter(Boolean).join('\n');
    throw new Error(text || 'Gemini未返回图片数据');
  }

  return image;
};

const shouldUseGeminiNativeApi = (baseURL) => {
  return Boolean(process.env.GEMINI_BASE_URL) || /googleapis\.com/i.test(baseURL);
};

const callOpenAICompatibleImageFlow = async (params) => {
  try {
    return await callOpenAICompatibleMultiImageEdit(params);
  } catch (chatErr) {
    console.warn('⚠️ /chat/completions 未返回图片，回退到 /images/generations reference_images 模式');
    return callImagesGenerationWithReferences(params);
  }
};

const callMultiImageStyleTransfer = async (params) => {
  if (!shouldUseGeminiNativeApi(params.baseURL)) {
    console.log('🔁 使用 OpenAI 兼容多图模式');
    return callOpenAICompatibleImageFlow(params);
  }

  try {
    return await callGeminiMultiImageEdit(params);
  } catch (err) {
    if (err.status !== 404) throw err;

    console.warn('⚠️ Gemini原生接口404，回退到 /chat/completions 多图模式');
    return callOpenAICompatibleImageFlow(params);
  }
};

app.get('/api/health', (req, res) => {
  res.json({ ok: true });
});

// ── 真正的画风转换：风格参考图 + 内容图 + 固定任务指令 → 直接输出图片 ──
app.post('/api/style-transfer-image', upload.fields([
  { name: 'style', maxCount: 1 },
  { name: 'photo', maxCount: 1 }
]), async (req, res) => {
  try {
    const styleFile = req.files?.['style']?.[0];
    const photoFile = req.files?.['photo']?.[0];
    if (!styleFile || !photoFile) return res.status(400).json({ error: '请上传画风参考图和要转换的图片' });

    const apiKey = getRequestApiKey(req);
    const baseURL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
    const imageModel = process.env.IMAGE_MODEL || 'gpt-image-2';
    if (!apiKey) return res.status(400).json({ error: '请先在页面中填写自己的 API Key' });

    const prompt = buildStyleTransferPrompt(req.body.extra);

    console.log('🎨 Gemini双图画风转换，模型:', imageModel);
    console.log('🖼️ 输入顺序: 1=画风参考图, 2=内容图');

    const image = await callMultiImageStyleTransfer({
      apiKey,
      baseURL,
      model: imageModel,
      prompt,
      styleInlineData: fileToInlineData(styleFile),
      photoInlineData: fileToInlineData(photoFile)
    });

    return res.json({ ok: true, image });
  } catch (err) {
    console.error('❌ 双图画风转换失败:', err.message);

    let userMessage = err.message || '图片生成失败';
    if (err.message?.includes('billing')) userMessage = 'API 余额不足，请充值后重试';
    if (err.message?.includes('content_policy')) userMessage = '图片内容不符合使用政策，请换一张图片';
    if (err.message?.includes('invalid_api_key')) userMessage = 'API Key 无效，请检查页面中填写的 API Key';

    res.status(500).json({ error: userMessage });
  }
});

// ── 结果图增强画质：只提升清晰度和细节，不改变内容 ──
app.post('/api/enhance-image', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: '缺少要增强的图片' });

    const apiKey = getRequestApiKey(req);
    const baseURL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
    const imageModel = process.env.IMAGE_MODEL || 'gpt-image-2';
    if (!apiKey) return res.status(400).json({ error: '请先在页面中填写自己的 API Key' });

    const imageInlineData = await imageInputToInlineData(image);

    console.log('✨ 增强画质，模型:', imageModel);

    const enhancedImage = await callMultiImageStyleTransfer({
      apiKey,
      baseURL,
      model: imageModel,
      prompt: buildEnhancePrompt(),
      styleInlineData: imageInlineData,
      photoInlineData: imageInlineData
    });

    return res.json({ ok: true, image: enhancedImage });
  } catch (err) {
    console.error('❌ 增强画质失败:', err.message);

    let userMessage = err.message || '增强画质失败';
    if (err.message?.includes('billing')) userMessage = 'API 余额不足，请充值后重试';
    if (err.message?.includes('content_policy')) userMessage = '图片内容不符合使用政策，请换一张图片';
    if (err.message?.includes('invalid_api_key')) userMessage = 'API Key 无效，请检查页面中填写的 API Key';

    res.status(500).json({ error: userMessage });
  }
});

// ── 拆分图层：基于当前结果图生成主体层、背景层、线稿层 ──
app.post('/api/split-layers', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: '缺少要拆分的图片' });

    const apiKey = getRequestApiKey(req);
    const baseURL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
    const imageModel = process.env.IMAGE_MODEL || 'gpt-image-2';
    if (!apiKey) return res.status(400).json({ error: '请先在页面中填写自己的 API Key' });

    const imageInlineData = await imageInputToInlineData(image);
    const layerDefs = [
      { key: 'subject', name: '主体层', prompt: buildLayerPrompt('subject') },
      { key: 'background', name: '背景层', prompt: buildLayerPrompt('background') },
      { key: 'lineart', name: '线稿层', prompt: buildLayerPrompt('lineart') }
    ];

    console.log('🧩 拆分图层，模型:', imageModel);

    const layers = [];
    for (const layer of layerDefs) {
      const layerImage = await callMultiImageStyleTransfer({
        apiKey,
        baseURL,
        model: imageModel,
        prompt: layer.prompt,
        styleInlineData: imageInlineData,
        photoInlineData: imageInlineData
      });

      layers.push({
        key: layer.key,
        name: layer.name,
        image: layerImage
      });
    }

    return res.json({ ok: true, layers });
  } catch (err) {
    console.error('❌ 拆分图层失败:', err.message);

    let userMessage = err.message || '拆分图层失败';
    if (err.message?.includes('billing')) userMessage = 'API 余额不足，请充值后重试';
    if (err.message?.includes('content_policy')) userMessage = '图片内容不符合使用政策，请换一张图片';
    if (err.message?.includes('invalid_api_key')) userMessage = 'API Key 无效，请检查页面中填写的 API Key';

    res.status(500).json({ error: userMessage });
  }
});

// ── 分析画风，生成提示词 ──
app.post('/api/transfer', upload.fields([
  { name: 'style', maxCount: 1 },
  { name: 'photo', maxCount: 1 }
]), async (req, res) => {
  try {
    const styleFile = req.files?.['style']?.[0];
    const photoFile = req.files?.['photo']?.[0];
    if (!styleFile || !photoFile) return res.status(400).json({ error: '请上传两张图片' });

    const apiKey = getRequestApiKey(req);
    const baseURL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
    if (!apiKey) return res.status(400).json({ error: '请先在页面中填写自己的 API Key' });

    const { intensity, format, extra } = req.body;
    const client = new OpenAI({ apiKey, baseURL });

    const toBase64URL = (file) => {
      const b64 = file.buffer.toString('base64');
      const mime = file.mimetype || 'image/jpeg';
      return `data:${mime};base64,${b64}`;
    };

    // ✅ 改进：让 gpt-4o 输出更结构化、更细致的风格描述
    const systemPrompt = `你是一位专业的AI图像生成提示词工程师，专精数字插画和半写实风格的图像转换。

    用户会发来两张图片：
    - 第一张：目标画风参考图（插画/艺术风格）
    - 第二张：需要转换的原始照片
    
    你的任务是：
    1. 极其仔细地分析第一张图的插画风格特征
    2. 提取第二张图的人物外貌、服装、姿势、背景等具体内容
    3. 将两者结合，生成一个能让图像生成AI把第二张照片画成第一张画风的精准提示词
    
    分析第一张图时必须关注以下细节：
    - 线条风格（是否有清晰黑色轮廓线？线条粗细？）
    - 眼睛画法（大小、虹膜颜色、光点位置、睫毛风格）
    - 皮肤质感（平滑卡通感？还是写实质感？有无赛璐璐阴影？）
    - 腮红方式（有无明显腮红？位置和颜色？）
    - 头发高光（分层高光？整体光泽？）
    - 嘴唇（光泽感？厚度？颜色？）
    - 整体色调（暖色/冷色？饱和度高低？）
    - 背景处理方式
    
    提示词要求：
    - 必须包含画风关键词（如 digital illustration, semi-realistic, sharp outlines 等）
    - 必须包含第二张图人物的具体描述（发色、发型、服装、配饰、姿势）
    - 必须包含第一张图特有的视觉风格细节
    - 英文提示词长度 150-250 词，越详细越好
    - 提示词开头必须先写画风关键词，再写人物内容
    - 画风关键词权重要高，用括号加强，例如：(digital illustration:1.4), (sharp outlines:1.3)
    - 人物的姿势、服装、背景必须完全照搬第二张图描述，不要改变
    - 负向提示词要明确排除写实照片感
    
    必须严格以如下 JSON 格式输出，不要有任何多余文字：
    {
      "style_analysis": "用3-4句中文详细描述第一张图的画风特征，重点说明线条、眼睛、皮肤、腮红、高光等插画细节",
      "style_tags": ["标签1", "标签2", "标签3", "标签4", "标签5", "标签6"],
      "prompt": "详细的英文图像生成提示词，必须同时包含画风描述和人物具体内容，150-250词",
      "negative_prompt": "英文负向提示词，明确排除写实照片感、模糊、比例失调等问题，40-60词",
      "tips": "1-2句中文建议，说明用哪个平台生成效果最好以及推荐的关键参数"
    }`;

    const userPrompt = `这是我的画风参考图（第一张）和要转换的照片（第二张）。
转换强度：${intensity || '平衡转换'}
输出风格类型：${format || '插画/数字绘画'}
${extra ? '补充要求：' + extra : ''}
请分析第一张图的画风，生成能将第二张照片转换为同款画风的专业提示词。`;

    const response = await client.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o',
      max_tokens: 1200,
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: toBase64URL(styleFile), detail: 'high' } },
            { type: 'image_url', image_url: { url: toBase64URL(photoFile), detail: 'high' } },
            { type: 'text', text: userPrompt }
          ]
        }
      ]
    });

    const raw = response.choices?.[0]?.message?.content || '';
    let parsed;
    try {
      parsed = JSON.parse(raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim());
    } catch {
      return res.status(500).json({ error: 'AI返回格式异常，请重试' });
    }

    // 把两张原始图片都返回给前端，生成阶段需要同时看到“风格图”和“内容图”。
    parsed.styleBase64 = toBase64URL(styleFile);
    parsed.photoBase64 = toBase64URL(photoFile);

    res.json({ ok: true, result: parsed });

  } catch (err) {
    console.error('❌', err.message);
    res.status(500).json({ error: err.message || '服务器错误' });
  }
});

// ── 直接生成图片 ──
// 核心：生成阶段必须同时输入风格参考图和内容图，不能只依赖文字提示词。
app.post('/api/generate', upload.none(), express.json({ limit: '50mb' }), async (req, res) => {
  try {
    const { prompt, negative_prompt, styleBase64, photoBase64 } = req.body;
    if (!photoBase64) return res.status(400).json({ error: '缺少要转换的图片' });
    if (!styleBase64) return res.status(400).json({ error: '缺少画风参考图' });

    const apiKey = getRequestApiKey(req);
    const baseURL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
    const imageModel = process.env.IMAGE_MODEL || 'gpt-image-2';
    if (!apiKey) return res.status(400).json({ error: '请先在页面中填写自己的 API Key' });

    const styleTransferPrompt = buildStyleTransferPrompt(`${prompt || '无'}

需要避免：
${negative_prompt || '写实照片感、模糊、比例失调、额外手指、文字、水印、边框'}`);

    console.log('🎨 Gemini生成图片，模型:', imageModel);
    console.log('📝 Prompt 长度:', styleTransferPrompt.length, '字符');
    console.log('🖼️ 使用 Gemini generateContent（双图参考模式：风格图 + 内容图）');

    const image = await callMultiImageStyleTransfer({
      apiKey,
      baseURL,
      model: imageModel,
      prompt: styleTransferPrompt,
      styleInlineData: base64ToInlineData(styleBase64),
      photoInlineData: base64ToInlineData(photoBase64)
    });

    return res.json({ ok: true, image });

  } catch (err) {
    console.error('❌ 生成图片失败:', err.message);

    // ✅ 友好的错误提示
    let userMessage = err.message || '图片生成失败';
    if (err.message?.includes('billing')) userMessage = 'API 余额不足，请充值后重试';
    if (err.message?.includes('content_policy')) userMessage = '图片内容不符合使用政策，请换一张照片';
    if (err.message?.includes('invalid_api_key')) userMessage = 'API Key 无效，请检查页面中填写的 API Key';

    res.status(500).json({ error: userMessage });
  }
});

app.use('/api', (req, res) => {
  res.status(404).json({ error: `接口不存在：${req.method} ${req.originalUrl}` });
});

app.use((err, req, res, next) => {
  console.error('❌ 服务端异常:', err.message);

  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: '图片文件过大，请压缩到 20MB 以内后再上传' });
  }

  if (err.type === 'entity.too.large') {
    return res.status(413).json({ error: '请求内容过大，请压缩图片后再试' });
  }

  res.status(500).json({ error: err.message || '服务器错误' });
});

app.listen(PORT, () => {
  console.log('\n🎨 AI画风转换工具已启动');
  console.log(`🌐 访问: http://localhost:${PORT}\n`);
});
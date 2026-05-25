const i18n = {
  zh: {
    subtitle: "支持指令与 mask 的图像编辑工作台",
    sourceTitle: "图像输入",
    waitingImage: "等待上传",
    uploadImage: "上传源图像",
    uploadHint: "PNG / JPG / WebP",
    maskTitle: "Mask 工作区",
    maskEmpty: "无 mask",
    brush: "画笔",
    eraser: "橡皮",
    fgPoint: "前景点",
    bgPoint: "背景点",
    brushSize: "笔刷",
    clearMask: "清空",
    saveMask: "保存 mask",
    segment: "SAM 分割",
    editTitle: "编辑参数",
    idle: "空闲",
    prompt: "指令",
    promptPlaceholder: "例如：把被 mask 覆盖的物体改成白色",
    negativePrompt: "负向指令",
    editProvider: "编辑模型",
    maskProvider: "分割模型",
    steps: "步数",
    device: "设备",
    maskBlur: "Mask 模糊",
    maskDilation: "Mask 膨胀",
    maskBox: "转为外接框",
    blurKernel: "模糊核",
    dilationKernel: "膨胀核",
    boxMargin: "框边距",
    runEdit: "开始编辑",
    resultTitle: "结果",
    noResult: "暂无结果",
    outputImage: "输出图",
    visualization: "可视化",
    uploaded: "已上传",
    saved: "已保存",
    drawing: "绘制中",
    running: "运行中",
    done: "完成",
    failed: "失败",
    needImage: "请先上传图像",
    needPrompt: "请输入编辑指令"
  },
  en: {
    subtitle: "Image editing workspace with instructions and masks",
    sourceTitle: "Source Image",
    waitingImage: "Waiting",
    uploadImage: "Upload source image",
    uploadHint: "PNG / JPG / WebP",
    maskTitle: "Mask Studio",
    maskEmpty: "No mask",
    brush: "Brush",
    eraser: "Eraser",
    fgPoint: "FG point",
    bgPoint: "BG point",
    brushSize: "Brush",
    clearMask: "Clear",
    saveMask: "Save mask",
    segment: "SAM segment",
    editTitle: "Edit Settings",
    idle: "Idle",
    prompt: "Prompt",
    promptPlaceholder: "Example: make the masked object white",
    negativePrompt: "Negative prompt",
    editProvider: "Edit model",
    maskProvider: "Mask model",
    steps: "Steps",
    device: "Device",
    maskBlur: "Mask blur",
    maskDilation: "Mask dilation",
    maskBox: "Mask to box",
    blurKernel: "Blur kernel",
    dilationKernel: "Dilation kernel",
    boxMargin: "Box margin",
    runEdit: "Run edit",
    resultTitle: "Result",
    noResult: "No result",
    outputImage: "Output",
    visualization: "Visualization",
    uploaded: "Uploaded",
    saved: "Saved",
    drawing: "Drawing",
    running: "Running",
    done: "Done",
    failed: "Failed",
    needImage: "Upload an image first",
    needPrompt: "Enter an edit prompt"
  }
};

const state = {
  lang: "zh",
  image: null,
  mask: null,
  tool: "brush",
  drawing: false,
  points: []
};

const $ = (id) => document.getElementById(id);
const t = (key) => i18n[state.lang][key] || key;
const canvas = $("maskCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

function applyI18n() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  $("langBtn").textContent = state.lang === "zh" ? "EN" : "中文";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
}

function setStatus(id, key, detail = "") {
  $(id).textContent = detail || t(key);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    rx: event.clientX - rect.left,
    ry: event.clientY - rect.top
  };
}

function resetCanvas(width, height) {
  canvas.width = width;
  canvas.height = height;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  canvas.style.opacity = "0.48";
  renderPoints();
}

function drawAt(point) {
  const size = Number($("brushSize").value);
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = state.tool === "eraser" ? "#000" : "#fff";
  ctx.beginPath();
  ctx.arc(point.x, point.y, size / 2, 0, Math.PI * 2);
  ctx.fill();
  setStatus("maskStatus", "drawing");
}

function renderPoints() {
  const layer = $("pointLayer");
  layer.innerHTML = "";
  const rect = canvas.getBoundingClientRect();
  state.points.forEach((point) => {
    const dot = document.createElement("span");
    dot.className = `point ${point.label === "foreground" ? "fg" : "bg"}`;
    dot.style.left = `${(point.x / canvas.width) * rect.width}px`;
    dot.style.top = `${(point.y / canvas.height) * rect.height}px`;
    layer.appendChild(dot);
  });
}

async function uploadBlob(blob, endpoint) {
  const form = new FormData();
  form.append("file", blob, "mask.png");
  const response = await fetch(endpoint, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function saveCanvasMask() {
  if (!state.image) throw new Error(t("needImage"));
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  state.mask = await uploadBlob(blob, "/api/masks");
  setStatus("maskStatus", "saved");
  return state.mask;
}

function useMaskImage(mask) {
  const image = new Image();
  image.onload = () => {
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  };
  image.src = mask.url;
  state.mask = mask;
  setStatus("maskStatus", "saved");
}

async function segment() {
  if (!state.image) throw new Error(t("needImage"));
  setStatus("maskStatus", "running");
  const response = await fetch("/api/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: $("maskProvider").value,
      image_id: state.image.id,
      points: state.points,
      multimask_output: true
    })
  });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  const list = $("maskCandidates");
  list.innerHTML = "";
  data.masks.forEach((mask) => {
    const button = document.createElement("button");
    const img = document.createElement("img");
    img.src = mask.url;
    button.appendChild(img);
    button.addEventListener("click", () => useMaskImage(mask));
    list.appendChild(button);
  });
  if (data.masks[0]) useMaskImage(data.masks[0]);
}

async function runEdit() {
  if (!state.image) throw new Error(t("needImage"));
  if (!$("prompt").value.trim()) throw new Error(t("needPrompt"));
  if (!state.mask) await saveCanvasMask();
  setStatus("runStatus", "running");
  const payload = {
    provider: $("editProvider").value,
    image_id: state.image.id,
    mask_id: state.mask.id,
    prompt: $("prompt").value.trim(),
    negative_prompt: $("negativePrompt").value,
    device: $("device").value,
    enable_mask_to_box: $("maskBox").checked,
    mask_box_margin: Number($("boxMargin").value),
    enable_mask_blur: $("maskBlur").checked,
    blur_kernel: Number($("blurKernel").value),
    enable_mask_dilation: $("maskDilation").checked,
    dilation_kernel: Number($("dilationKernel").value),
    num_inference_steps: Number($("steps").value)
  };
  const response = await fetch("/api/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  $("outputPreview").src = data.output.url;
  $("visPreview").src = data.visualization ? data.visualization.url : "";
  setStatus("runStatus", "done");
  setStatus("resultStatus", "done");
}

function bindEvents() {
  $("langBtn").addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    applyI18n();
  });

  $("imageInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/images", { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    state.image = await response.json();
    $("sourcePreview").src = state.image.url;
    $("canvasImage").src = state.image.url;
    $("canvasImage").onload = () => {
      resetCanvas($("canvasImage").naturalWidth, $("canvasImage").naturalHeight);
      setStatus("imageStatus", "uploaded");
    };
  });

  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-tool]").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      state.tool = button.dataset.tool;
    });
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.image) return;
    const point = canvasPoint(event);
    if (state.tool === "fg" || state.tool === "bg") {
      state.points.push({ x: point.x, y: point.y, label: state.tool === "fg" ? "foreground" : "background" });
      renderPoints();
      return;
    }
    state.drawing = true;
    canvas.setPointerCapture(event.pointerId);
    drawAt(point);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drawing) return;
    drawAt(canvasPoint(event));
  });
  canvas.addEventListener("pointerup", () => {
    state.drawing = false;
    state.mask = null;
  });
  window.addEventListener("resize", renderPoints);

  $("clearMaskBtn").addEventListener("click", () => {
    if (!canvas.width) return;
    resetCanvas(canvas.width, canvas.height);
    state.mask = null;
    state.points = [];
    $("maskCandidates").innerHTML = "";
    setStatus("maskStatus", "maskEmpty");
  });
  $("uploadMaskBtn").addEventListener("click", () => saveCanvasMask().catch(showError));
  $("segmentBtn").addEventListener("click", () => segment().catch(showError));
  $("runBtn").addEventListener("click", () => runEdit().catch(showError));
}

function showError(error) {
  console.error(error);
  setStatus("runStatus", "failed", `${t("failed")}: ${error.message || error}`);
  setStatus("maskStatus", "failed", `${t("failed")}`);
}

applyI18n();
bindEvents();


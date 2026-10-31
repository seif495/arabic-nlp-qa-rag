"use strict";

const GREETING =
  "مرحباً! أنا المساعد الذكي الخاص بك للإجابة على أسئلتك حول حلقات برنامج الدحيح الموسم الثامن (13 حلقة). كيف يمكنني مساعدتك اليوم؟\nHello! I am your AI assistant, here to answer your questions regarding the 13 episodes of ElDa7ee7 Season 8. How can I help you today?";
const STORE_KEY = "ms3-rag-state-v1";

const $ = (id) => document.getElementById(id);
const els = {
  messages: $("messages"),
  composer: $("composer"),
  question: $("question"),
  send: $("send"),
  reset: $("reset"),
  promptType: $("prompt_type"),
  memory: $("memory_strategy"),
  topK: $("top_k"),
  topKOut: $("top_k_out"),
  rerank: $("use_reranking"),
  genSet: $("gen_mode_set"),
  ms2Model: $("ms2_model"),
  modelMeta: $("model-meta"),
  modelStatus: $("model-status"),
  llmStatus: $("llm-status"),
  sidebar: $("sidebar"),
  menu: $("menu"),
};

// history: [{user, assistant}] sent to the stateless API every turn.
// messages: what the chat window renders.
let state = { history: [], messages: [] };
let ms2Available = false;
let ms2Label = "MS2 model";
let ms2Models = [];
let savedMs2Model = null;

function save() {
  try {
    localStorage.setItem(
      STORE_KEY,
      JSON.stringify({ ...state, settings: currentSettings() })
    );
  } catch (_) { /* storage unavailable: keep in memory only */ }
}

function load() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    state.history = Array.isArray(saved.history) ? saved.history : [];
    state.messages = Array.isArray(saved.messages) ? saved.messages : [];
    const s = saved.settings || {};
    if (s.prompt_type) els.promptType.value = s.prompt_type;
    if (s.memory_strategy) els.memory.value = s.memory_strategy;
    if (s.top_k) els.topK.value = s.top_k;
    if (s.generation_mode) {
      const radio = document.querySelector(`input[name="generation_mode"][value="${s.generation_mode}"]`);
      if (radio) radio.checked = true;
    }
    els.rerank.checked = Boolean(s.use_reranking);
    savedMs2Model = s.ms2_model || null;
  } catch (_) { /* ignore corrupt or blocked storage */ }
}

function currentSettings() {
  const mode = document.querySelector('input[name="generation_mode"]:checked').value;
  return {
    prompt_type: els.promptType.value,
    memory_strategy: els.memory.value,
    top_k: Number(els.topK.value),
    generation_mode: ms2Available ? mode : "llm_only",
    use_reranking: ms2Available && els.rerank.checked,
    ms2_model: ms2Available ? els.ms2Model.value : null,
  };
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function textBlock(text) {
  const node = el("div", "text", text);
  node.dir = "auto";
  return node;
}

function renderMessage(msg) {
  const node = document.getElementById("tpl-message").content.firstElementChild.cloneNode(true);
  node.classList.add(msg.role);
  node.querySelector(".avatar").textContent = msg.role === "user" ? "You" : "AI";
  const bubble = node.querySelector(".bubble");

  if (msg.role === "assistant" && msg.mode === "hybrid") {
    const split = el("div", "split");
    const llm = el("div", "pane llm");
    llm.append(el("h3", null, "LLM response"), textBlock(msg.llm_answer || ""));
    const ms2 = el("div", "pane ms2");
    ms2.append(el("h3", null, `${msg.ms2_label || ms2Label} response`), textBlock(msg.ms2_answer || ""));
    split.append(llm, ms2);
    bubble.append(split);
  } else if (msg.role === "assistant" && msg.mode === "ms2_only") {
    const pane = el("div", "pane ms2");
    pane.append(el("h3", null, `${msg.ms2_label || ms2Label} response`), textBlock(msg.content));
    bubble.append(pane);
  } else {
    bubble.append(textBlock(msg.content));
  }

  if (msg.logs) bubble.append(renderLogs(msg));
  els.messages.append(node);
  return node;
}

function renderLogs(msg) {
  const details = el("details", "logs");
  details.append(el("summary", null, "🛠️ Retrieval & context logs"));
  const body = el("div", "body");
  const logs = msg.logs;

  const kv = el("dl", "kv");
  const rows = [
    ["Standalone query", msg.standalone_query],
    ["Memory strategy", logs.memory_strategy],
    ["Prompt strategy", logs.prompt_type],
    ["Generation mode", logs.generation_mode],
    ["MS2 re-ranking", String(logs.reranking)],
    ["LLM", logs.llm_provider || "—"],
    ["History turns stored", String(logs.history_turns)],
    ["Timings (ms)", Object.entries(logs.timings_ms || {}).map(([k, v]) => `${k.replace("_ms", "")}=${v}`).join(" · ")],
  ];
  if (logs.ms2_window && logs.ms2_window.episode) {
    rows.push(["MS2 answer window", `${logs.ms2_window.episode} #${logs.ms2_window.chunk_id} (mean log-prob ${logs.ms2_window.mean_log_prob})`]);
  }
  if (logs.llm_errors && logs.llm_errors.length) rows.push(["LLM errors", logs.llm_errors.join("; ")]);
  for (const [k, v] of rows) {
    kv.append(el("dt", null, k));
    const dd = el("dd", null, v || "—");
    dd.dir = "auto";
    kv.append(dd);
  }
  body.append(kv);

  body.append(el("div", null, "History passed to the LLM:"));
  body.append(el("pre", null, logs.history_text || "(no history yet)"));

  if (msg.sources && msg.sources.length) {
    body.append(el("div", null, `Retrieved chunks (${msg.sources.length}):`));
    const list = el("ol", "sources");
    for (const s of msg.sources) {
      const li = el("li");
      const head = el("div", "src-head");
      const title = el("b", null, `${s.episode} · #${s.chunk_id}`);
      title.dir = "auto";
      const scores = s.ms2_score == null ? `BM25 ${s.bm25}` : `MS2 ${s.ms2_score} · BM25 ${s.bm25}`;
      head.append(title, el("span", null, scores));
      const preview = el("div", "preview", s.preview + "…");
      preview.dir = "auto";
      li.append(head, preview);
      list.append(li);
    }
    body.append(list);
  }
  details.append(body);
  return details;
}

function renderAll() {
  els.messages.replaceChildren();
  renderMessage({ role: "assistant", content: GREETING });
  for (const msg of state.messages) renderMessage(msg);
  scrollToEnd();
}

function scrollToEnd() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setBusy(busy) {
  els.send.disabled = busy;
  els.question.disabled = busy;
  els.send.textContent = busy ? "…" : "Send";
}

async function ask(question) {
  const userMsg = { role: "user", content: question };
  state.messages.push(userMsg);
  renderMessage(userMsg);
  const pending = renderMessage({ role: "assistant", content: "Retrieving context and generating response…" });
  pending.querySelector(".text").classList.add("typing");
  scrollToEnd();
  setBusy(true);

  try {
    const settings = currentSettings();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: state.history, ...settings }),
    });
    if (!res.ok) throw new Error(`/api/chat returned HTTP ${res.status}`);
    const data = await res.json();
    const msg = {
      role: "assistant",
      mode: data.logs.generation_mode,
      content: data.answer,
      llm_answer: data.llm_answer,
      ms2_answer: data.ms2_answer,
      standalone_query: data.standalone_query,
      sources: data.sources,
      logs: data.logs,
      ms2_label: ms2Label,
    };
    state.messages.push(msg);
    // Same rule as the Streamlit app: hybrid turns store the LLM answer.
    state.history.push({
      user: question,
      assistant: msg.mode === "hybrid" ? data.llm_answer || "" : data.answer,
    });
    pending.remove();
    renderMessage(msg);
  } catch (err) {
    pending.remove();
    const msg = {
      role: "assistant",
      content: `حدث خطأ غير متوقع في النظام. الرجاء المحاولة مرة أخرى لاحقاً.\nAn unexpected system error occurred. Please try again later.\n(${err.message})`,
    };
    state.messages.push(msg);
    renderMessage(msg);
    console.error(err);
  } finally {
    setBusy(false);
    save();
    scrollToEnd();
    els.question.focus();
  }
}

function showModel() {
  const model = ms2Models.find((m) => m.model === els.ms2Model.value);
  if (!model) return;
  ms2Label = model.display_name;
  els.modelMeta.textContent = `${model.description} · ${(model.parameter_count / 1e6).toFixed(2)}M params · positions: ${model.positional_encoding}`;
  els.modelStatus.textContent = `Loaded · dev token-F1 ${model.dev_token_f1} · EM ${model.dev_em}`;
  els.modelStatus.className = "model-status ok";
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    ms2Available = Boolean(cfg.ms2_available);
    ms2Models = [...(cfg.ms2_models || [])].sort((a, b) => a.rank - b.rank);
    els.ms2Model.replaceChildren(
      ...ms2Models.map((m) => {
        const best = m.model === cfg.ms2_default ? " · best" : "";
        const option = el("option", null, `${m.display_name}${best}`);
        option.value = m.model;
        return option;
      })
    );
    const wanted = ms2Models.some((m) => m.model === savedMs2Model) ? savedMs2Model : cfg.ms2_default;
    if (wanted) els.ms2Model.value = wanted;
    showModel();
    if (!ms2Available) els.modelStatus.textContent = "Weights not deployed. Run `python -m src.cli.ms3 export-web`.";
    els.modelStatus.className = "model-status " + (ms2Available ? "ok" : "off");
    els.llmStatus.textContent = cfg.llm_providers.length
      ? `LLM providers: ${cfg.llm_providers.join(" → ")} · ${cfg.chunks} chunks indexed`
      : `No LLM key configured. Only "MS2 only" mode can answer. · ${cfg.chunks} chunks indexed`;
  } catch (err) {
    els.modelStatus.textContent = `API unreachable: /api/config ${err.message}`;
    els.modelStatus.className = "model-status off";
  }
  els.genSet.classList.toggle("disabled", !ms2Available);
  els.ms2Model.closest("label").classList.toggle("disabled", !ms2Available);
  els.rerank.closest("label").classList.toggle("disabled", !ms2Available);
}

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (!question) return;
  els.question.value = "";
  autoGrow();
  ask(question);
});

els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

function autoGrow() {
  els.question.style.height = "auto";
  els.question.style.height = `${els.question.scrollHeight}px`;
}
els.question.addEventListener("input", autoGrow);

els.ms2Model.addEventListener("change", showModel);
els.topK.addEventListener("input", () => { els.topKOut.textContent = els.topK.value; });
for (const input of document.querySelectorAll("select, input")) input.addEventListener("change", save);

els.reset.addEventListener("click", () => {
  state = { history: [], messages: [] };
  save();
  renderAll();
});

els.menu.addEventListener("click", () => els.sidebar.classList.toggle("open"));

load();
els.topKOut.textContent = els.topK.value;
renderAll();
loadConfig();

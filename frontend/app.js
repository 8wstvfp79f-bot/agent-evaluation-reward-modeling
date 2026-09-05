const $ = (id) => document.getElementById(id);

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') { field += '"'; i += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === "," && !quoted) { row.push(field); field = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(field); field = "";
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift() || [];
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return parseCsv(await response.text());
}

const profileInfo = {
  "benefit-focused": { weights: ["0.8", "0.1", "0.1"], text: "更强调到达目标的收益，对路径成本和风险相对宽容。" },
  balanced: { weights: ["0.5", "0.3", "0.2"], text: "在到达目标、移动成本与障碍风险之间保持折中。" },
  "risk-averse": { weights: ["0.3", "0.2", "0.5"], text: "提高风险惩罚权重，倾向选择远离障碍物的轨迹。" },
};

function renderProfile(rows, profile) {
  const selected = rows.filter((row) => row.experiment === profile);
  const tail = selected.slice(-30);
  const average = tail.reduce((sum, row) => sum + Number(row.weighted_reward), 0) / Math.max(tail.length, 1);
  const info = profileInfo[profile];
  $("weights").innerHTML = ["Benefit", "Cost", "Risk"].map((name, index) =>
    `<div class="weight"><span>${name}</span><b>${info.weights[index]}</b></div>`
  ).join("");
  $("profileAverage").textContent = average.toFixed(2);
  $("profileExplanation").textContent = info.text;
  $("episodeCount").textContent = selected.length || "—";
}

function renderTrajectories(rows) {
  $("trajectoryCount").textContent = rows.length;
  if (rows.length) {
    $("bestScore").textContent = Number(rows[0].reward_score).toFixed(2);
    $("bestTrajectory").textContent = rows[0].trajectory_id;
  }
  $("trajectoryRows").innerHTML = rows.slice(0, 10).map((row) => `
    <tr>
      <td>${row.rank}</td><td>${row.trajectory_id}</td><td>${Number(row.reward_score).toFixed(2)}</td>
      <td>${row.total_benefit}</td><td>${row.total_cost}</td><td>${row.total_risk}</td><td>${row.num_steps}</td>
    </tr>`).join("");
}

function renderPair(rows) {
  if (!rows.length) return;
  const pair = [...rows].sort((a, b) => Number(a.uncertainty) - Number(b.uncertainty))[0];
  $("pairA").textContent = pair.trajectory_a;
  $("pairB").textContent = pair.trajectory_b;
  $("scoreA").textContent = `score ${Number(pair.score_a).toFixed(2)}`;
  $("scoreB").textContent = `score ${Number(pair.score_b).toFixed(2)}`;
  $("uncertainProbability").textContent = Number(pair.probability).toFixed(3);
}

async function init() {
  try {
    const [experiments, trajectories, pairs] = await Promise.all([
      loadCsv("../results/weight_experiments.csv"),
      loadCsv("../results/trajectory_scores.csv"),
      loadCsv("../results/active_querying_pairs.csv"),
    ]);
    renderProfile(experiments, $("profileSelect").value);
    renderTrajectories(trajectories);
    renderPair(pairs);
    $("profileSelect").addEventListener("change", (event) => renderProfile(experiments, event.target.value));
    $("status").className = "status ready";
    $("status").textContent = `已读取 ${experiments.length} 条权重实验、${trajectories.length} 条候选轨迹和 ${pairs.length} 组偏好对。`;
  } catch (error) {
    $("status").className = "status error";
    $("status").textContent = `结果读取失败：${error.message}。请从项目根目录运行 python -m http.server 8081，不要直接双击HTML。`;
  }
}

init();

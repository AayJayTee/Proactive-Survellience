const state = {
    jobId: null,
    pollTimer: null,
    pollEveryMs: 1200
};

const el = {
    uploadForm: document.getElementById("uploadForm"),
    videoInput: document.getElementById("videoInput"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    resultsSection: document.getElementById("resultsSection"),

    statusActionModel: document.getElementById("statusActionModel"),
    statusAnomalyModel: document.getElementById("statusAnomalyModel"),
    statusCaptionModel: document.getElementById("statusCaptionModel"),

    globalProgress: document.getElementById("globalProgress"),
    progressText: document.getElementById("progressText"),
    currentStage: document.getElementById("currentStage"),
    perFrameMs: document.getElementById("perFrameMs"),
    processingTime: document.getElementById("processingTime"),

    actionLabel: document.getElementById("actionLabel"),
    actionConfidenceBadge: document.getElementById("actionConfidenceBadge"),
    maxScore: document.getElementById("maxScore"),
    meanScore: document.getElementById("meanScore"),
    captionText: document.getElementById("captionText"),
    alertsList: document.getElementById("alertsList"),
    timelineList: document.getElementById("timelineList"),
    gradcamVideo: document.getElementById("gradcamVideo"),
    activationVideo: document.getElementById("activationVideo")
};

function setStatusChip(node, value) {
    const safe = (value || "ready").toLowerCase();
    node.textContent = safe.charAt(0).toUpperCase() + safe.slice(1);
    node.classList.remove("ready", "loading", "error");

    if (safe === "ready") {
        node.classList.add("ready");
    } else if (safe === "loading" || safe === "running") {
        node.classList.add("loading");
    } else {
        node.classList.add("error");
    }
}

function applyModelStatus(status) {
    const s = status || {};
    setStatusChip(el.statusActionModel, s.action || "ready");
    setStatusChip(el.statusAnomalyModel, s.anomaly || "ready");
    setStatusChip(el.statusCaptionModel, s.captioner || "ready");
}

function confidenceClass(conf) {
    if (conf >= 0.75) return "green";
    if (conf >= 0.45) return "orange";
    return "red";
}

function resetBeforeRun() {
    el.resultsSection.classList.add("hidden");
    el.globalProgress.value = 0;
    el.progressText.textContent = "Queued";
    el.currentStage.textContent = "Queued";
    el.perFrameMs.textContent = "-";
    el.processingTime.textContent = "-";
}

function formatStageName(stage) {
    if (!stage) return "Running";

    const custom = {
        init: "Initialization",
        decode: "Decoding Video",
        action: "Action Recognition",
        anomaly: "Anomaly Detection",
        gradcam: "Grad-CAM",
        activation: "Activation",
        caption: "Captioning",
        startup: "Startup",
        queued: "Queued",
        running: "Running",
        done: "Done",
        failed: "Failed"
    };

    const s = String(stage).toLowerCase();
    if (custom[s]) return custom[s];

    return s
        .replace(/[_-]+/g, " ")
        .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
}

function updateProgress(job) {
    const pct = Number(job.progress || 0);
    el.globalProgress.value = pct;
    el.progressText.textContent = "Progress: " + pct.toFixed(1) + "%";
    el.currentStage.textContent = formatStageName(job.current_stage);
    el.perFrameMs.textContent = job.per_frame_ms != null ? String(job.per_frame_ms) : "-";
    applyModelStatus(job.model_status || {});
}

function renderReport(report) {
    const action = report.action_recognition || {};
    const anomaly = report.anomaly || {};
    const alerts = report.alerts || [];
    const timeline = report.timeline || [];

    const conf = Number(action.confidence || 0);
    el.actionLabel.textContent = action.action || "-";
    el.actionConfidenceBadge.textContent = (conf * 100).toFixed(2) + "%";
    el.actionConfidenceBadge.classList.remove("green", "orange", "red");
    el.actionConfidenceBadge.classList.add(confidenceClass(conf));

    el.maxScore.textContent = Number(anomaly.max_score || 0).toFixed(4);
    el.meanScore.textContent = Number(anomaly.mean_score || 0).toFixed(4);
    el.captionText.textContent = report.caption || "-";
    el.processingTime.textContent = report.processing_time_sec != null ? String(report.processing_time_sec) : "-";

    el.alertsList.innerHTML = "";
    if (alerts.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No critical alerts";
        el.alertsList.appendChild(li);
    } else {
        alerts.forEach(function (a) {
            const li = document.createElement("li");
            li.textContent = a;
            el.alertsList.appendChild(li);
        });
    }

    el.timelineList.innerHTML = "";
    timeline.forEach(function (item) {
        const li = document.createElement("li");
        const start = Number(item.start || 0).toFixed(2);
        const end = Number(item.end || 0).toFixed(2);
        li.textContent = String(item.type) + " | " + start + "s -> " + end + "s";
        el.timelineList.appendChild(li);
    });

    if (report.outputs) {
        el.gradcamVideo.src = report.outputs.gradcam_video || "";
        el.activationVideo.src = report.outputs.activation_video || "";
    }

    el.resultsSection.classList.remove("hidden");
}

async function fetchStatusOnce() {
    if (!state.jobId) return;

    const res = await fetch("/api/status/" + state.jobId);
    const job = await res.json();

    if (!res.ok) {
        throw new Error(job.detail || "Could not fetch job status");
    }

    updateProgress(job);

    if (job.status === "completed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        el.progressText.textContent = "Completed";
        renderReport(job.result || {});
    } else if (job.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        el.progressText.textContent = "Failed: " + (job.error || "Unknown error");
        alert("Analysis failed: " + (job.error || "Unknown error"));
    }
}

async function startPolling() {
    await fetchStatusOnce();
    state.pollTimer = setInterval(function () {
        fetchStatusOnce().catch(function (err) {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
            el.progressText.textContent = "Status error";
            console.error(err);
        });
    }, state.pollEveryMs);
}

async function submitAnalyze(event) {
    event.preventDefault();

    const file = el.videoInput.files[0];
    if (!file) return;

    resetBeforeRun();

    const formData = new FormData();
    formData.append("video", file);

    el.analyzeBtn.disabled = true;
    el.progressText.textContent = "Uploading and starting job...";

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Failed to start analysis");
        }

        state.jobId = data.job_id;

        if (state.pollTimer) {
            clearInterval(state.pollTimer);
        }
        await startPolling();
    } catch (err) {
        alert("Could not start analysis: " + err.message);
    } finally {
        el.analyzeBtn.disabled = false;
    }
}

function init() {
    el.uploadForm.addEventListener("submit", submitAnalyze);
}

window.addEventListener("DOMContentLoaded", init);
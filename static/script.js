const chatWindow = document.getElementById("chat-window");
const questionInput = document.getElementById("question-input");
const chatForm = document.getElementById("chat-form");
const sendButton = document.getElementById("send-button");

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const fileList = document.getElementById("file-list");
const tabButtons = document.querySelectorAll(".tab-button");

const modeSelector = document.getElementById("mode-selector");
const modeButton = document.getElementById("mode-button");
const modeMenu = document.getElementById("mode-menu");
const modeMenuItems = document.querySelectorAll("#mode-menu li");
const folderDisplay = document.getElementById("folder-display");
const folderInput = document.getElementById("folder-input");
const syncButton = document.getElementById("sync-button");

const exportButton = document.getElementById("export-button");

const uploadButton = document.getElementById("upload-button");
const fileInput = document.getElementById("file-input");

const syncProgress = document.getElementById("sync-progress");
const syncProgressFill = document.getElementById("sync-progress-fill");
const syncProgressLabel = document.getElementById("sync-progress-label");

const storageChroma = document.getElementById("storage-chroma");
const storageUploads = document.getElementById("storage-uploads");
const storageTotal = document.getElementById("storage-total");
const clearButton = document.getElementById("clear-button");

const MODES = ["rag", "tool", "ai"];
let currentMode = "rag";
let currentFolder = "docs";
const syncedFolders = new Set();

function addMessage(text, kind, sources, autoDismissMs) {
    const el = document.createElement("div");
    el.className = `message ${kind}`;
    el.textContent = text;

    if (sources && sources.length > 0) {
        const sourcesEl = document.createElement("div");
        sourcesEl.className = "sources";
        sourcesEl.innerHTML = sources.map(s =>
            `<div class="source"><span class="source-name">${s.source}</span> <span class="source-score">${s.score.toFixed(4)}</span></div>`
        ).join("");
        el.appendChild(sourcesEl);
    }

    chatWindow.appendChild(el);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    if (autoDismissMs) {
        setTimeout(() => el.remove(), autoDismissMs);
    }

    return el;
}

// ----- Chat -----
chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    addMessage(question, "user");
    questionInput.value = "";
    questionInput.disabled = true;
    sendButton.disabled = true;

    const loadingEl = addMessage("Thinking...", "loading");

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question, mode: currentMode, folder: currentFolder })
        });
        const data = await response.json();
        loadingEl.remove();
        addMessage(data.answer, "ai", data.sources);
    } catch (err) {
        loadingEl.remove();
        addMessage("Something went wrong - please try again.", "ai");
    } finally {
        questionInput.disabled = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
});

// ----- Mode dropdown -----
modeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    modeMenu.classList.toggle("hidden");
});

modeMenuItems.forEach((item) => {
    item.addEventListener("click", () => {
        currentMode = item.dataset.mode;
        modeButton.textContent = currentMode;
        modeMenuItems.forEach((i) => i.classList.toggle("active", i === item));
        modeMenu.classList.add("hidden");
    });
});

document.addEventListener("click", (event) => {
    if (!modeSelector.contains(event.target)) {
        modeMenu.classList.add("hidden");
    }
});

// ----- Sidebar toggle -----
sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
});

// ----- File list -----
async function refreshFileList() {
    const response = await fetch(`/files?folder=${encodeURIComponent(currentFolder)}`);
    const data = await response.json();
    fileList.innerHTML = "";
    if (data.files.length === 0) {
        const li = document.createElement("li");
        li.className = "empty";
        li.textContent = "No files indexed yet.";
        fileList.appendChild(li);
        return;
    }
    for (const f of data.files) {
        const li = document.createElement("li");
        li.textContent = f;
        li.title = "Click to open";
        li.addEventListener("click", () => openFile(f));
        fileList.appendChild(li);
    }
}

async function openFile(path) {
    const response = await fetch("/open", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path })
    });
    const data = await response.json();
    if (data.error) {
        addMessage(data.error, "system");
    }
}

// ----- Streaming progress helpers -----
function showSyncProgress() {
    syncProgress.classList.remove("hidden");
    syncProgressFill.style.width = "0%";
    syncProgressLabel.textContent = "0 / 0";
}

function updateSyncProgress(done, total) {
    const pct = total > 0 ? (done / total) * 100 : 0;
    syncProgressFill.style.width = `${pct}%`;
    syncProgressLabel.textContent = `${done} / ${total}`;
}

function hideSyncProgress() {
    syncProgress.classList.add("hidden");
}

// Reads a newline-delimited-JSON streaming response, updating the progress
// bar as "progress" messages arrive, and returns the final "done"/"error" message.
async function readSyncStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalMessage = null;

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop(); // last entry may be an incomplete line - keep it for next time

        for (const line of lines) {
            if (!line.trim()) continue;
            const msg = JSON.parse(line);
            if (msg.type === "progress") {
                updateSyncProgress(msg.done, msg.total);
            } else {
                finalMessage = msg;
            }
        }
    }
    return finalMessage;
}

// ----- Folder sync -----
let syncInProgress = false;

async function syncFolder(folder) {
    if (syncInProgress) return; // ignore duplicate triggers (e.g. double-click) while one is already running
    syncInProgress = true;
    syncButton.disabled = true;

    const syncingEl = addMessage(`Syncing "${folder}"...`, "system");
    showSyncProgress();

    try {
        const response = await fetch("/sync", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder: folder })
        });
        const finalMessage = await readSyncStream(response);
        hideSyncProgress();
        syncingEl.remove();

        if (!finalMessage || finalMessage.type === "error") {
            if (folder === "uploaded_docs") {
                addMessage("You haven't uploaded any files yet - use the + button to add one.", "system");
            } else {
                addMessage(finalMessage ? finalMessage.error : "Sync failed.", "system");
            }
            refreshFileList();
            return;
        }
        const r = finalMessage.result;
        addMessage(
            `Added: ${r.num_added}, Updated: ${r.num_updated}, Skipped: ${r.num_skipped}, Deleted: ${r.num_deleted}`,
            "system", null, 5000
        );
        syncedFolders.add(folder);
        refreshFileList();
        refreshStorage();
    } finally {
        syncInProgress = false;
        syncButton.disabled = false;
    }
}

syncButton.addEventListener("click", () => {
    syncFolder(currentFolder);
});

// ----- Tabs (Main / Uploaded) -----
function setActiveTabButton() {
    tabButtons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.folder === currentFolder);
    });
}

function clearSystemMessages() {
    chatWindow.querySelectorAll(".message.system").forEach((el) => el.remove());
}

async function setFolder(folder) {
    clearSystemMessages();
    currentFolder = folder;
    folderDisplay.textContent = currentFolder;
    setActiveTabButton();

    const locked = currentFolder === "uploaded_docs";
    folderDisplay.classList.toggle("locked", locked);
    folderDisplay.title = locked ? "Fixed upload folder" : "Click to edit folder";

    if (!syncedFolders.has(currentFolder)) {
        await syncFolder(currentFolder);
    } else {
        refreshFileList();
    }
}

tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => setFolder(btn.dataset.folder));
});

// ----- Folder display (click to edit directly) -----
// "uploaded_docs" is a fixed system folder for the upload button's target,
// not an arbitrary path the user should be able to retarget.
folderDisplay.addEventListener("click", () => {
    if (currentFolder === "uploaded_docs") return;
    folderDisplay.classList.add("hidden");
    folderInput.classList.remove("hidden");
    folderInput.value = currentFolder;
    folderInput.focus();
    folderInput.select();
});

function confirmFolderDisplayEdit() {
    const newFolder = folderInput.value.trim() || currentFolder;
    folderDisplay.classList.remove("hidden");
    folderInput.classList.add("hidden");

    // keep the matching tab's data-folder + active state in sync, if one exists
    tabButtons.forEach((btn) => {
        if (btn.dataset.folder === currentFolder) {
            btn.dataset.folder = newFolder;
        }
    });

    setFolder(newFolder);
}

folderInput.addEventListener("blur", confirmFolderDisplayEdit);
folderInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        folderInput.blur();
    }
});

// ----- Upload -----
uploadButton.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const uploadingEl = addMessage(`Uploading "${file.name}"...`, "system");
    showSyncProgress();

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const finalMessage = await readSyncStream(response);
        hideSyncProgress();
        uploadingEl.remove();

        if (!finalMessage || finalMessage.type === "error") {
            addMessage(finalMessage ? finalMessage.error : "Upload failed.", "system");
            return;
        }
        const r = finalMessage.result;
        addMessage(
            `Added: ${r.num_added}, Updated: ${r.num_updated}, Skipped: ${r.num_skipped}, Deleted: ${r.num_deleted}`,
            "system", null, 5000
        );

        currentFolder = "uploaded_docs";
        folderDisplay.textContent = currentFolder;
        setActiveTabButton();
        syncedFolders.add(currentFolder);
        refreshFileList();
        refreshStorage();
    } catch (err) {
        hideSyncProgress();
        uploadingEl.remove();
        addMessage("Upload failed - please try again.", "system");
    } finally {
        fileInput.value = "";
    }
});

// ----- Storage -----
async function refreshStorage() {
    const response = await fetch("/storage");
    const data = await response.json();
    storageChroma.textContent = `${data.chroma_mb.toFixed(2)} MB`;
    storageUploads.textContent = `${data.uploads_mb.toFixed(2)} MB`;
    storageTotal.textContent = `${data.total_mb.toFixed(2)} MB`;
}

clearButton.addEventListener("click", async () => {
    const confirmed = confirm(
        "This deletes the vector database, record manager, and uploaded files. This cannot be undone. Continue?"
    );
    if (!confirmed) return;

    clearButton.disabled = true;
    await fetch("/clear", { method: "POST" });

    syncedFolders.clear();
    addMessage("All local data cleared.", "system", null, 5000);
    refreshFileList();
    refreshStorage();
    clearButton.disabled = false;
});

// ----- Export -----
exportButton.addEventListener("click", () => {
    const messages = chatWindow.querySelectorAll(".message.user, .message.ai");
    let transcript = "";
    messages.forEach((el) => {
        const role = el.classList.contains("user") ? "You" : "AI";
        transcript += `${role}: ${el.firstChild.textContent}\n\n`;
    });
    if (!transcript) {
        addMessage("Nothing to export yet.", "system");
        return;
    }
    const blob = new Blob([transcript], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "conversation.txt";
    a.click();
    URL.revokeObjectURL(url);
});

// ----- Initial load -----
syncedFolders.add(currentFolder); // assume the default folder is already synced from earlier sessions
refreshFileList();
refreshStorage();

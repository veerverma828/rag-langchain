const chatWindow = document.getElementById("chat-window");
const questionInput = document.getElementById("question-input");
const chatForm = document.getElementById("chat-form");
const sendButton = document.getElementById("send-button");

function addMessage(text, kind, sources) {
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
    return el;
}

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
            body: JSON.stringify({ question: question })
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

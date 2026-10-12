const TOKEN_KEY = "cropcare_access_token";
const authView = document.querySelector("#auth-view");
const dashboardView = document.querySelector("#dashboard-view");
const authForm = document.querySelector("#auth-form");
const authError = document.querySelector("#auth-error");
const authSubmit = document.querySelector("#auth-submit");
const emailField = document.querySelector("#email-field");
const passwordInput = document.querySelector("#password");
const loginTab = document.querySelector("#login-tab");
const registerTab = document.querySelector("#register-tab");
const userMenu = document.querySelector("#user-menu");
const apiStatus = document.querySelector("#api-status");
const imageInput = document.querySelector("#image-input");
const dropzone = document.querySelector("#dropzone");
const analyzeButton = document.querySelector("#analyze-button");
const MAX_FILE_SIZE = 5 * 1024 * 1024;
let authMode = "login";
let selectedFile = null;
let previewUrl = null;
let currentUser = null;

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setApiStatus(online) {
  apiStatus.classList.toggle("is-online", online);
  apiStatus.querySelector("span:last-child").textContent = online ? "API connected" : "API unavailable";
}

async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");

  let response;
  try {
    response = await fetch(path, { ...options, headers });
  } catch {
    setApiStatus(false);
    throw new Error("Cannot reach the API. Make sure the backend is running and try again.");
  }
  setApiStatus(true);

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    if (response.status === 401 && getToken()) {
      logout("Your session expired. Please sign in again.");
    }
    const detail = payload?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(" ")
      : detail || `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return payload;
}

function setAuthMode(mode) {
  authMode = mode;
  const isRegister = mode === "register";
  const usernameLabel = document.querySelector("#username-label");
  const usernameInput = document.querySelector("#username");
  loginTab.classList.toggle("is-active", !isRegister);
  registerTab.classList.toggle("is-active", isRegister);
  loginTab.setAttribute("aria-selected", String(!isRegister));
  registerTab.setAttribute("aria-selected", String(isRegister));
  emailField.hidden = !isRegister;
  document.querySelector("#email").required = isRegister;
  usernameLabel.textContent = isRegister ? "Username" : "Email address";
  usernameInput.type = isRegister ? "text" : "email";
  usernameInput.autocomplete = isRegister ? "username" : "email";
  usernameInput.placeholder = isRegister ? "Your username" : "you@example.com";
  document.querySelector("#form-title").textContent = isRegister ? "Create your account" : "Welcome back";
  document.querySelector("#form-subtitle").textContent = isRegister
    ? "Save your checks and keep your growing notes together."
    : "Sign in to continue to your workspace.";
  authSubmit.innerHTML = isRegister ? 'Create account <span aria-hidden="true">→</span>' : 'Sign in <span aria-hidden="true">→</span>';
  passwordInput.autocomplete = isRegister ? "new-password" : "current-password";
  clearMessage(authError);
}

function showAuth(message = "") {
  authView.hidden = false;
  dashboardView.hidden = true;
  userMenu.hidden = true;
  authForm.hidden = false;
  if (message) showMessage(authError, message);
}

function showDashboard(user) {
  currentUser = user;
  authView.hidden = true;
  dashboardView.hidden = false;
  userMenu.hidden = false;
  const username = user?.username || "grower";
  document.querySelector("#user-name").textContent = username;
  document.querySelector("#user-avatar").textContent = username.slice(0, 1).toUpperCase();
  document.querySelector("#welcome-name").textContent = username;
  loadHistory();
}

function logout(message = "") {
  localStorage.removeItem(TOKEN_KEY);
  currentUser = null;
  showAuth(message);
}

function showMessage(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function clearMessage(element) {
  element.textContent = "";
  element.hidden = true;
}

loginTab.addEventListener("click", () => setAuthMode("login"));
registerTab.addEventListener("click", () => setAuthMode("register"));
document.querySelector("#logout-button").addEventListener("click", () => logout());

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage(authError);
  const form = new FormData(authForm);
  const loginIdentifier = String(form.get("username") || "").trim();
  const password = String(form.get("password") || "");
  if (!loginIdentifier || !password) {
    showMessage(authError, "Enter your email address and password.");
    return;
  }

  authSubmit.disabled = true;
  authSubmit.textContent = authMode === "register" ? "Creating account…" : "Signing in…";
  try {
    let email = loginIdentifier;
    if (authMode === "register") {
      email = String(form.get("email") || "").trim();
      await apiRequest("/auth/register", {
        method: "POST",
        body: JSON.stringify({ username: loginIdentifier, email, password }),
      });
    }
    const tokenResponse = await apiRequest("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem(TOKEN_KEY, tokenResponse.access_token);
    const user = await apiRequest("/auth/me");
    showDashboard(user);
  } catch (error) {
    showMessage(authError, error.message);
  } finally {
    authSubmit.disabled = false;
    authSubmit.innerHTML = authMode === "register" ? 'Create account <span aria-hidden="true">→</span>' : 'Sign in <span aria-hidden="true">→</span>';
  }
});

function formatBytes(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clearSelectedFile() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  imageInput.value = "";
  document.querySelector("#drop-empty").hidden = false;
  document.querySelector("#drop-preview").hidden = true;
  analyzeButton.disabled = true;
  clearMessage(document.querySelector("#upload-error"));
}

function selectFile(file) {
  const errorElement = document.querySelector("#upload-error");
  clearMessage(errorElement);
  if (!file) return;
  const allowed = ["image/jpeg", "image/png"].includes(file.type) || /\.(jpe?g|png)$/i.test(file.name);
  if (!allowed) {
    clearSelectedFile();
    showMessage(errorElement, "Choose a JPG or PNG image.");
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    clearSelectedFile();
    showMessage(errorElement, "This image is over 5 MB. Choose a smaller file.");
    return;
  }
  clearSelectedFile();
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  document.querySelector("#preview-image").src = previewUrl;
  document.querySelector("#preview-name").textContent = file.name;
  document.querySelector("#preview-size").textContent = formatBytes(file.size);
  document.querySelector("#drop-empty").hidden = true;
  document.querySelector("#drop-preview").hidden = false;
  analyzeButton.disabled = false;
  document.querySelector("#result-panel").hidden = true;
}

document.querySelector("#browse-button").addEventListener("click", (event) => {
  event.stopPropagation();
  imageInput.click();
});
dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  imageInput.click();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
});
imageInput.addEventListener("change", () => selectFile(imageInput.files?.[0]));
document.querySelector("#remove-image").addEventListener("click", (event) => {
  event.stopPropagation();
  clearSelectedFile();
});
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
}
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer?.files?.[0]));

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  const errorElement = document.querySelector("#upload-error");
  clearMessage(errorElement);
  const formData = new FormData();
  formData.append("file", selectedFile);
  analyzeButton.disabled = true;
  analyzeButton.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> Analyzing image…';
  try {
    const result = await apiRequest("/predictions", { method: "POST", body: formData });
    displayResult(result);
    await loadHistory();
    document.querySelector("#result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showMessage(errorElement, error.message);
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.innerHTML = 'Analyze image <span aria-hidden="true">→</span>';
  }
});

function displayResult(result) {
  const confidence = Math.max(0, Math.min(1, Number(result.confidence) || 0));
  document.querySelector("#result-crop").textContent = result.crop || "Unknown";
  document.querySelector("#result-disease").textContent = result.disease || "Unknown";
  document.querySelector("#result-confidence").textContent = `${(confidence * 100).toFixed(1)}%`;
  document.querySelector("#confidence-bar").style.width = `${confidence * 100}%`;
  document.querySelector("#result-date").textContent = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date());
  document.querySelector("#result-panel").hidden = false;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

async function loadHistory() {
  const loading = document.querySelector("#history-loading");
  const empty = document.querySelector("#history-empty");
  const list = document.querySelector("#history-list");
  const errorElement = document.querySelector("#history-error");
  loading.hidden = false;
  empty.hidden = true;
  clearMessage(errorElement);
  try {
    const entries = await apiRequest("/history");
    loading.hidden = true;
    list.replaceChildren();
    const records = Array.isArray(entries) ? entries : [];
    document.querySelector("#checks-count").textContent = String(records.length);
    document.querySelector("#history-count").textContent = String(records.length);
    if (!records.length) {
      empty.hidden = false;
      return;
    }
    for (const entry of records.slice(0, 8)) {
      const item = document.createElement("article");
      item.className = "history-item";
      const marker = document.createElement("span");
      marker.className = `history-marker ${String(entry.disease).toLowerCase() === "healthy" ? "marker-healthy" : "marker-watch"}`;
      marker.textContent = String(entry.disease).toLowerCase() === "healthy" ? "✓" : "!";
      const details = document.createElement("div");
      details.className = "history-details";
      const title = document.createElement("strong");
      title.textContent = `${entry.crop || "Crop"} · ${entry.disease || "Result"}`;
      const date = document.createElement("span");
      date.textContent = formatDate(entry.created_at);
      details.append(title, date);
      const confidence = document.createElement("span");
      confidence.className = "history-confidence";
      confidence.textContent = `${(Number(entry.confidence || 0) * 100).toFixed(0)}%`;
      item.append(marker, details, confidence);
      list.append(item);
    }
  } catch (error) {
    loading.hidden = true;
    document.querySelector("#checks-count").textContent = "—";
    document.querySelector("#history-count").textContent = "—";
    showMessage(errorElement, error.message);
  }
}

async function initialize() {
  try {
    await apiRequest("/openapi.json");
  } catch {
    // Keep the auth screen visible and show backend availability separately.
  }
  const token = getToken();
  if (!token) {
    showAuth();
    return;
  }
  try {
    const user = await apiRequest("/auth/me");
    showDashboard(user);
  } catch {
    localStorage.removeItem(TOKEN_KEY);
    showAuth("Your saved session could not be restored. Please sign in again.");
  }
}

setAuthMode("login");
initialize();

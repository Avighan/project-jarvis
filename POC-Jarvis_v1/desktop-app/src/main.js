/**
 * Jarvis Desktop — Electron main process
 * Spawns the FastAPI backend (web-app/main.py) and loads it in a native window.
 * Both apps share the same backend — desktop is just a chrome around localhost:8000.
 */

const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const BACKEND_PORT = 8000;
const BACKEND_URL  = `http://127.0.0.1:${BACKEND_PORT}`;

let mainWindow = null;
let backendProcess = null;

// ── Start FastAPI backend ────────────────────────────────────────────────────

function startBackend() {
  const webAppDir  = path.join(__dirname, '..', '..', 'web-app');
  const projectRoot = path.join(__dirname, '..', '..', '..');
  const python3    = '/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/bin/python3';
  backendProcess = spawn(python3, ['main.py'], {
    cwd: webAppDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONPATH: projectRoot },
  });

  backendProcess.stdout.on('data', d => console.log('[backend]', d.toString().trim()));
  backendProcess.stderr.on('data', d => console.error('[backend]', d.toString().trim()));

  backendProcess.on('close', code => {
    console.log(`[backend] exited with code ${code}`);
  });
}

// ── Wait for backend to be ready then open window ────────────────────────────

function waitForBackend(retries = 20) {
  return new Promise((resolve, reject) => {
    const http = require('http');
    let attempts = 0;
    const check = () => {
      http.get(BACKEND_URL, res => {
        resolve();
      }).on('error', () => {
        attempts++;
        if (attempts >= retries) {
          reject(new Error('Backend did not start in time'));
        } else {
          setTimeout(check, 500);
        }
      });
    };
    setTimeout(check, 1000); // give Python 1s head start
  });
}

// ── Create window ─────────────────────────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width:  1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0f0f11',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadURL(BACKEND_URL);
  mainWindow.on('closed', () => { mainWindow = null; });

  // Open external links in browser, not Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  // Check if backend is already running (started externally)
  const http = require('http');
  const alreadyUp = await new Promise(resolve => {
    http.get(BACKEND_URL, () => resolve(true)).on('error', () => resolve(false));
  });

  if (!alreadyUp) {
    startBackend();
  } else {
    console.log('[backend] already running, skipping spawn');
  }

  try {
    await waitForBackend();
    createWindow();
  } catch (e) {
    console.error('Failed to reach backend:', e.message);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (!mainWindow) createWindow();
});

app.on('before-quit', () => {
  if (backendProcess) backendProcess.kill();
});

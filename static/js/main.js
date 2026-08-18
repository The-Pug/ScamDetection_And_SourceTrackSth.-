// Live Hamming-distance calculator for the perceptual-similarity form
function initHammingCalc() {
  const a = document.getElementById("hash-a");
  const b = document.getElementById("hash-b");
  const out = document.getElementById("hamming-result");
  if (!a || !b || !out) return;

  async function update() {
    if (!a.value || !b.value) { out.textContent = ""; return; }
    const res = await fetch(`/api/hamming?a=${encodeURIComponent(a.value)}&b=${encodeURIComponent(b.value)}`);
    const data = await res.json();
    if (data.distance === null) {
      out.innerHTML = '<span class="badge badge-risk">Invalid hexadecimal hash</span>';
    } else {
      const pct = Math.round((1 - data.distance / 256) * 100);
      const cls = data.distance <= 12 ? "badge-warn" : "badge-ok";
      out.innerHTML = `<span class="badge ${cls}">Hamming distance: ${data.distance} · ~${pct}% similar</span>`;
    }
  }
  a.addEventListener("input", update);
  b.addEventListener("input", update);
}

// Recursively walk a dropped folder (DataTransferItem -> FileSystemEntry) into a flat file list.
async function readAllDirectoryEntries(reader) {
  let all = [];
  let batch;
  do {
    batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    all = all.concat(batch);
  } while (batch.length > 0);
  return all;
}

async function collectFilesFromEntry(entry, files) {
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    files.push(file);
  } else if (entry.isDirectory) {
    const entries = await readAllDirectoryEntries(entry.createReader());
    for (const child of entries) {
      await collectFilesFromEntry(child, files);
    }
  }
}

function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const status = document.getElementById("upload-status");
  const zone = document.getElementById("dropzone");
  if (status) status.textContent = `Uploading and analyzing ${files.length} file(s)... this can take a while for a large folder.`;
  if (zone) zone.style.pointerEvents = "none";

  const formData = new FormData();
  files.forEach(f => formData.append("files", f, f.name));

  fetch("/evidence/upload", { method: "POST", body: formData })
    .then(resp => { window.location.href = resp.url; })
    .catch(() => {
      if (status) status.textContent = "Upload failed — check that the server is still running.";
      if (zone) zone.style.pointerEvents = "auto";
    });
}

function initUploader() {
  const zone = document.getElementById("dropzone");
  if (!zone) return;
  const filesInput = document.getElementById("files-input");
  const folderInput = document.getElementById("folder-input");
  const pickFilesBtn = document.getElementById("pick-files-btn");
  const pickFolderBtn = document.getElementById("pick-folder-btn");

  pickFilesBtn.addEventListener("click", () => filesInput.click());
  pickFolderBtn.addEventListener("click", () => folderInput.click());
  filesInput.addEventListener("change", () => uploadFiles(filesInput.files));
  folderInput.addEventListener("change", () => uploadFiles(folderInput.files));

  ["dragenter", "dragover"].forEach(evt =>
    zone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); zone.classList.add("drag"); })
  );
  zone.addEventListener("dragleave", e => { e.preventDefault(); zone.classList.remove("drag"); });

  zone.addEventListener("drop", async e => {
    e.preventDefault();
    e.stopPropagation();
    zone.classList.remove("drag");
    const items = e.dataTransfer.items;
    let files = [];
    if (items && items.length && items[0].webkitGetAsEntry) {
      for (const item of items) {
        const entry = item.webkitGetAsEntry();
        if (entry) await collectFilesFromEntry(entry, files);
      }
    } else {
      files = Array.from(e.dataTransfer.files);
    }
    uploadFiles(files);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initHammingCalc();
  initUploader();
});

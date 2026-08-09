// Local IndexedDB wrapper for Popup UI to manage assets
class VectorStore {
    static async getDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open("SmartDispatchStore", 1);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains("assets")) {
                    db.createObjectStore("assets", { keyPath: "id", autoIncrement: true });
                }
            };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    static async getAllAssets() {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction("assets", "readonly");
            const store = tx.objectStore("assets");
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    static async deleteAsset(id) {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction("assets", "readwrite");
            const store = tx.objectStore("assets");
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
}

async function loadAssets() {
    const listEl = document.getElementById('assetsList');
    try {
        const assets = await VectorStore.getAllAssets();
        if (assets.length === 0) {
            listEl.innerHTML = '<p class="text-xs text-gray-500 italic">No assets saved yet.</p>';
            return;
        }
        
        let html = '';
        assets.forEach(a => {
            html += `
            <div class="flex justify-between items-center p-2 bg-white rounded shadow-sm text-xs border border-gray-200">
                <div class="truncate pr-2 w-3/4">
                    <strong class="block truncate">${a.label}</strong>
                    <span class="text-gray-400 truncate text-[10px]">${a.url}</span>
                </div>
                <button class="text-red-500 hover:text-red-700 delete-btn font-semibold" data-id="${a.id}">Delete</button>
            </div>
            `;
        });
        listEl.innerHTML = html;
        
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const id = parseInt(e.target.getAttribute('data-id'));
                await VectorStore.deleteAsset(id);
                loadAssets(); // refresh list
            });
        });
    } catch (e) {
        listEl.innerHTML = '<p class="text-xs text-red-500">Error loading assets.</p>';
    }
}

// Initial load
loadAssets();

document.getElementById('assetForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const asset = {
        label: document.getElementById('label').value,
        keywords: document.getElementById('keywords').value,
        url: document.getElementById('url').value,
        type: document.getElementById('type').value
    };
    
    const statusEl = document.getElementById('status');
    const submitBtn = document.getElementById('submitBtn');
    
    statusEl.textContent = 'Generating semantic embedding...';
    statusEl.classList.remove('hidden');
    submitBtn.disabled = true;
    submitBtn.classList.add('opacity-50');
    
    chrome.runtime.sendMessage({
        type: 'ADD_ASSET',
        asset: asset
    }, (response) => {
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-50');
        
        if (response && response.success) {
            statusEl.textContent = 'Asset saved securely in IndexedDB.';
            statusEl.classList.remove('text-red-600');
            statusEl.classList.add('text-green-600');
            document.getElementById('assetForm').reset();
            loadAssets(); // Refresh list to show new asset
        } else {
            statusEl.textContent = 'Error saving asset. Ensure offscreen document is ready.';
            statusEl.classList.remove('text-green-600');
            statusEl.classList.add('text-red-600');
        }
    });
});

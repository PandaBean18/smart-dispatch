function makeDraggable(el, handle = el) {
    let isDragging = false;
    let startX, startY, initialX, initialY;
    let hasMoved = false;

    handle.style.cursor = 'grab';

    handle.addEventListener('mousedown', (e) => {
        // Prevent dragging if clicking an interactive element inside the modal
        if (['INPUT', 'BUTTON', 'LABEL'].includes(e.target.tagName) || e.target.closest('button')) {
            return;
        }
        
        isDragging = true;
        hasMoved = false;
        handle.style.cursor = 'grabbing';
        startX = e.clientX;
        startY = e.clientY;
        
        const rect = el.getBoundingClientRect();
        
        // Convert to fixed positioning coordinates to avoid CSS conflicts
        el.style.position = 'fixed';
        el.style.right = 'auto';
        el.style.bottom = 'auto';
        el.style.margin = '0';
        el.style.left = rect.left + 'px';
        el.style.top = rect.top + 'px';
        
        initialX = rect.left;
        initialY = rect.top;
        
        // Stop text selection during drag
        e.preventDefault(); 
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
            hasMoved = true;
        }
        
        el.style.left = (initialX + dx) + 'px';
        el.style.top = (initialY + dy) + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            handle.style.cursor = 'grab';
            
            // Mark the element so click handlers know if a drag just finished
            if (hasMoved) {
                el.dataset.dragged = 'true';
            } else {
                el.dataset.dragged = 'false';
            }
        }
    });
}

const composeBoxes = new WeakSet();

// Observe Gmail to find newly opened compose windows
const observer = new MutationObserver((mutations) => {
    const boxes = document.querySelectorAll('div[contenteditable="true"][aria-label*="Message Body"]');
    boxes.forEach(box => {
        if (!composeBoxes.has(box)) {
            composeBoxes.add(box);
            injectAnalyzeButton(box);
        }
    });
});
observer.observe(document.body, { childList: true, subtree: true });

function injectAnalyzeButton(composeBox) {
    const btn = document.createElement('div');
    btn.className = 'sd-analyze-btn';
    btn.innerHTML = '✨ Analyze Draft';
    btn.title = "Click to find contextual links and files for this draft";
    
    // Apply draggable logic to the button itself
    makeDraggable(btn);
    
    btn.onclick = (e) => {
        // Prevent click if we just dragged the button
        if (btn.dataset.dragged === 'true') {
            btn.dataset.dragged = 'false';
            return;
        }
        
        const text = composeBox.innerText;
        btn.innerHTML = '✨ Analyzing...';
        btn.style.opacity = '0.7';
        
        chrome.runtime.sendMessage({
            type: 'ANALYZE_DRAFT',
            text: text
        }, (response) => {
            btn.innerHTML = '✨ Analyze Draft';
            btn.style.opacity = '1';
            
            if (response && response.success && response.suggestions) {
                const validSuggestions = response.suggestions.filter(s => s.score > 0.4);
                if (validSuggestions.length > 0) {
                    showModal(composeBox, validSuggestions, response.profiler);
                } else {
                    alert("Smart Dispatch: No relevant matches found in your database for this draft.");
                }
            } else {
                alert("Smart Dispatch: Failed to analyze. Ensure background worker is active.");
            }
        });
    };
    
    // Gmail compose boxes are usually inside a structured table/div setup.
    // We attach it to the parent container so it floats over the top right.
    const parent = composeBox.parentElement;
    parent.style.position = 'relative';
    parent.appendChild(btn);
}

let currentModal = null;

function showModal(container, suggestions, profiler) {
    if (currentModal) {
        currentModal.remove();
    }
    
    const modal = document.createElement('div');
    modal.id = 'smart-dispatch-modal';
    
    let html = `<div class="sd-header sd-modal-handle" style="cursor: grab;">Smart Dispatch ✨</div>`;
    html += `<div class="sd-instructions">Select assets to insert:</div>`;
    html += `<ul class="sd-list">`;
    suggestions.forEach((s, idx) => {
        html += `<li class="sd-item">
                    <label style="display: flex; align-items: center; cursor: pointer; width: 100%;">
                        <input type="checkbox" class="sd-checkbox" value="${idx}" checked style="margin-right: 8px;">
                        <span style="flex-grow: 1;"><strong>${s.label}</strong></span> 
                        <span class="sd-score">${(s.score*100).toFixed(0)}%</span>
                    </label>
                 </li>`;
    });
    html += `</ul>`;
    html += `<div class="sd-actions">
                <button class="sd-btn sd-btn-cancel">Cancel</button>
                <button class="sd-btn sd-btn-proceed">Proceed</button>
             </div>`;
             
    modal.innerHTML = html;
    document.body.appendChild(modal);
    currentModal = modal;
    
    // Position modal centered over the compose box
    const rect = container.getBoundingClientRect();
    modal.style.top = Math.max(0, rect.top + (rect.height / 2) - (modal.offsetHeight / 2)) + 'px';
    modal.style.left = Math.max(0, rect.left + (rect.width / 2) - (modal.offsetWidth / 2)) + 'px';
    
    // Make modal draggable (using the header as the handle)
    makeDraggable(modal, modal.querySelector('.sd-modal-handle'));
    
    // Setup Action Buttons
    modal.querySelector('.sd-btn-cancel').onclick = () => {
        modal.remove();
        currentModal = null;
    };
    
    modal.querySelector('.sd-btn-proceed').onclick = () => {
        const checkboxes = modal.querySelectorAll('.sd-checkbox:checked');
        if (checkboxes.length === 0) {
            modal.remove();
            currentModal = null;
            return;
        }
        
        let htmlToInsert = '<br><br><b>Relevant Assets:</b><ul>';
        checkboxes.forEach(cb => {
            const item = suggestions[cb.value];
            htmlToInsert += `<li><a href="${item.url}">${item.label}</a></li>`;
        });
        htmlToInsert += '</ul><br>';
        
        insertHtmlAtCursor(container, htmlToInsert);
        
        modal.remove();
        currentModal = null;
    };
}

function insertHtmlAtCursor(container, html) {
    container.focus();
    // Use execCommand to insert HTML at the current caret position in the contenteditable div
    const success = document.execCommand('insertHTML', false, html);
    if (!success) {
        // Fallback if execCommand is fully deprecated or fails
        container.innerHTML += html;
    }
}

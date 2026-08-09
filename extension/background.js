// Background Service Worker
let creating; // Promise to track offscreen document creation
async function setupOffscreenDocument(path) {
  const url = chrome.runtime.getURL(path);
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [url]
  });

  if (existingContexts.length > 0) {
    return;
  }
  
  if (creating) {
    await creating;
  } else {
    creating = chrome.offscreen.createDocument({
      url: path,
      reasons: ['WORKERS'], // Or DOM_PARSER, MATCH_MEDIA etc.
      justification: 'Run ONNX Inference using Transformers.js'
    });
    await creating;
    creating = null;
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE_DRAFT') {
    (async () => {
      await setupOffscreenDocument('offscreen/offscreen.html');
      // Forward the message to the offscreen document
      chrome.runtime.sendMessage({
        type: 'INFERENCE_REQUEST',
        text: message.text
      }, (response) => {
        sendResponse(response);
      });
    })();
    return true; // Keep the message channel open for the async response
  }
  
  if (message.type === 'ADD_ASSET') {
      (async () => {
          await setupOffscreenDocument('offscreen/offscreen.html');
          chrome.runtime.sendMessage({
              type: 'EMBED_ASSET_REQUEST',
              asset: message.asset
          }, (response) => {
              sendResponse(response);
          });
      })();
      return true;
  }
});

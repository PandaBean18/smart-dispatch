#!/bin/bash
cd /home/rndbn/projects/smart-dispatch/extension
mkdir -p lib
echo "Downloading Tailwind CSS..."
curl -sL https://cdn.tailwindcss.com -o lib/tailwindcss.js
echo "Downloading Transformers.js..."
curl -sL https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js -o lib/transformers.min.js
echo "Setup complete."

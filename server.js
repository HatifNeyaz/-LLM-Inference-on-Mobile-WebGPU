import http from 'http';
import fs from 'fs';
import path from 'path';

const PORT = 8080;
const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.wasm': 'application/wasm',
    '.json': 'application/json'
};

const currentDir = import.meta.dirname;

const server = http.createServer((req, res) => {
    let filePath = path.join(currentDir, req.url === '/' ? 'index.html' : req.url);
    const extname = path.extname(filePath);
    
    fs.readFile(filePath, (err, content) => {
        if (err) {
            res.writeHead(404);
            res.end('File not found');
            return;
        }

        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        res.setHeader('Content-Type', MIME_TYPES[extname] || 'text/plain');
        
        res.writeHead(200);
        res.end(content, 'utf-8');
    });
});

server.listen(PORT, () => {
    console.log(`Profiling Server running at http://localhost:${PORT}`);
    console.log(`Make sure to open this in Chrome/Edge.`);
});
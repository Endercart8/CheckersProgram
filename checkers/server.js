// server.js
const express = require('express');
const http = require('http');
const path = require('path');
const WebSocket = require('ws');

const app = express();
const PORT = process.env.PORT || 8080;

// Serve static client files from ./public
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

let pythonSocket = null; // Store the AI connection

// Broadcast JSON to all browser clients
function broadcastJSON(obj) {
  const msg = JSON.stringify(obj);
  wss.clients.forEach((client) => {
    // Only broadcast to browser clients (not Python)
    if (client.readyState === WebSocket.OPEN && client !== pythonSocket) {
      client.send(msg);
    }
  });
}

wss.on('connection', (ws, req) => {
  const path = req.url; // "/" for browsers, "/python" for AI
  console.log('Connection on path:', path, 'from', req.socket.remoteAddress);

  if (path === '/python') {
    console.log('Python AI connected');
    pythonSocket = ws;

    ws.on('message', (raw) => {
      try {
        const obj = JSON.parse(raw);
        // Forward any state from Python to browsers
        broadcastJSON(obj);
      } catch (e) {
        console.error('Invalid JSON from Python:', e);
      }
    });

    ws.on('close', () => {
      console.log('Python disconnected');
      pythonSocket = null;
    });

  } else { // Browser connection
    console.log('Browser connected');

    ws.on('message', (raw) => {
      try {
        const obj = JSON.parse(raw);

        if (obj.type === 'player_move'|| obj.type === 'start_training' || obj.type === 'start_game' || obj.type === 'stop') {
          // Forward human move to Python AI
          if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
            pythonSocket.send(JSON.stringify(obj));
          }
        } else {
          // Optionally broadcast other messages to browsers
          broadcastJSON(obj);
        }
      } catch (e) {
        console.error('Invalid JSON from browser:', e);
      }
    });

    ws.on('close', () => {
      console.log('Browser disconnected');
    });
  }
});

// Optional HTTP endpoint for Python POST (if needed)
app.use(express.json({ limit: '1mb' }));
app.post('/push_state', (req, res) => {
  broadcastJSON(req.body);
  res.json({ status: 'ok' });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server listening on http://0.0.0.0:${PORT}`);
});
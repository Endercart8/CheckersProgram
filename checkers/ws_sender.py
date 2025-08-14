import json
import websocket  # websocket-client package
import time

WS_URL = "ws://192.168.137.116:8080"  # use your server IP if remote

class WSClient:
    def __init__(self, url=WS_URL):
        self.url = url
        self.ws = None
        self.connect()

    def connect(self):
        try:
            self.ws = websocket.create_connection(self.url, timeout=5)
        except Exception as e:
            print("WS connect error:", e)
            self.ws = None

    def send_state(self, board, current_player, legal_moves):
        if self.ws is None:
            self.connect()
            if self.ws is None:
                return
        msg = {
            "type": "state",
            "board": board.tolist() if hasattr(board, 'tolist') else board,
            "current_player": int(current_player),
            "legal_moves": legal_moves
        }
        try:
            self.ws.send(json.dumps(msg))
        except Exception as e:
            print("WS send error, reconnecting:", e)
            self.ws = None
    
    def close(self):
        if self.ws:
            self.ws.close()
            self.ws = None

# Example usage in your self-play:
# ws = WSClient("ws://192.168.1.5:8080")
# ws.send_state(env.board, env.current_player, last_move=((r0,c0),(r1,c1)))
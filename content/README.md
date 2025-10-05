# Open-LLM-VTuber
Original Github: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber

### Youtube Reference
#### Open-LLM-VTuber Running on VPS
<p align="center">
  <a href="https://www.youtube.com/watch?v=JefGof3G-o8" target="_blank">
    <img src="https://img.youtube.com/vi/JefGof3G-o8/0.jpg" alt="YouTube Video Thumbnail" width="480" />
  </a>
</p>
Click the image above to watch the video

#### Steps Run Open-LLM-VTuber on VPS
<p align="center">
  <a href="https://www.youtube.com/watch?v=lGCeIjEgtPU" target="_blank">
    <img src="https://img.youtube.com/vi/lGCeIjEgtPU/0.jpg" alt="YouTube Video Thumbnail" width="480" />
  </a>
</p>
Click the image above to watch the video

##### 1. Update & Install
```
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip curl
```


##### 2. Clone Repository
```
git clone --recursive https://github.com/Open-LLM-VTuber/Open-LLM-VTuber.git
cd Open-LLM-VTuber
```

##### 3. Prepare Python Virtual Env
```
cd Open-LLM-VTuber
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel packaging
pip install -r requirements.txt
```

##### 4. Install modules
```
pip install pyyaml ruamel.yaml uvicorn loguru fastapi websockets requests tomli packaging chardet numpy mcp anthropic openai pysbd langdetect jinja2 Letta pydub sherpa_onnx edge_tts
```

##### 5. Install ffmpeg
```
sudo apt install -y ffmpeg
```

##### 6. Setting conf.yaml
Run python run_server.py to make conf.yaml
```
python3 run_server.py
```
- Edit URL (replace to 0.0.0.0):
```
system_config:
  conf_version: 'v1.2.1'
  host: '0.0.0.0' # use 0.0.0.0 if you want other devices to access this page
```
- Host Letta
```
  letta_agent:
      host: '0.0.0.0' # Host address
```
- Set LLM provider
```
    agent_settings:
      basic_memory_agent:
        # The Basic AI Agent. Nothing fancy.
        # choose one of the llm provider from the llm_config
        # and set the required parameters in the corresponding field
        # examples: 
        # 'openai_compatible_llm', 'llama_cpp_llm', 'claude_llm', 'ollama_llm'
        # 'openai_llm', 'gemini_llm', 'zhipu_llm', 'deepseek_llm', 'groq_llm'
        # 'mistral_llm', 'lmstudio_llm', and more
        llm_provider: 'gemini_llm'
```
- Fill API key
```
      gemini_llm:
        llm_api_key: 'YOUR API KEY'
        model: 'gemini-2.5-flash-lite'
        temperature: 0.4 # value between 0 to 2
```

##### 7. Edit file main-nu7uwxNJ.js
Copy file main-nu7uwxNJ.js from this github, copy to folder /frontend/assets/

##### 8. Run Server
```
cd Open-LLM-VTuber
python3 -m venv venv
source venv/bin/activate
python3 run_server.py
```

##### 9.  Open from Browser
IP ADDRESS:12393

#### Setup SSL for HTTPS access
##### 1. Create Self-Signed Certificate folder on VPS
```
sudo mkdir -p /etc/ssl/open-llm-vtuber
cd /etc/ssl/open-llm-vtuber
```
##### 2. Create private key and certificate
```
sudo openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout server.key \
  -out server.crt
```
When asked Common Name (CN), fill with your IP VPS:
Common Name (CN): 103.171.85.170

#### 3.Edit run_server.py
```
    # Run the Uvicorn server
    logger.info(f"Starting server on {server_config.host}:{server_config.port}")
    uvicorn.run(
        app=server.app,
        host=server_config.host,
        port=server_config.port,
        log_level=console_log_level.lower(),
        ssl_certfile="/etc/ssl/open-llm-vtuber/server.crt",
        ssl_keyfile="/etc/ssl/open-llm-vtuber/server.key",)
```

#### 4.Change Permission
```
sudo chown yaseralie:yaseralie /etc/ssl/open-llm-vtuber/server.key
sudo chmod 600 /etc/ssl/open-llm-vtuber/server.key
sudo chown yaseralie:yaseralie /etc/ssl/open-llm-vtuber/server.crt
sudo chmod 644 /etc/ssl/open-llm-vtuber/server.crt
```

#### 5.Change configuration in frontend page
WebSocket URL:
```
wss://103.171.85.170:12393/client-ws
```

Base URL:
```
https://103.171.85.170:12393
```

#### AI Assistant With Avatar in OEE Monitoring System Open-LLM-VTuber
<p align="center">
  <a href="https://www.youtube.com/watch?v=8UlB-iYJ-lQ" target="_blank">
    <img src="https://img.youtube.com/vi/8UlB-iYJ-lQ/0.jpg" alt="YouTube Video Thumbnail" width="480" />
  </a>
</p>
Click the image above to watch the video

##### Replace file src/open_llm_vtuber/conversations/single_conversation.py
##### Run Server
```
cd Open-LLM-VTuber
python3 -m venv venv
source venv/bin/activate
python3 run_server.py
```

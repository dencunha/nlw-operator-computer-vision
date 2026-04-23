const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const qualitySlider = document.getElementById("quality-slider");
const qualityValue = document.getElementById("quality-value");
const showLandmarks = document.getElementById("show-landmarks");

if (qualitySlider) {
    qualitySlider.oninput = () => {
        qualityValue.innerText = qualitySlider.value;
    };
}

if (canvas) {
    const ctx = canvas.getContext('2d');
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    let ws;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                initWS();
            };
        })
        .catch(err => console.error("Erro na câmera: ", err));

    function initWS() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onmessage = (event) => {
            // Desempacota o JSON enviado pelo Python
            const data = JSON.parse(event.data);

            const img = new Image();
            img.onload = () => {
                // 1. Pinta a imagem limpa no Canvas
                ctx.drawImage(img, 0, 0);

                // 2. Atualiza o FPS
                const fpsCounter = document.getElementById("fps-counter");
                if (fpsCounter && data.fps !== undefined) {
                    fpsCounter.innerText = `${data.fps} FPS`;
                }

                // 3. Monta o LIVE FEED DATA organizado
                const container = document.getElementById("gesture-container");
                container.innerHTML = "";

                if (data.labels.length === 0) {
                    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem; text-align: center; margin-top: 1rem; font-style: italic;">Aguardando...</div>`;
                } else {
                    data.labels.forEach(label => {
                        // Converte a probabilidade para porcentagem (ex: 69.0%)
                        const percent = (label.probability * 100).toFixed(1) + "%";

                        // Cria a linha (Row)
                        const row = document.createElement("div");
                        row.className = "feed-row";

                        // Lado esquerdo (Ex: Right: paz)
                        const nameDiv = document.createElement("div");
                        nameDiv.className = "feed-label";
                        nameDiv.innerText = `${label.hand}: ${label.gesture}`;

                        // Lado direito (Ex: 69.0%)
                        const probDiv = document.createElement("div");
                        probDiv.className = "feed-value";
                        probDiv.innerText = percent;

                        row.appendChild(nameDiv);
                        row.appendChild(probDiv);
                        container.appendChild(row);
                    });
                }

                // 3. Mostra a imagem de gesto no DETECTED GESTURE
                const gestureImg = document.getElementById("gesture-image");
                if (data.matching_image) {
                    gestureImg.src = `/assets/images/gestures/${data.matching_image}`;
                    gestureImg.style.display = "block";
                } else {
                    gestureImg.style.display = "none";
                }

                // 4. Pede a próxima foto
                setTimeout(sendFrame, 33);
            };
            img.src = data.image;
        };

        ws.onopen = sendFrame;
        ws.onclose = () => setTimeout(initWS, 1000);
    }

    function sendFrame() {
        if (ws && ws.readyState === WebSocket.OPEN && video.videoWidth > 0) {
            tempCtx.drawImage(video, 0, 0);
            const quality = parseFloat(qualitySlider.value);
            const show = showLandmarks.checked;
            
            ws.send(JSON.stringify({ 
                "image": tempCanvas.toDataURL("image/jpeg", quality),
                "show_landmarks": show,
                "quality": quality
            }));
        }
    }
}
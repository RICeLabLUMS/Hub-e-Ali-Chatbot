let activeSource = 'all';

function setSource(btn, src) {
    document.querySelectorAll('.ha-src-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeSource = src;
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { 
        e.preventDefault(); 
        sendMsg(); 
    }
}

async function sendMsg() {
    const inp = document.getElementById('ha-input');
    const text = inp.value.trim();
    if (!text) return;

    const msgs = document.getElementById('ha-messages');
    
    // 1. Add User Bubble Directly
    const userDiv = document.createElement('div');
    userDiv.className = 'ha-msg user';
    userDiv.innerHTML = `<div class="ha-avatar">U</div><div class="ha-bubble">${text}</div>`;
    msgs.appendChild(userDiv);
    
    inp.value = '';
    msgs.scrollTop = msgs.scrollHeight;

    // 2. Add Typing Indicator Directly
    const typingDiv = document.createElement('div');
    typingDiv.className = 'ha-msg';
    typingDiv.innerHTML = `
        <div class="ha-avatar">☽</div>
        <div class="ha-bubble">
            <div class="ha-typing">
                <div class="ha-dot"></div>
                <div class="ha-dot"></div>
                <div class="ha-dot"></div>
            </div>
        </div>`;
    msgs.appendChild(typingDiv);
    msgs.scrollTop = msgs.scrollHeight;

    // 3. Call FastAPI Backend
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, source: activeSource })
        });
        
        const data = await response.json();
        typingDiv.remove();

        // 4. Add Bot Response Directly
        const respDiv = document.createElement('div');
        respDiv.className = 'ha-msg';
        respDiv.innerHTML = `
            <div class="ha-avatar">☽</div>
            <div>
                <div class="ha-label">${data.source_label}</div>
                <div class="ha-bubble">${data.answer}</div>
            </div>`;
        msgs.appendChild(respDiv);
        msgs.scrollTop = msgs.scrollHeight;

    } catch (error) {
        typingDiv.remove();
        console.error("Error:", error);
    }
}
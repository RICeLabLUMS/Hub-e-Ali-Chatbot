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
                <div class="ha-bubble">${escapeHtml(data.answer)}</div>
                ${renderCitations(data.citations)}
            </div>`;
        msgs.appendChild(respDiv);
        msgs.scrollTop = msgs.scrollHeight;

    } catch (error) {
        typingDiv.remove();
        console.error("Error:", error);
    }
}

function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderCitations(citations) {
    if (!citations || !citations.length) return '';
    const items = citations.map(c => {
        const title = escapeHtml(c.title || 'Source');
        const type = c.content_type ? `<span class="ha-cite-type">${escapeHtml(c.content_type)}</span>` : '';
        const page = (c.page != null && String(c.content_type).toUpperCase() === 'PDF')
            ? `<span class="ha-cite-page">p.&nbsp;${escapeHtml(c.page)}</span>` : '';
        const inner = `${title}${type ? ' &middot; ' + type : ''}${page ? ' &middot; ' + page : ''}`;
        return c.url
            ? `<a class="ha-cite" href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${inner}</a>`
            : `<span class="ha-cite">${inner}</span>`;
    }).join('');
    return `<div class="ha-citations">${items}</div>`;
}
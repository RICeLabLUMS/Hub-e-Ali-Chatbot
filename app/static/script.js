let activeSource = 'all';

// Backend base URL injected by the template via window.HA_CONFIG. Empty in the
// default same-origin deployment - apiUrl() then returns a plain relative path
// like "/chat". When the widget is embedded on a different domain, the server
// sets PUBLIC_BACKEND_URL and apiUrl() emits the absolute URL.
const HA_BACKEND_URL = (window.HA_CONFIG && window.HA_CONFIG.backendUrl) || '';
function apiUrl(path) {
    const p = path.startsWith('/') ? path : '/' + path;
    return HA_BACKEND_URL ? (HA_BACKEND_URL.replace(/\/+$/, '') + p) : p;
}

// Persist a per-browser session id so multi-turn rewrite_query has the right
// history scoped to this user (and not to whatever other user happened to be
// chatting at the same time on the server).
const HA_SESSION_KEY = 'hubeali.session_id';
function getSessionId() {
    try { return localStorage.getItem(HA_SESSION_KEY); } catch (_) { return null; }
}
function setSessionId(id) {
    if (!id) return;
    try { localStorage.setItem(HA_SESSION_KEY, id); } catch (_) {}
}

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
        const response = await fetch(apiUrl('/chat'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                source: activeSource,
                session_id: getSessionId(),
            })
        });

        const data = await response.json();
        // Server echoes (and on first call generates) a session id - persist
        // it so this browser keeps the same conversation thread.
        setSessionId(data.session_id);
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
        const ct = String(c.content_type || '');
        const ctU = ct.toUpperCase();

        const bits = [];
        if (c.content_type) bits.push(`<span class="ha-cite-type">${escapeHtml(c.content_type)}</span>`);
        if (c.volume != null) bits.push(`<span class="ha-cite-num">Vol.&nbsp;${escapeHtml(c.volume)}</span>`);
        if (c.page != null && ctU === 'PDF') {
            bits.push(`<span class="ha-cite-page">p.&nbsp;${escapeHtml(c.page)}</span>`);
        }
        if (c.section_title) bits.push(`<span class="ha-cite-num">§${escapeHtml(c.section_title)}</span>`);
        else if (c.chapter_num != null && c.verse_range) {
            bits.push(`<span class="ha-cite-num">Ch.&nbsp;${escapeHtml(c.chapter_num)} v.&nbsp;${escapeHtml(c.verse_range)}</span>`);
        } else if (c.chapter_num != null) {
            bits.push(`<span class="ha-cite-num">Ch.&nbsp;${escapeHtml(c.chapter_num)}</span>`);
        }
        const refs = Array.isArray(c.refs_quran) ? c.refs_quran : [];
        if (refs.length) {
            const shown = refs.slice(0, 3).map(escapeHtml).join(', ');
            const more = refs.length > 3 ? ` +${refs.length - 3}` : '';
            bits.push(`<span class="ha-cite-num">Quran&nbsp;${shown}${more}</span>`);
        }
        const hadith = Array.isArray(c.hadith_refs) ? c.hadith_refs : [];
        if (hadith.length) {
            const shown = hadith.slice(0, 3).map(escapeHtml).join(', ');
            const more = hadith.length > 3 ? ` +${hadith.length - 3}` : '';
            bits.push(`<span class="ha-cite-num">${shown}${more}</span>`);
        }

        const inner = title + (bits.length ? ' &middot; ' + bits.join(' &middot; ') : '');
        return c.url
            ? `<a class="ha-cite" href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${inner}</a>`
            : `<span class="ha-cite">${inner}</span>`;
    }).join('');
    return `<div class="ha-citations">${items}</div>`;
}
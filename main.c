/* =========================================================================
 * src/main.c — FULL NETNAVI DAEMON SERVER, CONVERSATIONAL ENGINE & WEB HUD
 * ========================================================================= */
#include "../include/navi.h"

static const char *NAVI_HUD_HTML =
"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
"<!DOCTYPE html><html><head><meta charset='utf-8'><title>NAVI_CORE // PET TERMINAL</title>"
"<style>"
"body{background:#060a0e;color:#00ffcc;font-family:'Courier New',monospace;margin:0;display:flex;height:100vh;overflow:hidden;}"
"#left-panel{width:42%;border-right:2px solid #005544;display:flex;flex-direction:column;padding:16px;box-sizing:border-box;background:#04070a;}"
"#right-panel{width:58%;display:flex;flex-direction:column;padding:16px;box-sizing:border-box;}"
".panel-title{font-size:13px;letter-spacing:2px;color:#00aa88;border-bottom:1px solid #004433;padding-bottom:6px;margin-bottom:10px;}"
"canvas{background:#020406;border:1px solid #006655;border-radius:6px;width:100%;margin-bottom:12px;}"
"#telemetry{font-size:12px;line-height:1.6;background:#081016;padding:10px;border:1px solid #004433;border-radius:4px;flex:1;overflow-y:auto;}"
"#chat-log{flex:1;overflow-y:auto;border:1px solid #004433;background:#080d12;padding:12px;border-radius:4px;margin-bottom:12px;}"
".msg-op{color:#ffcc00;margin-bottom:8px;} .msg-navi{color:#00ffcc;margin-bottom:12px;border-left:2px solid #00ffcc;padding-left:8px;}"
"#input-row{display:flex;gap:8px;}"
"#cmd-in{flex:1;background:#030609;color:#00ffcc;border:1px solid #008866;padding:10px;font-family:inherit;font-size:14px;border-radius:4px;outline:none;}"
"button{background:#005544;color:#00ffcc;border:1px solid #00ffcc;padding:10px 18px;font-family:inherit;cursor:pointer;font-weight:bold;border-radius:4px;}"
".badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:bold;margin-right:6px;}"
".b-0{background:#004422;color:#00ff88;} .b-1{background:#443300;color:#ffcc00;}"
".b-2{background:#551111;color:#ff5555;} .b-3{background:#330033;color:#ff00ff;}"
"</style></head><body>"
"<div id='left-panel'>"
"  <div class='panel-title'>2D LISSAJOUS AVATAR CORE (F1 vs F2 PHASE)</div>"
"  <canvas id='scope-xy' width='360' height='240'></canvas>"
"  <div class='panel-title'>TIME-DOMAIN ACOUSTIC WAVE y(t) + CLICK SPIKES</div>"
"  <canvas id='scope-t' width='360' height='120'></canvas>"
"  <div class='panel-title'>SUBCONSCIOUS COGNITION & DIALECT TELEMETRY</div>"
"  <div id='telemetry'>Awaiting Operator input...</div>"
"</div>"
"<div id='right-panel'>"
"  <div class='panel-title'>NETNAVI COPILOT // 3-MAP PHONETIC-TO-SENTENCE ENGINE</div>"
"  <div id='chat-log'>"
"    <div class='msg-navi'><b>[NAVI_CORE]:</b> 3-Map Substrate + UTF-8 Non-Latin + Conversational Engine Online, Sir Gamez. Try English, Spanish, Mandarin (<i>ni hao</i> or <i>你好</i>), or teach new chains!</div>"
"  </div>"
"  <div id='input-row'>"
"    <input id='cmd-in' type='text' placeholder='Transmit sentence...' autofocus/>"
"    <button onclick='sendCmd()'>TRANSMIT</button>"
"  </div>"
"</div>"
"<script>"
"let wavePCM=new Array(128).fill(0),scopeX=new Array(128).fill(0),scopeY=new Array(128).fill(0),phase=0;"
"const ctxXY=document.getElementById('scope-xy').getContext('2d'),ctxT=document.getElementById('scope-t').getContext('2d');"
"function drawScopes(){"
"  phase+=0.06; ctxXY.fillStyle='rgba(2,4,6,0.22)'; ctxXY.fillRect(0,0,360,240);"
"  ctxXY.strokeStyle='#00ffcc'; ctxXY.lineWidth=1.8; ctxXY.beginPath();"
"  for(let i=0;i<wavePCM.length;i++){"
"    let x=180+(scopeX[i]*Math.cos(phase*0.5)-scopeY[i]*0.2)*95, y=120+(scopeY[i]*Math.sin(phase*0.5)+scopeX[i]*0.2)*75;"
"    if(i===0)ctxXY.moveTo(x,y); else ctxXY.lineTo(x,y);"
"  } ctxXY.stroke();"
"  ctxT.fillStyle='#020406'; ctxT.fillRect(0,0,360,120); ctxT.strokeStyle='#00ff88'; ctxT.lineWidth=1.5; ctxT.beginPath();"
"  for(let i=0;i<wavePCM.length;i++){"
"    let x=(i/wavePCM.length)*360, y=60-wavePCM[i]*48;"
"    if(i===0)ctxT.moveTo(x,y); else ctxT.lineTo(x,y);"
"  } ctxT.stroke(); requestAnimationFrame(drawScopes);"
"} drawScopes();"
"async function sendCmd(){"
"  const inp=document.getElementById('cmd-in'), txt=inp.value.trim(); if(!txt)return;"
"  const log=document.getElementById('chat-log');"
"  log.innerHTML+=`<div class='msg-op'><b>[SIR GAMEZ]:</b> ${txt}</div>`; inp.value='';"
"  const res=await fetch('/api?msg='+encodeURIComponent(txt)), data=await res.json();"
"  wavePCM=data.pcm; scopeX=data.sx; scopeY=data.sy;"
"  const bNames=['ATTESTED (SNAP)','UNATTESTED (LEARNED)','ILLICIT (VIOLATION)','CORRUPTED (TRAP)'];"
"  log.innerHTML+=`<div class='msg-navi'><span class='badge b-${data.state}'>${bNames[data.state]}</span><b>[NAVI]:</b> ${data.spoken}<br><small>${data.reply}</small></div>`;"
"  document.getElementById('telemetry').innerHTML=data.telemetry; log.scrollTop=log.scrollHeight;"
"}"
"document.getElementById('cmd-in').addEventListener('keydown',e=>{if(e.key==='Enter')sendCmd();});"
"</script></body></html>";

static void url_decode(char *dst, const char *src, size_t max_len) {
    size_t i = 0;
    while (*src && i + 1 < max_len) {
        if (*src == '%' && isxdigit((unsigned char)src[1]) && isxdigit((unsigned char)src[2])) {
            char hex[3] = { src[1], src[2], 0 };
            dst[i++] = (char)strtol(hex, NULL, 16);
            src += 3;
        } else if (*src == '+') {
            dst[i++] = ' ';
            src++;
        } else {
            dst[i++] = *src++;
        }
    }
    dst[i] = '\0';
}

/* UTF-8 Aware Token Sanitizer: Preserves multi-byte Non-Latin characters */
static void sanitize_token_utf8(char *dst, const char *src, size_t max_len) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 1 < max_len; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"' || c == '\\' || c == '<' || c == '>' ||
            c == '?' || c == '!' || c == '.' || c == ',' || c == ';' || c == ':') {
            continue;
        }
        dst[j++] = (c < 128) ? (char)tolower(c) : (char)c;
    }
    dst[j] = '\0';
}

static void format_lang_tags(uint16_t mask, int preferred_lang, char *out_buf, size_t max_len) {
    out_buf[0] = '\0';
    int count = 0;
    if (preferred_lang >= 0 && preferred_lang < LANG_COUNT && (mask & (1U << preferred_lang))) {
        strncat(out_buf, LANG_NAMES[preferred_lang], max_len - strlen(out_buf) - 1);
        count++;
    }
    for (int i = 0; i < LANG_COUNT; i++) {
        if (i == preferred_lang) continue;
        if (mask & (1U << i)) {
            if (count > 0) strncat(out_buf, "|", max_len - strlen(out_buf) - 1);
            strncat(out_buf, LANG_NAMES[i], max_len - strlen(out_buf) - 1);
            if (++count >= 3) break;
        }
    }
    if (count == 0) strncpy(out_buf, "LRN", max_len - 1);
}

/* Extended Conversational Dhatu Classifier (catches follow-ups, questions, diagnostics) */
static uint16_t classify_extended_dhatu(const char *word) {
    uint16_t base = subconscious_classify_dhatu(word);
    if (base != DHATU_UNKNOWN) return base;

    if (!strcmp(word, "hey") || !strcmp(word, "yo") || !strcmp(word, "morning") ||
        !strcmp(word, "buenos") || !strcmp(word, "dias")) {
        return DHATU_GREET;
    }
    if (!strcmp(word, "why") || !strcmp(word, "where") || !strcmp(word, "who") ||
        !strcmp(word, "when") || !strcmp(word, "answer") || !strcmp(word, "question") ||
        !strcmp(word, "really") || !strcmp(word, "think") || !strcmp(word, "qué")) {
        return DHATU_QUERY;
    }
    if (!strcmp(word, "good") || !strcmp(word, "great") || !strcmp(word, "fine") ||
        !strcmp(word, "well") || !strcmp(word, "doing") || !strcmp(word, "feeling") ||
        !strcmp(word, "bien") || !strcmp(word, "todo")) {
        return DHATU_BEING;
    }
    if (!strcmp(word, "yourself") || !strcmp(word, "navi") || !strcmp(word, "myself") ||
        !strcmp(word, "sir") || !strcmp(word, "operator")) {
        return DHATU_IDENTITY;
    }
    if (!strcmp(word, "nodes") || !strcmp(word, "matrix") || !strcmp(word, "save") ||
        !strcmp(word, "train") || !strcmp(word, "teach") || !strcmp(word, "substrate")) {
        return DHATU_COMPUTE;
    }
    return DHATU_UNKNOWN;
}

/* Walk Map 3 edges starting from the best trigger node in the user's sentence */
static void walk_best_map3_chain(const int32_t *sent_nodes, int node_count, char *out_chain, size_t max_len) {
    out_chain[0] = '\0';
    int32_t seed_node = -1;

    /* Check from the end of the sentence backwards for any node that has outgoing Map 3 links */
    for (int i = node_count - 1; i >= 0; i--) {
        char probe[32] = {0};
        if (sent_nodes[i] > 0 && navi_map3_predict_next(sent_nodes[i], probe, sizeof(probe)) > 0) {
            seed_node = sent_nodes[i];
            break;
        }
    }
    if (seed_node <= 0) return;

    int32_t curr = seed_node;
    int32_t visited[8] = {0};
    for (int step = 0; step < 7 && curr > 0; step++) {
        /* Prevent infinite loops on circular bigrams */
        bool loop = false;
        for (int v = 0; v < step; v++) {
            if (visited[v] == curr) { loop = true; break; }
        }
        if (loop) break;
        visited[step] = curr;

        char next_w[32] = {0};
        int32_t next_idx = navi_map3_predict_next(curr, next_w, sizeof(next_w));
        if (next_idx <= 0 || strcmp(next_w, "none") == 0) break;

        strncat(out_chain, next_w, max_len - strlen(out_chain) - 1);
        strncat(out_chain, " ", max_len - strlen(out_chain) - 1);
        curr = next_idx;
    }
}

/* Generates a clean JSON-safe conversational response (NO unescaped double quotes!) */
static void generate_spoken_response(uint16_t dhatu, int lang, const int32_t *sent_nodes,
                                     int node_count, const char *chain_words,
                                     char *out_spoken, size_t max_len) {
    char chain[256] = {0};
    walk_best_map3_chain(sent_nodes, node_count, chain, sizeof(chain));

    if (lang == LANG_CMN) {
        if (dhatu == DHATU_GREET) {
            snprintf(out_spoken, max_len, "你好 (Nǐ hǎo), Sir Gamez! 系统在线 (System Online). %s",
                     chain[0] ? chain : "有什么命令 (Awaiting command)?");
        } else {
            snprintf(out_spoken, max_len, "收到 (Understood), Sir Gamez — '%s' -> %s",
                     chain_words, chain[0] ? chain : "已记录 (Logged in Map 3).");
        }
    } else if (lang == LANG_SPA) {
        if (dhatu == DHATU_GREET) {
            snprintf(out_spoken, max_len, "¡Hola, Sir Gamez! Sistemas listos — %s",
                     chain[0] ? chain : "¿cómo estás hoy?");
        } else if (dhatu == DHATU_QUERY || dhatu == DHATU_BEING || dhatu == DHATU_IDENTITY) {
            snprintf(out_spoken, max_len, "Operando al cien por ciento, Sir Gamez. %s",
                     chain[0] ? chain : "Núcleo fonético estable.");
        } else {
            snprintf(out_spoken, max_len, "Entendido, Sir Gamez: '%s' %s",
                     chain_words, chain[0] ? chain : "(vinculado en Mapa 3).");
        }
    } else if (lang == LANG_FRA) {
        snprintf(out_spoken, max_len, "Bonjour, Sir Gamez! Navigateur actif — %s",
                 chain[0] ? chain : "prêt pour vos ordres.");
    } else {
        switch (dhatu) {
            case DHATU_GREET:
                snprintf(out_spoken, max_len, "Salutations, Sir Gamez. Systems online — %s",
                         chain[0] ? chain : "standing by for your command.");
                break;
            case DHATU_BEING:
                snprintf(out_spoken, max_len, "Glad to hear it, Sir Gamez. I am operating at peak efficiency (%u Map 3 links). %s",
                         GLOBAL_MAP3.edge_count, chain[0] ? chain : "All acoustic and matrix channels nominal.");
                break;
            case DHATU_QUERY:
                if (chain[0]) {
                    snprintf(out_spoken, max_len, "Query resolved via Map 3 chain: %s(linked from '%s')",
                             chain, chain_words);
                } else {
                    snprintf(out_spoken, max_len, "Direct answer: My Map 3 graph currently holds %u transition links. Teach me a new association with 'teach: word1 word2' or keep chatting so I wire more paths!",
                             GLOBAL_MAP3.edge_count);
                }
                break;
            case DHATU_IDENTITY:
                snprintf(out_spoken, max_len, "I am your local NetNavi operating core, Sir Gamez — running %d Trie nodes across 16 languages. %s",
                         GLOBAL_TRIE.node_count, chain[0] ? chain : "Ready when you are.");
                break;
            case DHATU_COMPUTE:
                snprintf(out_spoken, max_len, "Diagnostics: %d Trie nodes | %u Map 3 transitions | %u Medals. %s",
                         GLOBAL_TRIE.node_count, GLOBAL_MAP3.edge_count, GLOBAL_MEDAL.count, chain);
                break;
            case DHATU_LEARNED:
                snprintf(out_spoken, max_len, "Assimilated new vocabulary into [%s] substrate: '%s' -> %s",
                         LANG_NAMES[lang], chain_words,
                         chain[0] ? chain : "synapse recorded in Map 3.");
                break;
            default:
                if (chain[0]) {
                    snprintf(out_spoken, max_len, "%s-> %s", chain_words, chain);
                } else {
                    snprintf(out_spoken, max_len, "Synapse wired for '%s' (%u total Map 3 links). Keep chaining or type 'save' to lock into navi_base.bin.",
                             chain_words, GLOBAL_MAP3.edge_count);
                }
                break;
        }
    }
}

void handle_chat_query(const char *raw_msg, char *out_http_buf, size_t buf_size) {
    char decoded[256];
    url_decode(decoded, raw_msg, sizeof(decoded));

    /* Special Command: 'save' persists all newly learned Trie nodes & Map 3 links to disk */
    if (strcmp(decoded, "save") == 0) {
        navi_save_substrate_state("data/navi_base.bin");
        snprintf(out_http_buf, buf_size,
                 "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n"
                 "{\"state\":0,\"dhatu\":105,\"lang\":\"ENG\","
                 "\"spoken\":\"Substrate state saved to data/navi_base.bin (%d nodes, %u Map 3 links).\","
                 "\"reply\":\"[DISK COMMIT OK]\",\"telemetry\":\"Saved\",\"pcm\":[0],\"sx\":[0],\"sy\":[0]}",
                 GLOBAL_TRIE.node_count, GLOBAL_MAP3.edge_count);
        return;
    }

    /* Step 1: Subconscious Sentence-Level Language Consensus Identification */
    int sentence_lang = subconscious_detect_sentence_lang(&GLOBAL_TRIE, decoded, GLOBAL_LANG_STATS.active_lang);
    GLOBAL_LANG_STATS.active_lang = sentence_lang;

    char reply[1200] = {0};
    char clean_sentence[256] = {0};
    char ipa_sentence[512] = {0};
    char telemetry[1200] = {0};
    int primary_state = STATE_ATTESTED;
    uint32_t synth_mask = FEAT_SYLLABIC | FEAT_VOICE;
    uint8_t synth_tone = TONE_NEUTRAL;
    uint16_t dominant_dhatu = DHATU_UNKNOWN;

    char *token = strtok(decoded, " \t\r\n");
    int token_count = 0;
    int32_t prev_node_idx = -1;
    int32_t last_node_idx = -1;
    int32_t sentence_nodes[16] = {0};

    while (token != NULL && token_count < 16) {
        char word[64] = {0};
        sanitize_token_utf8(word, token, sizeof(word));
        uint8_t len = (uint8_t)strlen(word);

        /* Skip command prefix word 'teach' so it only links the actual phrase */
        if (token_count == 0 && strcmp(word, "teach") == 0) {
            dominant_dhatu = DHATU_LEARNED;
            token = strtok(NULL, " \t\r\n");
            continue;
        }

        if (len > 0) {
            uint32_t masks[64];
            uint8_t  tones[64];
            for (uint8_t i = 0; i < len; i++) {
                masks[i] = char_to_phoible_mask(word[i], &tones[i]);
            }

            NaviLookupResult res = navi_resolve_word(&GLOBAL_TRIE, masks, word, len);

            if (res.resolved_state == STATE_ATTESTED && res.matched_node_idx > 0) {
                NaviTrieNode *n = &GLOBAL_TRIE.nodes[res.matched_node_idx];
                synth_mask |= n->phoible_mask;
                synth_tone  = n->tone;

                uint16_t word_dhatu = classify_extended_dhatu(res.matched_word);
                /* Prioritize specific conversational intents (Query/Identity/Being) over generic Greet */
                if (word_dhatu != DHATU_UNKNOWN) {
                    if (dominant_dhatu == DHATU_UNKNOWN || word_dhatu == DHATU_QUERY || word_dhatu == DHATU_IDENTITY) {
                        dominant_dhatu = word_dhatu;
                    }
                }

                if (n->lang_mask & (1U << sentence_lang)) {
                    GLOBAL_LANG_STATS.hits[sentence_lang]++;
                }
                for (int l = 0; l < LANG_COUNT; l++) {
                    if (l != sentence_lang && (n->lang_mask & (1U << l))) {
                        GLOBAL_LANG_STATS.hits[l]++;
                    }
                }

                char ltags[32];
                format_lang_tags(n->lang_mask, sentence_lang, ltags, sizeof(ltags));

                const char *glyph = n->ipa_utf8[0] ? n->ipa_utf8 : res.matched_word;
                char part[180];
                snprintf(part, sizeof(part), "[%s &rarr; <b>%s</b> %s (%s) %.2f] ",
                         word, res.matched_word, glyph, ltags, res.best_error_cost);
                strncat(reply, part, sizeof(reply) - strlen(reply) - 1);

                strncat(clean_sentence, res.matched_word, sizeof(clean_sentence) - strlen(clean_sentence) - 1);
                strncat(clean_sentence, " ", sizeof(clean_sentence) - strlen(clean_sentence) - 1);

                strncat(ipa_sentence, glyph, sizeof(ipa_sentence) - strlen(ipa_sentence) - 1);
                strncat(ipa_sentence, " ", sizeof(ipa_sentence) - strlen(ipa_sentence) - 1);

                if (prev_node_idx > 0) {
                    navi_map3_record_transition(prev_node_idx, res.matched_node_idx, res.matched_word);
                }
                sentence_nodes[token_count] = res.matched_node_idx;
                prev_node_idx = res.matched_node_idx;
                last_node_idx = res.matched_node_idx;
            } else {
                char auto_ipa[16];
                snprintf(auto_ipa, sizeof(auto_ipa), "/%.12s/", word);
                uint16_t learn_mask = (uint16_t)((1U << sentence_lang) | (1U << LANG_OTH));
                int32_t new_idx = navi_trie_insert(&GLOBAL_TRIE, masks, word, tones, len, auto_ipa, learn_mask);

                CognitiveBridiTuple fact = navi_subconscious_parse(1, DHATU_LEARNED, (uint32_t)new_idx,
                                                                   (uint32_t)prev_node_idx, (uint32_t)sentence_lang, 1);
                navi_medal_commit(&GLOBAL_MEDAL, fact, (uint32_t)new_idx, 1);

                primary_state = STATE_UNATTESTED;
                if (dominant_dhatu == DHATU_UNKNOWN) dominant_dhatu = DHATU_LEARNED;
                GLOBAL_LANG_STATS.hits[LANG_OTH]++;

                char part[128];
                snprintf(part, sizeof(part), "[<b>%s</b>: LEARNED(#%d &rarr; %s)] ",
                         word, new_idx, LANG_NAMES[sentence_lang]);
                strncat(reply, part, sizeof(reply) - strlen(reply) - 1);

                strncat(clean_sentence, word, sizeof(clean_sentence) - strlen(clean_sentence) - 1);
                strncat(clean_sentence, " ", sizeof(clean_sentence) - strlen(clean_sentence) - 1);

                strncat(ipa_sentence, auto_ipa, sizeof(ipa_sentence) - strlen(ipa_sentence) - 1);
                strncat(ipa_sentence, " ", sizeof(ipa_sentence) - strlen(ipa_sentence) - 1);

                if (prev_node_idx > 0 && new_idx > 0) {
                    navi_map3_record_transition(prev_node_idx, new_idx, word);
                }
                sentence_nodes[token_count] = new_idx;
                prev_node_idx = new_idx;
                last_node_idx = new_idx;
            }
            token_count++;
        }
        token = strtok(NULL, " \t\r\n");
    }

    char predicted_word[32] = "none";
    int32_t pred_node = -1;
    if (last_node_idx > 0) {
        pred_node = navi_map3_predict_next(last_node_idx, predicted_word, sizeof(predicted_word));
    }

    char spoken_reply[512] = {0};
    generate_spoken_response(dominant_dhatu, sentence_lang, sentence_nodes,
                             token_count, clean_sentence, spoken_reply, sizeof(spoken_reply));

    CognitiveBridiTuple sentence_bridi = navi_subconscious_parse(
        1, dominant_dhatu, (uint32_t)(last_node_idx > 0 ? last_node_idx : 0),
        (uint32_t)(pred_node > 0 ? pred_node : 0), (uint32_t)sentence_lang, 1
    );

    char extra_info[350];
    snprintf(extra_info, sizeof(extra_info),
             "<br><b>Phonetics/Glyphs:</b> <code>%s</code> | <b>Next-Token:</b> <code>%s</code>",
             ipa_sentence, predicted_word);
    strncat(reply, extra_info, sizeof(reply) - strlen(reply) - 1);

    snprintf(telemetry, sizeof(telemetry),
             "<b>Subconscious Lang ID (x4):</b> [%s] | <b>Dhatu/Selbri:</b> %s<br>"
             "<b>Bridi Tuple:</b> [x1:Op, x2:Node_%u, x3:Next_%u, x4:Lang_%s]<br>"
             "<b>US English [ENG]:</b> %u | <b>Spanish [SPA]:</b> %u | <b>French [FRA]:</b> %u<br>"
             "<b>Mandarin [CMN]:</b> %u | <b>Map 3 Edges:</b> %u | <b>Medals:</b> %u",
             LANG_NAMES[sentence_lang], subconscious_dhatu_name(sentence_bridi.selbri_relation),
             sentence_bridi.sumti_x[1], sentence_bridi.sumti_x[2], LANG_NAMES[sentence_lang],
             GLOBAL_LANG_STATS.hits[LANG_ENG], GLOBAL_LANG_STATS.hits[LANG_SPA],
             GLOBAL_LANG_STATS.hits[LANG_FRA], GLOBAL_LANG_STATS.hits[LANG_CMN],
             GLOBAL_MAP3.edge_count, GLOBAL_MEDAL.count);

    AcousticSynthFrame frame = navi_features_to_formants(synth_mask, synth_tone);
    OscilloscopeBuffer scope;
    navi_render_oscilloscope(&frame, synth_tone, &scope);

    char pcm_str[1536] = {0}, sx_str[1536] = {0}, sy_str[1536] = {0};
    size_t p_off = 0, x_off = 0, y_off = 0;
    for (int i = 0; i < 128; i++) {
        p_off += snprintf(pcm_str + p_off, sizeof(pcm_str) - p_off, "%s%.3f", (i ? "," : ""), scope.pcm_samples[i]);
        x_off += snprintf(sx_str + x_off,  sizeof(sx_str) - x_off,  "%s%.3f", (i ? "," : ""), scope.scope_x[i]);
        y_off += snprintf(sy_str + y_off,  sizeof(sy_str) - y_off,  "%s%.3f", (i ? "," : ""), scope.scope_y[i]);
    }

    snprintf(out_http_buf, buf_size,
             "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n"
             "{\"state\":%d,\"dhatu\":%u,\"lang\":\"%s\",\"spoken\":\"%s\",\"reply\":\"%s\",\"telemetry\":\"%s\",\"pcm\":[%s],\"sx\":[%s],\"sy\":[%s]}",
             primary_state, (unsigned)dominant_dhatu, LANG_NAMES[sentence_lang],
             spoken_reply, reply, telemetry, pcm_str, sx_str, sy_str);
}

int main(int argc, char **argv) {
    seed_initial_substrate();
    if (argc > 1 && strcmp(argv[1], "--compile") == 0) {
        printf("[+] Base state compilation complete. Exiting cleanly.\n");
        return 0;
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port        = htons(8080);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        printf("[!] Port 8080 in use! Run 'killall navi_core' first.\n");
        return 1;
    }
    listen(server_fd, 5);
    printf("============================================================\n");
    printf("  NETWORK NAVIGATOR ONLINE // http://127.0.0.1:8080\n");
    printf("  Node Size: %zu Bytes | Active Nodes: %d | Sentence Edges: %u\n",
           sizeof(NaviTrieNode), GLOBAL_TRIE.node_count, GLOBAL_MAP3.edge_count);
    printf("============================================================\n");

    char req_buf[2048], resp_buf[8192];
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) continue;

        int n = recv(client_fd, req_buf, sizeof(req_buf) - 1, 0);
        if (n > 0) {
            req_buf[n] = '\0';
            if (strncmp(req_buf, "GET /api?msg=", 13) == 0) {
                char *end = strstr(req_buf + 13, " HTTP/");
                if (!end) end = strchr(req_buf + 13, ' ');
                if (end) *end = '\0';
                handle_chat_query(req_buf + 13, resp_buf, sizeof(resp_buf));
                send(client_fd, resp_buf, (int)strlen(resp_buf), 0);
            } else {
                send(client_fd, NAVI_HUD_HTML, (int)strlen(NAVI_HUD_HTML), 0);
            }
        }
        closesocket(client_fd);
    }
    return 0;
}
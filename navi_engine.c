/* =========================================================================
 * src/navi_engine.c — FULL 3-MAP ENGINE, FUZZY DFS, UTF-8 & DATASETS
 * ========================================================================= */
#include "../include/navi.h"

NaviTriePool     GLOBAL_TRIE;
MedalCore        GLOBAL_MEDAL;
SentenceTransitionMap GLOBAL_MAP3 = {0};
LanguageTelemetryTracker GLOBAL_LANG_STATS = { .active_lang = LANG_ENG };

const char *LANG_NAMES[LANG_COUNT] = {
    "ENG", "ENK", "SPA", "FRA", "DEU", "ZUL", "XHO", "CMN",
    "YUE", "JPN", "KOR", "ARA", "FAS", "SWA", "EPO", "LRN"
};

static uint32_t ASCII_PHOIBLE_MAP[256];
static float KEY_COST_MATRIX[128][128];

typedef struct {
    char  key_char;
    float x;
    float y;
} QwertyKeyCoord;

static const QwertyKeyCoord QWERTY_GRID[] = {
    {'1',0.0f,0.0f},{'2',1.0f,0.0f},{'3',2.0f,0.0f},{'4',3.0f,0.0f},{'5',4.0f,0.0f},
    {'6',5.0f,0.0f},{'7',6.0f,0.0f},{'8',7.0f,0.0f},{'9',8.0f,0.0f},{'0',9.0f,0.0f},
    {'q',0.25f,1.0f},{'w',1.25f,1.0f},{'e',2.25f,1.0f},{'r',3.25f,1.0f},{'t',4.25f,1.0f},
    {'y',5.25f,1.0f},{'u',6.25f,1.0f},{'i',7.25f,1.0f},{'o',8.25f,1.0f},{'p',9.25f,1.0f},
    {'a',0.50f,2.0f},{'s',1.50f,2.0f},{'d',2.50f,2.0f},{'f',3.50f,2.0f},{'g',4.50f,2.0f},
    {'h',5.50f,2.0f},{'j',6.50f,2.0f},{'k',7.50f,2.0f},{'l',8.50f,2.0f},
    {'z',0.75f,3.0f},{'x',1.75f,3.0f},{'c',2.75f,3.0f},{'v',3.75f,3.0f},{'b',4.75f,3.0f},
    {'n',5.75f,3.0f},{'m',6.75f,3.0f}
};
#define QWERTY_KEY_COUNT (sizeof(QWERTY_GRID) / sizeof(QWERTY_GRID[0]))

int navi_popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#elif defined(_MSC_VER)
    return __popcnt(x);
#else
    x -= ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    return (((x + (x >> 4)) & 0x0F0F0F0FU) * 0x01010101U) >> 24;
#endif
}

uint32_t default_char_to_phoible_mask(char c, uint8_t *out_tone) {
    *out_tone = TONE_NEUTRAL;
    unsigned char uc = (unsigned char)c;
    if (uc >= 128) {
        /* Multi-byte UTF-8 non-Latin byte: assign dorsal/syllabic resonant signature */
        return FEAT_SYLLABIC | FEAT_SONORANT | FEAT_VOICE | FEAT_DORSAL;
    }
    uc = (unsigned char)tolower(uc);
    switch (uc) {
        case 'a': return FEAT_SYLLABIC | FEAT_SONORANT | FEAT_VOICE | FEAT_LOW | FEAT_BACK;
        case 'e': return FEAT_SYLLABIC | FEAT_SONORANT | FEAT_VOICE;
        case 'i': case 'y': return FEAT_SYLLABIC | FEAT_SONORANT | FEAT_VOICE | FEAT_HIGH;
        case 'o': return FEAT_SYLLABIC | FEAT_SONORANT | FEAT_VOICE | FEAT_BACK | FEAT_ROUND;
        case 'u': case 'w': return FEAT_SYLLABIC | FEAT_SONORANT | FEAT_VOICE | FEAT_HIGH | FEAT_BACK | FEAT_ROUND;
        case '*': return FEAT_SYLLABIC | FEAT_WILDCARD_V;
        case '|': return FEAT_VELARIC_CLICK | FEAT_CLICK_DENTAL   | FEAT_CORONAL;
        case '!': return FEAT_VELARIC_CLICK | FEAT_CLICK_ALVEOLAR | FEAT_CORONAL;
        case 'p': return FEAT_CONSONANTAL | FEAT_LABIAL;
        case 'b': return FEAT_CONSONANTAL | FEAT_LABIAL | FEAT_VOICE;
        case 'm': return FEAT_CONSONANTAL | FEAT_LABIAL | FEAT_SONORANT | FEAT_NASAL | FEAT_VOICE;
        case 'f': return FEAT_CONSONANTAL | FEAT_LABIAL | FEAT_CONTINUANT;
        case 'v': return FEAT_CONSONANTAL | FEAT_LABIAL | FEAT_CONTINUANT | FEAT_VOICE;
        case 't': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_ANTERIOR;
        case 'd': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_ANTERIOR | FEAT_VOICE;
        case 'n': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_SONORANT | FEAT_NASAL | FEAT_VOICE;
        case 's': case 'c': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_CONTINUANT | FEAT_STRIDENT;
        case 'z': case 'x': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_CONTINUANT | FEAT_STRIDENT | FEAT_VOICE;
        case 'l': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_SONORANT | FEAT_LATERAL | FEAT_VOICE;
        case 'r': return FEAT_CONSONANTAL | FEAT_CORONAL | FEAT_SONORANT | FEAT_VOICE;
        case 'k': case 'q': return FEAT_CONSONANTAL | FEAT_DORSAL;
        case 'g': case 'j': return FEAT_CONSONANTAL | FEAT_DORSAL | FEAT_VOICE;
        case 'h': return FEAT_CONTINUANT | FEAT_GLOTTALIC;
        default:  return FEAT_CONSONANTAL | FEAT_VOICE;
    }
}

void navi_compile_phoible_csv(const char *csv_path) {
    uint8_t dummy_tone;
    for (int i = 0; i < 256; i++) {
        ASCII_PHOIBLE_MAP[i] = default_char_to_phoible_mask((char)i, &dummy_tone);
    }
    FILE *fp = fopen(csv_path, "r");
    if (!fp) return;

    char line[2048];
    uint32_t rows = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (rows++ == 0) continue;
        char *p = strchr(line, ',');
        for (int col = 1; col < 6 && p; col++) p = strchr(p + 1, ',');
        if (!p) continue;
        unsigned char ch = (unsigned char)p[1];
        if (ch == '"' && p[2]) ch = (unsigned char)p[2];
        if (ch >= 'a' && ch <= 'z' && strstr(line, ",+,")) {
            ASCII_PHOIBLE_MAP[ch] |= default_char_to_phoible_mask((char)ch, &dummy_tone);
        }
        if (rows >= 25000) break;
    }
    fclose(fp);
    printf("[+] MAP 1 (PHOIBLE.CSV) | Compiled acoustic feature masks from %u rows\n", rows);
}

uint32_t char_to_phoible_mask(char c, uint8_t *out_tone) {
    *out_tone = TONE_NEUTRAL;
    unsigned char uc = (unsigned char)c;
    if (uc < 128) uc = (unsigned char)tolower(uc);
    return ASCII_PHOIBLE_MAP[uc];
}

uint32_t ipa_string_to_composite_mask(const char *ipa_str, uint8_t *out_tone) {
    uint32_t mask = 0;
    *out_tone = TONE_NEUTRAL;
    for (size_t i = 0; ipa_str[i]; i++) {
        unsigned char c = (unsigned char)ipa_str[i];
        if (c >= '1' && c <= '5') {
            *out_tone = (uint8_t)(c - '0');
        } else if (isalpha(c) || c >= 128) {
            uint8_t t;
            mask |= char_to_phoible_mask((char)c, &t);
        }
    }
    return mask ? mask : (FEAT_SYLLABIC | FEAT_VOICE);
}

void navi_init_key_matrix(void) {
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            KEY_COST_MATRIX[i][j] = (i == j) ? 0.0f : 1.0f;
        }
    }
    for (size_t i = 0; i < QWERTY_KEY_COUNT; i++) {
        for (size_t j = 0; j < QWERTY_KEY_COUNT; j++) {
            unsigned char k1 = (unsigned char)QWERTY_GRID[i].key_char;
            unsigned char k2 = (unsigned char)QWERTY_GRID[j].key_char;
            float dx = QWERTY_GRID[i].x - QWERTY_GRID[j].x;
            float dy = QWERTY_GRID[i].y - QWERTY_GRID[j].y;
            float dist = sqrtf(dx * dx + dy * dy);
            KEY_COST_MATRIX[k1][k2] = (dist <= 1.5f) ? (dist * 0.20f) : fminf(1.0f, dist * 0.35f);
        }
    }
}

static inline float navi_get_max_error_budget(uint8_t word_len) {
    if (word_len <= 3)  return 0.25f;
    if (word_len <= 6)  return 0.45f;
    if (word_len <= 10) return 0.70f;
    return 0.90f;
}

static inline float navi_sound_distance(uint32_t mask_a, uint32_t mask_b) {
    if (((mask_a & FEAT_WILDCARD_V) && (mask_b & FEAT_SYLLABIC)) ||
        ((mask_b & FEAT_WILDCARD_V) && (mask_a & FEAT_SYLLABIC))) {
        return 0.0f;
    }
    uint32_t diff = (mask_a ^ mask_b) & 0x3FFFFFFFU;
    return (float)navi_popcount32(diff) * 0.08f;
}

static inline float navi_dual_step_cost(uint32_t mask_a, char char_a,
                                        uint32_t mask_b, char char_b,
                                        bool is_first_phoneme) {
    unsigned char ua = (unsigned char)char_a;
    unsigned char ub = (unsigned char)char_b;
    if (ua < 128) ua = (unsigned char)tolower(ua);
    if (ub < 128) ub = (unsigned char)tolower(ub);
    if (ua == ub) return 0.0f;
    if (ua >= 128 || ub >= 128) return 1.0f; /* Require exact byte match on multi-byte UTF-8 */

    float d_sound = navi_sound_distance(mask_a, mask_b);
    float d_key   = KEY_COST_MATRIX[ua & 0x7F][ub & 0x7F];
    float best    = fminf(d_sound, d_key);
    if (best < 0.12f) best = 0.18f;
    if (is_first_phoneme && best > 0.05f) best *= 2.2f;
    return best;
}

void navi_trie_init(NaviTriePool *pool) {
    memset(pool, 0, sizeof(NaviTriePool));
    pool->node_count = 1;
    pool->nodes[0].first_child_idx  = -1;
    pool->nodes[0].next_sibling_idx = -1;
}

int32_t navi_trie_insert(NaviTriePool *pool, const uint32_t *masks,
                         const char *raw_chars, const uint8_t *tones,
                         uint8_t length, const char *ipa_utf8, uint16_t lang_bit) {
    int32_t curr = 0;
    for (uint8_t i = 0; i < length; i++) {
        unsigned char uc = (unsigned char)raw_chars[i];
        uint8_t ch_low = (uc < 128) ? (uint8_t)tolower(uc) : uc;
        int32_t child  = pool->nodes[curr].first_child_idx;
        int32_t prev   = -1;

        while (child != -1) {
            if (pool->nodes[child].key_ascii == ch_low) break;
            prev  = child;
            child = pool->nodes[child].next_sibling_idx;
        }

        if (child == -1) {
            if (pool->node_count >= MAX_TRIE_NODES) return -1;
            child = pool->node_count++;
            NaviTrieNode *n     = &pool->nodes[child];
            n->phoible_mask     = masks[i];
            n->key_ascii        = ch_low;
            n->tone             = tones[i];
            n->state_flag       = STATE_UNATTESTED;
            n->first_child_idx  = -1;
            n->next_sibling_idx = -1;

            if (prev == -1) pool->nodes[curr].first_child_idx = child;
            else            pool->nodes[prev].next_sibling_idx = child;
        }
        curr = child;
    }

    NaviTrieNode *leaf = &pool->nodes[curr];
    leaf->state_flag  = STATE_ATTESTED;
    leaf->lang_mask  |= lang_bit;
    if (ipa_utf8 && ipa_utf8[0] && (leaf->ipa_utf8[0] == '\0' || (lang_bit & ((1U << LANG_ENG) | (1U << LANG_CMN))))) {
        strncpy(leaf->ipa_utf8, ipa_utf8, 14);
        leaf->ipa_utf8[14] = '\0';
    }
    return curr;
}

static int32_t navi_trie_exact_node(const NaviTriePool *pool, const char *word, uint8_t len) {
    int32_t curr = 0;
    for (uint8_t i = 0; i < len; i++) {
        unsigned char uc = (unsigned char)word[i];
        uint8_t ch_low = (uc < 128) ? (uint8_t)tolower(uc) : uc;
        int32_t child  = pool->nodes[curr].first_child_idx;
        int32_t next   = -1;
        while (child != -1) {
            if (pool->nodes[child].key_ascii == ch_low) { next = child; break; }
            child = pool->nodes[child].next_sibling_idx;
        }
        if (next == -1) return -1;
        curr = next;
    }
    return (pool->nodes[curr].state_flag == STATE_ATTESTED) ? curr : -1;
}

PhonotacticState navi_evaluate_phonotactics(const uint32_t *masks, const char *raw_chars, uint8_t len) {
    if (len == 0 || len >= MAX_WORD_LEN) return STATE_CORRUPTED;
    uint8_t consecutive_consonants = 0;
    uint8_t identical_key_run      = 1;

    for (uint8_t i = 0; i < len; i++) {
        unsigned char uc = (unsigned char)raw_chars[i];
        if (uc < 32) return STATE_CORRUPTED;
        if (uc >= 128) continue; /* Valid UTF-8 sequence byte */
        if (i > 0 && raw_chars[i] == raw_chars[i - 1]) {
            if (++identical_key_run >= 4) return STATE_CORRUPTED;
        } else {
            identical_key_run = 1;
        }
        bool is_vowel = (masks[i] & (FEAT_SYLLABIC | FEAT_WILDCARD_V)) != 0;
        bool is_click = (masks[i] & FEAT_VELARIC_CLICK) != 0;
        if (is_vowel) consecutive_consonants = 0;
        else if (!is_click && ++consecutive_consonants >= 5) return STATE_ILLICIT;
    }
    return STATE_UNATTESTED;
}

static void trie_fuzzy_dfs(const NaviTriePool *pool, int32_t curr_node_idx,
                           const uint32_t *in_masks, const char *in_chars,
                           uint8_t in_len, uint8_t depth, float accumulated_cost,
                           float max_budget, char *path_buf, uint8_t path_len,
                           NaviLookupResult *result) {
    if (accumulated_cost > max_budget || accumulated_cost >= result->best_error_cost) return;

    const NaviTrieNode *node = &pool->nodes[curr_node_idx];
    if (depth == in_len && node->state_flag == STATE_ATTESTED) {
        float final_cost = accumulated_cost;
        if (node->lang_mask & (1U << GLOBAL_LANG_STATS.active_lang)) {
            final_cost -= 0.04f;
        } else if (!(node->lang_mask & ((1U << LANG_ENG) | (1U << LANG_ENK) | (1U << LANG_SPA) | (1U << LANG_CMN)))) {
            final_cost += 0.25f;
        }
        if (final_cost < 0.0f) final_cost = 0.0f;

        if (final_cost < result->best_error_cost && final_cost <= max_budget) {
            result->best_error_cost  = final_cost;
            result->matched_node_idx = curr_node_idx;
            result->resolved_state   = STATE_ATTESTED;
            path_buf[path_len]       = '\0';
            strncpy(result->matched_word, path_buf, MAX_WORD_LEN - 1);
        }
        return;
    }

    int32_t child_idx = node->first_child_idx;
    while (child_idx != -1) {
        const NaviTrieNode *child = &pool->nodes[child_idx];
        if (path_len + 1 < MAX_WORD_LEN) {
            path_buf[path_len] = (char)child->key_ascii;
            if (depth < in_len) {
                float step_cost = navi_dual_step_cost(in_masks[depth], in_chars[depth],
                                                      child->phoible_mask, (char)child->key_ascii, (depth == 0));
                trie_fuzzy_dfs(pool, child_idx, in_masks, in_chars, in_len,
                               depth + 1, accumulated_cost + step_cost, max_budget,
                               path_buf, path_len + 1, result);
            }
            trie_fuzzy_dfs(pool, child_idx, in_masks, in_chars, in_len,
                           depth, accumulated_cost + 0.35f, max_budget,
                           path_buf, path_len + 1, result);
        }
        child_idx = child->next_sibling_idx;
    }

    if (depth < in_len) {
        trie_fuzzy_dfs(pool, curr_node_idx, in_masks, in_chars, in_len,
                       depth + 1, accumulated_cost + 0.35f, max_budget,
                       path_buf, path_len, result);
    }
}

NaviLookupResult navi_resolve_word(const NaviTriePool *pool, const uint32_t *in_masks,
                                   const char *in_chars, uint8_t in_len) {
    char clean_exact[MAX_WORD_LEN] = {0};
    for (uint8_t i = 0; i < in_len; i++) {
        unsigned char uc = (unsigned char)in_chars[i];
        clean_exact[i] = (uc < 128) ? (char)tolower(uc) : (char)uc;
    }

    int32_t exact_idx = navi_trie_exact_node(pool, clean_exact, in_len);
    if (exact_idx != -1) {
        NaviLookupResult exact_res = { exact_idx, 0.0f, STATE_ATTESTED, {0} };
        strncpy(exact_res.matched_word, clean_exact, MAX_WORD_LEN - 1);
        return exact_res;
    }

    float max_budget = navi_get_max_error_budget(in_len);
    NaviLookupResult res = { -1, 999.0f, STATE_CORRUPTED, {0} };
    char path_buf[MAX_WORD_LEN] = {0};

    trie_fuzzy_dfs(pool, 0, in_masks, in_chars, in_len, 0, 0.0f, max_budget, path_buf, 0, &res);

    if (res.matched_node_idx == -1) {
        res.resolved_state = navi_evaluate_phonotactics(in_masks, in_chars, in_len);
        strncpy(res.matched_word, clean_exact, MAX_WORD_LEN - 1);
    }
    return res;
}

void navi_map3_record_transition(int32_t from_node, int32_t to_node, const char *to_word) {
    if (from_node <= 0 || to_node <= 0) return;
    for (uint32_t i = 0; i < GLOBAL_MAP3.edge_count; i++) {
        if (GLOBAL_MAP3.edges[i].from_node_idx == from_node &&
            GLOBAL_MAP3.edges[i].to_node_idx == to_node) {
            GLOBAL_MAP3.edges[i].weight++;
            return;
        }
    }
    if (GLOBAL_MAP3.edge_count < MAX_TRANSITIONS) {
        WordTransitionEdge *e = &GLOBAL_MAP3.edges[GLOBAL_MAP3.edge_count++];
        e->from_node_idx = from_node;
        e->to_node_idx   = to_node;
        e->weight        = 1;
        strncpy(e->to_word, to_word, 23);
        e->to_word[23]   = '\0';
    }
}

int32_t navi_map3_predict_next(int32_t from_node, char *out_word, size_t max_len) {
    uint32_t best_w = 0;
    int32_t  best_to = -1;
    for (uint32_t i = 0; i < GLOBAL_MAP3.edge_count; i++) {
        if (GLOBAL_MAP3.edges[i].from_node_idx == from_node &&
            GLOBAL_MAP3.edges[i].weight > best_w) {
            best_w  = GLOBAL_MAP3.edges[i].weight;
            best_to = GLOBAL_MAP3.edges[i].to_node_idx;
            strncpy(out_word, GLOBAL_MAP3.edges[i].to_word, max_len - 1);
            out_word[max_len - 1] = '\0';
        }
    }
    return best_to;
}

uint16_t subconscious_classify_dhatu(const char *word) {
    if (!strcmp(word, "hello") || !strcmp(word, "hi") || !strcmp(word, "hola") ||
        !strcmp(word, "salutations") || !strcmp(word, "greetings") || !strcmp(word, "bonjour") ||
        !strcmp(word, "nihao") || !strcmp(word, "ni") || !strcmp(word, "hao") ||
        !strcmp(word, "\xe4\xbd\xa0\xe5\xa5\xbd")) {
        return DHATU_GREET;
    }
    if (!strcmp(word, "how") || !strcmp(word, "como") || !strcmp(word, "comment") || !strcmp(word, "what")) {
        return DHATU_QUERY;
    }
    if (!strcmp(word, "are") || !strcmp(word, "estas") || !strcmp(word, "esta") || !strcmp(word, "allez")) {
        return DHATU_BEING;
    }
    if (!strcmp(word, "you") || !strcmp(word, "llamo") || !strcmp(word, "johnr") || !strcmp(word, "doe")) {
        return DHATU_IDENTITY;
    }
    if (!strcmp(word, "status") || !strcmp(word, "compile") || !strcmp(word, "memory")) {
        return DHATU_COMPUTE;
    }
    return DHATU_UNKNOWN;
}

const char *subconscious_dhatu_name(uint16_t dhatu) {
    switch (dhatu) {
        case DHATU_GREET:    return "coi/VAD (Greeting)";
        case DHATU_QUERY:    return "ma/PRACCH (Query)";
        case DHATU_BEING:    return "zasti/AS (State)";
        case DHATU_IDENTITY: return "cmene/NAMAN (Identity)";
        case DHATU_COMPUTE:  return "skami/GAN (System)";
        case DHATU_LEARNED:  return "cilre/JNA (Learned)";
        default:             return "bridi/DHATU (General)";
    }
}

CognitiveBridiTuple navi_subconscious_parse(uint32_t agent_x1, uint16_t action_selbri,
                                            uint32_t patient_x2, uint32_t dest_x3,
                                            uint32_t detected_lang_x4, uint8_t tense) {
    CognitiveBridiTuple b;
    memset(&b, 0, sizeof(b));
    b.dhatu_root_id    = action_selbri;
    b.selbri_relation  = action_selbri;
    b.sumti_x[0]       = agent_x1;
    b.sumti_x[1]       = patient_x2;
    b.sumti_x[2]       = dest_x3;
    b.sumti_x[3]       = detected_lang_x4;
    b.tense_aspect     = tense;
    b.certainty_weight = 1.0f;
    return b;
}

bool navi_medal_commit(MedalCore *medal, CognitiveBridiTuple fact,
                       uint32_t custom_node_idx, uint8_t priority_lock) {
    if (medal->count >= MAX_MEDAL_MEM) return false;
    MedalMemoryEntry *e     = &medal->entries[medal->count++];
    e->timestamp            = (uint64_t)medal->count;
    e->learned_fact         = fact;
    e->custom_trie_node_idx = custom_node_idx;
    e->priority_lock        = priority_lock;
    return true;
}

AcousticSynthFrame navi_features_to_formants(uint32_t mask, uint8_t tone) {
    AcousticSynthFrame f;
    memset(&f, 0, sizeof(f));
    switch (tone) {
        case TONE_HIGH:    f.f0_pitch_hz = 195.0f; break;
        case TONE_RISING:  f.f0_pitch_hz = 160.0f; break;
        case TONE_DIPPING: f.f0_pitch_hz = 115.0f; break;
        case TONE_FALLING: f.f0_pitch_hz = 175.0f; break;
        case TONE_LOW:     f.f0_pitch_hz = 105.0f; break;
        default:           f.f0_pitch_hz = 140.0f; break;
    }
    f.f1_hz = (mask & FEAT_HIGH) ? 300.0f : ((mask & FEAT_LOW) ? 750.0f : 500.0f);
    f.f2_hz = (mask & FEAT_BACK) ? 900.0f : 2100.0f;
    if (mask & FEAT_ROUND) f.f2_hz -= 250.0f;
    f.f3_hz = 2600.0f;
    f.bandwidth_hz[0] = 60.0f; f.bandwidth_hz[1] = 90.0f; f.bandwidth_hz[2] = 120.0f;
    if (mask & FEAT_VELARIC_CLICK) {
        f.click_impulse_amp = 1.0f;
        if (mask & FEAT_CLICK_DENTAL)   f.f2_hz = 3200.0f;
        if (mask & FEAT_CLICK_ALVEOLAR) f.f1_hz = 1100.0f;
        if (mask & FEAT_CLICK_LATERAL)  f.f2_hz = 1600.0f;
    }
    if ((mask & FEAT_CONTINUANT) && !(mask & FEAT_SYLLABIC)) {
        f.fricative_noise_amp = (mask & FEAT_STRIDENT) ? 0.45f : 0.20f;
    }
    return f;
}

void navi_render_oscilloscope(const AcousticSynthFrame *frame, uint8_t tone, OscilloscopeBuffer *out_buf) {
    out_buf->sample_count = 512;
    uint32_t rng_state = 0x9E3779B9U;
    for (uint16_t n = 0; n < 512; n++) {
        float t = (float)n / SAMPLE_RATE_HZ;
        float progress = (float)n / 512.0f;
        float f0 = frame->f0_pitch_hz;
        if (tone == TONE_RISING)  f0 *= (1.0f + 0.30f * progress);
        if (tone == TONE_FALLING) f0 *= (1.20f - 0.45f * progress);
        float env1 = expf(-M_PI * frame->bandwidth_hz[0] * t);
        float env2 = expf(-M_PI * frame->bandwidth_hz[1] * t);
        float env3 = expf(-M_PI * frame->bandwidth_hz[2] * t);
        float w1 = env1 * sinf(2.0f * M_PI * frame->f1_hz * t);
        float w2 = 0.6f * env2 * sinf(2.0f * M_PI * frame->f2_hz * t + (2.0f * M_PI * f0 * t));
        float w3 = 0.3f * env3 * sinf(2.0f * M_PI * frame->f3_hz * t);
        float click_burst = 0.0f;
        if (frame->click_impulse_amp > 0.0f && n < 64) {
            float decay = expf(-(float)n * 0.12f);
            click_burst = frame->click_impulse_amp * decay * sinf(2.0f * M_PI * frame->f2_hz * t);
        }
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
        float noise = ((float)(rng_state & 0xFFFF) / 32768.0f - 1.0f) * frame->fricative_noise_amp;
        out_buf->pcm_samples[n] = (w1 + w2 + w3) * 0.5f + click_burst + noise;
        out_buf->scope_x[n] = w1 + (click_burst * 0.7f);
        out_buf->scope_y[n] = w2 + w3 + (noise * 0.4f);
    }
}

int subconscious_detect_sentence_lang(const NaviTriePool *pool, const char *sentence, int fallback_lang) {
    char buf[256];
    strncpy(buf, sentence, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    uint32_t scores[LANG_COUNT] = {0};
    char *tok = strtok(buf, " \t\r\n");

    while (tok != NULL) {
        char word[64] = {0};
        size_t j = 0;
        for (size_t i = 0; tok[i] && j + 1 < sizeof(word); i++) {
            unsigned char c = (unsigned char)tok[i];
            if (c == '"' || c == '\\' || c == '?' || c == '!' || c == '.' || c == ',') continue;
            word[j++] = (c < 128) ? (char)tolower(c) : (char)c;
        }
        word[j] = '\0';

        uint8_t len = (uint8_t)strlen(word);
        if (len > 0) {
            int32_t idx = navi_trie_exact_node(pool, word, len);
            if (idx > 0) {
                uint16_t mask = pool->nodes[idx].lang_mask;
                int lang_bits = navi_popcount32((uint32_t)mask);
                uint32_t weight = (lang_bits == 1) ? 4 : 1;
                for (int l = 0; l < LANG_COUNT; l++) {
                    if (mask & (1U << l)) scores[l] += weight;
                }
            }
        }
        tok = strtok(NULL, " \t\r\n");
    }

    int best_lang = fallback_lang;
    uint32_t best_score = 0;
    for (int l = 0; l < LANG_COUNT; l++) {
        if (scores[l] > best_score) {
            best_score = scores[l];
            best_lang  = l;
        }
    }
    return best_lang;
}

static int32_t ingest_clean_token(NaviTriePool *pool, const char *raw_word,
                                  const char *ipa_label, uint8_t default_tone, uint16_t lang_bit) {
    char clean_chars[64];
    uint32_t masks[64];
    uint8_t  tones[64];
    uint8_t  len = 0;
    uint8_t  detected_tone = default_tone;

    for (size_t i = 0; raw_word[i] != '\0' && len < 60; i++) {
        unsigned char c = (unsigned char)raw_word[i];
        if (c == '/' || c == '[' || c == ']' || c == '\'' || c == '"') continue;
        if (c >= '1' && c <= '5') {
            detected_tone = (uint8_t)(c - '0');
            if (len > 0) tones[len - 1] = detected_tone;
            continue;
        }
        if (isalnum(c) || c == '*' || c == '!' || c == '|' || c >= 128) {
            uint8_t t = 0;
            clean_chars[len] = (c < 128) ? (char)tolower(c) : (char)c;
            masks[len]       = char_to_phoible_mask(clean_chars[len], &t);
            tones[len]       = (len == 0) ? detected_tone : 0;
            len++;
        }
    }
    if (len > 0) {
        clean_chars[len] = '\0';
        if (ipa_label && ipa_label[0]) {
            uint8_t ipa_tone = TONE_NEUTRAL;
            masks[len - 1] |= ipa_string_to_composite_mask(ipa_label, &ipa_tone);
            if (ipa_tone != TONE_NEUTRAL) tones[len - 1] = ipa_tone;
        }
        const char *ipa = (ipa_label && ipa_label[0]) ? ipa_label : clean_chars;
        return navi_trie_insert(pool, masks, clean_chars, tones, len, ipa, lang_bit);
    }
    return -1;
}

static uint32_t navi_load_language_file(NaviTriePool *pool, const char *filepath,
                                        int lang_id, uint8_t default_tone, uint32_t target_entries) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) return 0;

    char line[512];
    uint32_t total_lines = 0;
    while (fgets(line, sizeof(line), fp)) total_lines++;
    rewind(fp);

    uint32_t stride = (total_lines > target_entries && target_entries > 0) ? (total_lines / target_entries) : 1;
    uint32_t loaded = 0, line_idx = 0;
    uint16_t lang_bit = (uint16_t)(1U << lang_id);

    while (fgets(line, sizeof(line), fp)) {
        line_idx++;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char col1[64] = {0}, col2[64] = {0};
        if (sscanf(line, "%63s %63s", col1, col2) >= 1) {
            size_t wlen = strlen(col1);
            if (wlen > 5 && (line_idx % stride != 0)) continue;

            const char *ipa_str = (col2[0] != '\0') ? col2 : col1;
            ingest_clean_token(pool, col1, ipa_str, default_tone, lang_bit);
            loaded++;
            if (pool->node_count >= MAX_TRIE_NODES - 512) break;
        }
    }
    fclose(fp);
    printf("[+] MAP 2 [%-3s] | Indexed %6u words from %-24s | Nodes: %d\n",
           LANG_NAMES[lang_id], loaded, filepath, pool->node_count);
    return loaded;
}

static void navi_compile_seed_sentences(NaviTriePool *pool) {
    const struct { const char *sentence; int lang_id; } corpora[] = {
        {"hello how are you doing today",                  LANG_ENG},
        {"greetings sir network navigator online",   LANG_ENG},
        {"salutations sir network navigator online", LANG_ENG},
        {"how are you operating at peak efficiency",       LANG_ENG},
        {"compile phonetic substrate into memory",         LANG_ENG},
        {"hola como estas mi amigo",                       LANG_SPA},
        {"hola me llamo john",                           LANG_SPA},
        {"buenos dias como te va",                         LANG_SPA},
        {"bonjour comment allez vous",                     LANG_FRA},
        {"ni hao wo shi wangluo linghangyuan",             LANG_CMN},
        {"qala ukuxoxa ngolimi",                           LANG_ZUL}
    };

    for (size_t s = 0; s < sizeof(corpora) / sizeof(corpora[0]); s++) {
        char buf[256];
        strncpy(buf, corpora[s].sentence, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';

        char *tok = strtok(buf, " ");
        int32_t prev_node = -1;
        uint16_t lang_bit = (uint16_t)(1U << corpora[s].lang_id);

        while (tok != NULL) {
            char ipa_tag[16];
            snprintf(ipa_tag, sizeof(ipa_tag), "/%.12s/", tok);
            int32_t curr_node = ingest_clean_token(pool, tok, ipa_tag, TONE_NEUTRAL, lang_bit);
            if (prev_node > 0 && curr_node > 0) {
                navi_map3_record_transition(prev_node, curr_node, tok);
            }
            prev_node = curr_node;
            tok = strtok(NULL, " ");
        }
    }

    /* Seed Mandarin Chinese Pinyin <-> Hanzi Glyphs so "ni hao" and "你好" both work natively! */
    int32_t n_ni  = ingest_clean_token(pool, "ni",    "你(nǐ)",     TONE_DIPPING, 1U << LANG_CMN);
    int32_t n_hao = ingest_clean_token(pool, "hao",   "好(hǎo)",    TONE_DIPPING, 1U << LANG_CMN);
    ingest_clean_token(pool, "nihao", "你好(nǐhǎo)", TONE_DIPPING, 1U << LANG_CMN);
    ingest_clean_token(pool, "\xe4\xbd\xa0\xe5\xa5\xbd", "你好(nǐhǎo)", TONE_DIPPING, 1U << LANG_CMN);
    if (n_ni > 0 && n_hao > 0) {
        navi_map3_record_transition(n_ni, n_hao, "hao");
    }

    printf("[+] MAP 3 (SENTENCES) | Compiled %u word-to-sentence transition links\n", GLOBAL_MAP3.edge_count);
}

bool navi_save_substrate_state(const char *filepath) {
    FILE *fp = fopen(filepath, "wb");
    if (!fp) return false;
    uint32_t magic = SUBSTRATE_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(ASCII_PHOIBLE_MAP, sizeof(uint32_t), 256, fp);
    fwrite(&GLOBAL_TRIE.node_count, sizeof(int32_t), 1, fp);
    fwrite(GLOBAL_TRIE.nodes, sizeof(NaviTrieNode), (size_t)GLOBAL_TRIE.node_count, fp);
    fwrite(&GLOBAL_MAP3.edge_count, sizeof(uint32_t), 1, fp);
    fwrite(GLOBAL_MAP3.edges, sizeof(WordTransitionEdge), GLOBAL_MAP3.edge_count, fp);
    fclose(fp);
    printf("[+] BASE STATE SAVED  | %s (%d Trie nodes, %u Sentence edges)\n",
           filepath, GLOBAL_TRIE.node_count, GLOBAL_MAP3.edge_count);
    return true;
}

bool navi_load_substrate_state(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return false;
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != SUBSTRATE_MAGIC) {
        fclose(fp); return false;
    }
    if (fread(ASCII_PHOIBLE_MAP, sizeof(uint32_t), 256, fp) != 256) { fclose(fp); return false; }
    if (fread(&GLOBAL_TRIE.node_count, sizeof(int32_t), 1, fp) != 1) { fclose(fp); return false; }
    if (fread(GLOBAL_TRIE.nodes, sizeof(NaviTrieNode), (size_t)GLOBAL_TRIE.node_count, fp) != (size_t)GLOBAL_TRIE.node_count) {
        fclose(fp); return false;
    }
    if (fread(&GLOBAL_MAP3.edge_count, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
    if (fread(GLOBAL_MAP3.edges, sizeof(WordTransitionEdge), GLOBAL_MAP3.edge_count, fp) != GLOBAL_MAP3.edge_count) {
        fclose(fp); return false;
    }
    fclose(fp);
    printf("[+] FAST-BOOT COMPLETE| Loaded compiled base state from %s (%d nodes, %u edges)\n",
           filepath, GLOBAL_TRIE.node_count, GLOBAL_MAP3.edge_count);
    return true;
}

void navi_boot_all_datasets(NaviTriePool *pool) {
    navi_compile_phoible_csv("data/phoible.csv");
    navi_load_language_file(pool, "data/en_US.txt",   LANG_ENG, 0, 55000);
    navi_load_language_file(pool, "data/en_UK.txt",   LANG_ENK, 0, 45000);
    navi_load_language_file(pool, "data/es_MX.txt",   LANG_SPA, 0, 45000);
    navi_load_language_file(pool, "data/es_ES.txt",   LANG_SPA, 0, 25000);
    navi_load_language_file(pool, "data/fr_FR.txt",   LANG_FRA, 0, 35000);
    navi_load_language_file(pool, "data/de.txt",      LANG_DEU, 0, 30000);
    navi_load_language_file(pool, "data/SAD-IV.NCHLT_Lemma-Morph-POS.Final_TRAIN.2026-03-31.zu.txt", LANG_ZUL, 1, 25000);
    navi_load_language_file(pool, "data/SAD-IV.NCHLT_Lemma-Morph-POS.Final_TRAIN.2026-03-31.xh.txt", LANG_XHO, 1, 25000);
    navi_load_language_file(pool, "data/zh_hans.txt", LANG_CMN, 1, 25000);
    navi_load_language_file(pool, "data/yue.txt",     LANG_YUE, 1, 15000);
    navi_load_language_file(pool, "data/ja.txt",      LANG_JPN, 0, 20000);
    navi_load_language_file(pool, "data/ko.txt",      LANG_KOR, 0, 15000);
    navi_load_language_file(pool, "data/ar.txt",      LANG_ARA, 0, 15000);
    navi_load_language_file(pool, "data/fa.txt",      LANG_FAS, 0, 15000);
    navi_load_language_file(pool, "data/sw.txt",      LANG_SWA, 0, 15000);
    navi_load_language_file(pool, "data/eo.txt",      LANG_EPO, 0, 15000);
    navi_compile_seed_sentences(pool);
}

void seed_initial_substrate(void) {
    navi_init_key_matrix();
    navi_trie_init(&GLOBAL_TRIE);
    memset(&GLOBAL_MEDAL, 0, sizeof(MedalCore));

    if (navi_load_substrate_state("data/navi_base.bin")) {
        /* Always ensure conversational seed sentences & Mandarin glyphs are present */
        if (GLOBAL_MAP3.edge_count == 0) {
            navi_compile_seed_sentences(&GLOBAL_TRIE);
            navi_save_substrate_state("data/navi_base.bin");
        }
        return;
    }

    printf("[*] Compiling 3-Tier Substrate Pipeline (PHOIBLE -> Words -> Sentences)...\n");
    navi_boot_all_datasets(&GLOBAL_TRIE);
    navi_save_substrate_state("data/navi_base.bin");
}

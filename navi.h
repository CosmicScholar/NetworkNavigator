#ifndef NAVI_H
#define NAVI_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define closesocket close
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_TRIE_NODES     2097152  /* 64 MB Cache-Aligned Pool (32B per node) */
#define MAX_WORD_LEN       64
#define MAX_MEDAL_MEM      4096
#define MAX_TRANSITIONS    65536    /* Map 3 Word->Sentence Bigram Transitions */
#define SAMPLE_RATE_HZ     16000.0f
#define SUBSTRATE_MAGIC    0x4E415634U /* "NAV4" Compiled Base State Header */

/* Substrate Feature Masks */
#define FEAT_SYLLABIC       (1U << 0)
#define FEAT_CONSONANTAL    (1U << 1)
#define FEAT_SONORANT       (1U << 2)
#define FEAT_VOICE          (1U << 3)
#define FEAT_CONTINUANT     (1U << 4)
#define FEAT_NASAL          (1U << 5)
#define FEAT_STRIDENT       (1U << 6)
#define FEAT_LATERAL        (1U << 7)
#define FEAT_LABIAL         (1U << 8)
#define FEAT_CORONAL        (1U << 9)
#define FEAT_ANTERIOR       (1U << 10)
#define FEAT_DORSAL         (1U << 11)
#define FEAT_HIGH           (1U << 12)
#define FEAT_LOW            (1U << 13)
#define FEAT_BACK           (1U << 14)
#define FEAT_ROUND          (1U << 15)
#define FEAT_VELARIC_CLICK  (1U << 16)
#define FEAT_CLICK_BILABIAL (1U << 17)
#define FEAT_CLICK_DENTAL   (1U << 18)
#define FEAT_CLICK_ALVEOLAR (1U << 19)
#define FEAT_CLICK_PALATAL  (1U << 20)
#define FEAT_CLICK_LATERAL  (1U << 21)
#define FEAT_GLOTTALIC      (1U << 22)
#define FEAT_WILDCARD_V     (1U << 30)
#define FEAT_WILDCARD_C     (1U << 31)

enum {
    LANG_ENG = 0, LANG_ENK = 1, LANG_SPA = 2, LANG_FRA = 3,
    LANG_DEU = 4, LANG_ZUL = 5, LANG_XHO = 6, LANG_CMN = 7,
    LANG_YUE = 8, LANG_JPN = 9, LANG_KOR = 10, LANG_ARA = 11,
    LANG_FAS = 12, LANG_SWA = 13, LANG_EPO = 14, LANG_OTH = 15,
    LANG_COUNT = 16
};

extern const char *LANG_NAMES[LANG_COUNT];

enum {
    DHATU_UNKNOWN  = 0,
    DHATU_GREET    = 101,
    DHATU_QUERY    = 102,
    DHATU_BEING    = 103,
    DHATU_IDENTITY = 104,
    DHATU_COMPUTE  = 105,
    DHATU_LEARNED  = 900
};

typedef enum {
    STATE_ATTESTED   = 0,
    STATE_UNATTESTED = 1,
    STATE_ILLICIT    = 2,
    STATE_CORRUPTED  = 3
} PhonotacticState;

typedef enum {
    TONE_NEUTRAL = 0, TONE_HIGH = 1, TONE_RISING = 2,
    TONE_DIPPING = 3, TONE_FALLING = 4, TONE_LOW = 5
} ToneRegister;

typedef struct {
    uint32_t phoible_mask;
    int32_t  first_child_idx;
    int32_t  next_sibling_idx;
    uint16_t lang_mask;
    uint8_t  key_ascii;
    uint8_t  tone;
    uint8_t  state_flag;
    char     ipa_utf8[15];
} NaviTrieNode;

typedef struct {
    NaviTrieNode nodes[MAX_TRIE_NODES];
    int32_t      node_count;
} NaviTriePool;

typedef struct {
    int32_t          matched_node_idx;
    float            best_error_cost;
    PhonotacticState resolved_state;
    char             matched_word[MAX_WORD_LEN];
} NaviLookupResult;

typedef struct {
    uint32_t hits[LANG_COUNT];
    int      active_lang;
} LanguageTelemetryTracker;

typedef struct {
    int32_t  from_node_idx;
    int32_t  to_node_idx;
    uint32_t weight;
    char     to_word[24];
} WordTransitionEdge;

typedef struct {
    WordTransitionEdge edges[MAX_TRANSITIONS];
    uint32_t           edge_count;
} SentenceTransitionMap;

typedef struct {
    float f0_pitch_hz, f1_hz, f2_hz, f3_hz;
    float bandwidth_hz[3];
    float click_impulse_amp, fricative_noise_amp;
} AcousticSynthFrame;

typedef struct {
    float pcm_samples[512], scope_x[512], scope_y[512];
    uint16_t sample_count;
} OscilloscopeBuffer;

typedef struct {
    uint32_t dhatu_root_id;
    uint16_t selbri_relation;
    uint32_t sumti_x[5];
    uint8_t  tense_aspect;
    float    certainty_weight;
} CognitiveBridiTuple;

typedef struct {
    uint64_t timestamp;
    CognitiveBridiTuple learned_fact;
    uint32_t custom_trie_node_idx;
    uint8_t  priority_lock;
} MedalMemoryEntry;

typedef struct {
    MedalMemoryEntry entries[MAX_MEDAL_MEM];
    uint32_t count;
} MedalCore;

extern NaviTriePool     GLOBAL_TRIE;
extern MedalCore        GLOBAL_MEDAL;
extern SentenceTransitionMap GLOBAL_MAP3;
extern LanguageTelemetryTracker GLOBAL_LANG_STATS;

/* Function Prototypes */
int navi_popcount32(uint32_t x);
uint32_t char_to_phoible_mask(char c, uint8_t *out_tone);
uint32_t ipa_string_to_composite_mask(const char *ipa_str, uint8_t *out_tone);
void navi_init_key_matrix(void);
void navi_trie_init(NaviTriePool *pool);
int32_t navi_trie_insert(NaviTriePool *pool, const uint32_t *masks, const char *raw_chars, const uint8_t *tones, uint8_t length, const char *ipa_utf8, uint16_t lang_bit);
NaviLookupResult navi_resolve_word(const NaviTriePool *pool, const uint32_t *in_masks, const char *in_chars, uint8_t in_len);
void navi_map3_record_transition(int32_t from_node, int32_t to_node, const char *to_word);
int32_t navi_map3_predict_next(int32_t from_node, char *out_word, size_t max_len);
uint16_t subconscious_classify_dhatu(const char *word);
const char *subconscious_dhatu_name(uint16_t dhatu);
CognitiveBridiTuple navi_subconscious_parse(uint32_t agent_x1, uint16_t action_selbri, uint32_t patient_x2, uint32_t dest_x3, uint32_t detected_lang_x4, uint8_t tense);
bool navi_medal_commit(MedalCore *medal, CognitiveBridiTuple fact, uint32_t custom_node_idx, uint8_t priority_lock);
AcousticSynthFrame navi_features_to_formants(uint32_t mask, uint8_t tone);
void navi_render_oscilloscope(const AcousticSynthFrame *frame, uint8_t tone, OscilloscopeBuffer *out_buf);
void navi_compile_phoible_csv(const char *csv_path);
void navi_boot_all_datasets(NaviTriePool *pool);
bool navi_save_substrate_state(const char *filepath);
bool navi_load_substrate_state(const char *filepath);
void seed_initial_substrate(void);
int subconscious_detect_sentence_lang(const NaviTriePool *pool, const char *sentence, int fallback_lang);
void handle_chat_query(const char *raw_msg, char *out_http_buf, size_t buf_size);

#endif
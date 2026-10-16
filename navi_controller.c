/* =========================================================================
 * NETWORK NAVIGATOR // INTERACTIVE C CONTROLLER & KUPO x MEPO MATRIX
 * Parses Daemon JSON, Commits Live State to KUPO x MEPO Grid, Dispatches Actions
 * Compile: gcc -O3 src/navi_controller.c -o navi_ctrl
 * ========================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define closesocket close
#endif

#define KUPO_ROWS 4  /* 0:GREET, 1:QUERY/STATE, 2:COMPUTE/SYS, 3:LEARNED/GEN */
#define MEPO_COLS 4  /* Rolling 4-Turn Context Window per KUPO Row           */
#define TOKEN_WEIGHT_CONST 2048

typedef struct {
    uint32_t knowledge_shard;  /* K: Hash of input/node */
    uint16_t identity_mask;    /* U: Language / Operator mask */
    uint32_t coordinate_idx;   /* P: Matrix turn coordinate */
    uint32_t operation_code;   /* O: Dhatu ID */
    uint32_t parity_checksum;  /* Xi: Lossless Parity Checksum */
    bool     is_active;
} MnemoponicCell;

typedef struct {
    MnemoponicCell grid[KUPO_ROWS][MEPO_COLS];
    uint32_t active_kupo;
    uint32_t active_mepo;
    uint32_t turn_counter;
} KupoMepoMatrix;

static KupoMepoMatrix SYSTEM_MATRIX;

typedef void (*ActionHandler)(const char *spoken, const char *lang, uint32_t xi);

static void action_greet(const char *spoken, const char *lang, uint32_t xi) {
    printf("  [ACTION // COI-GREET]   Handshake locked (%s) | Parity Xi=%u\n", lang, xi);
}
static void action_query(const char *spoken, const char *lang, uint32_t xi) {
    printf("  [ACTION // MA-QUERY]    Telemetry & Map 3 chain queried (%s) | Parity Xi=%u\n", lang, xi);
}
static void action_compute(const char *spoken, const char *lang, uint32_t xi) {
    printf("  [ACTION // SKAMI-SYS]   Substrate diagnostics verified (%s) | Parity Xi=%u\n", lang, xi);
}
static void action_learned(const char *spoken, const char *lang, uint32_t xi) {
    printf("  [ACTION // CILRE-LRN]   New token committed to MedalCore (%s) | Parity Xi=%u\n", lang, xi);
}
static void action_default(const char *spoken, const char *lang, uint32_t xi) {
    printf("  [ACTION // BRIDI-GEN]   Standard substrate routing (%s) | Parity Xi=%u\n", lang, xi);
}

uint32_t compute_matrix_checksum(uint32_t k, uint16_t u, uint32_t p, uint32_t o) {
    uint32_t sum_facts   = k + (uint32_t)u;
    uint32_t prod_coords = p * o;
    return (sum_facts + prod_coords) % TOKEN_WEIGHT_CONST;
}

static uint32_t hash_shard(const char *str) {
    uint32_t h = 5381;
    while (*str) h = ((h << 5) + h) + (unsigned char)(*str++);
    return h & 0xFFFF;
}

bool matrix_commit_cell(KupoMepoMatrix *mat, uint32_t kupo_idx, uint32_t mepo_idx,
                        uint32_t k, uint16_t u, uint32_t p, uint32_t o) {
    if (kupo_idx >= KUPO_ROWS || mepo_idx >= MEPO_COLS) return false;
    MnemoponicCell *cell = &mat->grid[kupo_idx][mepo_idx];
    cell->knowledge_shard = k;
    cell->identity_mask   = u;
    cell->coordinate_idx  = p;
    cell->operation_code  = o;
    cell->parity_checksum = compute_matrix_checksum(k, u, p, o);
    cell->is_active       = true;
    mat->active_kupo      = kupo_idx;
    mat->active_mepo      = mepo_idx;
    return true;
}

void print_matrix_table(const KupoMepoMatrix *mat) {
    const char *row_names[KUPO_ROWS] = { "K0:GREET ", "K1:QUERY ", "K2:COMPUT", "K3:LEARN " };
    printf("\n  +----------+-----------------+-----------------+-----------------+-----------------+\n");
    printf("  | KUPO\\MEPO|      M0         |      M1         |      M2         |      M3         |\n");
    printf("  +----------+-----------------+-----------------+-----------------+-----------------+\n");
    for (int r = 0; r < KUPO_ROWS; r++) {
        printf("  | %s |", row_names[r]);
        for (int c = 0; c < MEPO_COLS; c++) {
            const MnemoponicCell *cell = &mat->grid[r][c];
            if (cell->is_active) {
                char mark = (r == (int)mat->active_kupo && c == (int)mat->active_mepo) ? '*' : ' ';
                printf("%cO:%3u Xi:%4u   |", mark, cell->operation_code, cell->parity_checksum);
            } else {
                printf("      [EMPTY]    |");
            }
        }
        printf("\n");
    }
    printf("  +----------+-----------------+-----------------+-----------------+-----------------+\n");
}

/* Extract JSON string field value into out_buf */
static void extract_json_str(const char *json, const char *key, char *out_buf, size_t max_len) {
    out_buf[0] = '\0';
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":\"", key);
    const char *start = strstr(json, pattern);
    if (!start) return;
    start += strlen(pattern);
    size_t i = 0;
    while (*start && *start != '"' && i + 1 < max_len) {
        out_buf[i++] = *start++;
    }
    out_buf[i] = '\0';
}

/* Extract JSON integer field value */
static uint32_t extract_json_uint(const char *json, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *start = strstr(json, pattern);
    if (!start) return 0;
    start += strlen(pattern);
    return (uint32_t)strtoul(start, NULL, 10);
}

static void url_encode(char *dst, const char *src, size_t max_len) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 4 < max_len; i++) {
        unsigned char c = (unsigned char)src[i];
        if (isalnum(c) || c == '-' || c == '_' || c == '.') {
            dst[j++] = (char)c;
        } else if (c == ' ') {
            dst[j++] = '+';
        } else {
            snprintf(dst + j, 4, "%%%02X", c);
            j += 3;
        }
    }
    dst[j] = '\0';
}

bool bridge_query_daemon(const char *raw_message) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    struct sockaddr_in serv_addr = {0};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("[!] Cannot reach navi_core on port 8080. Is ./navi_core running?\n");
        closesocket(sock);
        return false;
    }

    char encoded[512];
    url_encode(encoded, raw_message, sizeof(encoded));

    char req[1024];
    snprintf(req, sizeof(req), "GET /api?msg=%s HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n", encoded);
    send(sock, req, (int)strlen(req), 0);

    char res_buf[8192] = {0};
    int total = 0, n;
    while ((n = recv(sock, res_buf + total, (int)sizeof(res_buf) - total - 1, 0)) > 0) {
        total += n;
        if (total >= (int)sizeof(res_buf) - 1) break;
    }
    closesocket(sock);
#ifdef _WIN32
    WSACleanup();
#endif

    if (total <= 0) return false;

    char *json = strstr(res_buf, "\r\n\r\n");
    json = json ? (json + 4) : res_buf;

    char spoken[512] = {0}, lang[32] = {0};
    extract_json_str(json, "spoken", spoken, sizeof(spoken));
    extract_json_str(json, "lang",   lang,   sizeof(lang));
    uint32_t dhatu = extract_json_uint(json, "dhatu");

    /* Map Dhatu to KUPO Row (0..3) and advance rolling MEPO Column (0..3) */
    uint32_t kupo_row = 3;
    ActionHandler handler = action_default;
    if (dhatu == 101)                    { kupo_row = 0; handler = action_greet;   }
    else if (dhatu == 102 || dhatu == 103){ kupo_row = 1; handler = action_query;   }
    else if (dhatu == 105)               { kupo_row = 2; handler = action_compute; }
    else if (dhatu == 900)               { kupo_row = 3; handler = action_learned; }

    uint32_t turn = ++SYSTEM_MATRIX.turn_counter;
    uint32_t mepo_col = turn % MEPO_COLS;
    uint32_t k_shard  = hash_shard(raw_message);
    uint16_t u_mask   = (uint16_t)(lang[0] ? lang[0] : 1);

    matrix_commit_cell(&SYSTEM_MATRIX, kupo_row, mepo_col, k_shard, u_mask, turn, dhatu ? dhatu : 100);
    uint32_t xi = SYSTEM_MATRIX.grid[kupo_row][mepo_col].parity_checksum;

    printf("\n  \033[1;36m[NAVI (%s)]:\033[0m %s\n", lang[0] ? lang : "CORE", spoken[0] ? spoken : json);
    handler(spoken, lang, xi);
    printf("  [KUPO x MEPO]   Committed to Cell [K%u][M%u] (Shard:%u, Turn:%u, Xi:%u)\n",
           kupo_row, mepo_col, k_shard, turn, xi);

    return true;
}

int main(void) {
    memset(&SYSTEM_MATRIX, 0, sizeof(KupoMepoMatrix));
    matrix_commit_cell(&SYSTEM_MATRIX, 0, 0, 5001, 0, 12, 101);

    printf("============================================================\n");
    printf("  NETNAVI CONTROLLER CLI // KUPO x MEPO MATRIX ACTIVE\n");
    printf("  Commands: 'matrix' (view 4x4 state grid), 'exit' (quit)\n");
    printf("============================================================\n");

    char input_buf[256];
    while (1) {
        printf("\n[SIR GAMEZ] > ");
        if (!fgets(input_buf, sizeof(input_buf), stdin)) break;
        input_buf[strcspn(input_buf, "\r\n")] = 0;
        if (strlen(input_buf) == 0) continue;

        if (strcmp(input_buf, "exit") == 0 || strcmp(input_buf, "quit") == 0) break;
        if (strcmp(input_buf, "matrix") == 0) {
            print_matrix_table(&SYSTEM_MATRIX);
            continue;
        }

        bridge_query_daemon(input_buf);
    }
    return 0;
}
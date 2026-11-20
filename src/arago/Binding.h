#pragma once

#ifdef __cplusplus
extern "C" {
#endif

unsigned carlota_abi_version(void);
/* UTF-8 JSON, owned by the calling thread until its next invocation. */
const char* carlota_layer(const char* config, const char* algorithm,
                          unsigned batch, unsigned queries, unsigned context);

#ifdef __cplusplus
}
#endif

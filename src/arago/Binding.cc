#include "Binding.h"
#include "Region.h"
#include <stdexcept>

extern "C" unsigned carlota_abi_version() { return 1; }

extern "C" const char* carlota_layer(const char* config, const char* algorithm,
                                     unsigned batch, unsigned queries, unsigned context) {
  thread_local std::string response;
  try {
    if (!config || !algorithm) throw std::invalid_argument("missing device config or algorithm");
    arago::Region model(nlohmann::json::parse(config), algorithm);
    response = model.run({batch, queries, context}).dump();
  } catch (const std::exception& e) {
    response = nlohmann::json{{"error", e.what()}}.dump();
  }
  return response.c_str();
}

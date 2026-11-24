#pragma once

#include "DecoderLayer.h"
#include <nlohmann/json.hpp>
#include <string>

namespace arago {
// A configured decoder model can be reused for successive request batches.
class Region {
public:
  Region(const nlohmann::json& config, const std::string& algorithm);
  nlohmann::json run(LayerShape shape) const;
private:
  nlohmann::json config_;
  std::string algorithm_;
  HardwareConfig hardware_;
  AttentionSchedule schedule_;
  uint64_t dram_capacity_;
};
}

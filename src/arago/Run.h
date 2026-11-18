#pragma once

#include <iosfwd>
#include <string>

namespace arago {
void run_trace(const std::string& model_path, const std::string& hardware_path,
               const std::string& trace_path, const std::string& attention,
               unsigned batch_limit, std::ostream& result);
}

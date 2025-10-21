///==========================================================================
/// File: ArtsAbstractMachine.cpp
///
/// Implementation of ARTS Abstract Machine with comprehensive configuration
/// parsing and abstractMachine information capabilities.
///==========================================================================

/// Debug
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(abstract_machine);

#include "arts/Utils/ArtsAbstractMachine.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace mlir {
namespace arts {

ArtsAbstractMachine::ArtsAbstractMachine(const std::string &configFile) {
  ARTS_DEBUG_HEADER(AbstractMachine);

  /// Helper function to get default config path
  auto getDefaultConfigPath = [&]() -> std::string {
    std::error_code ec;
    auto cwd = std::filesystem::current_path(ec);
    if (ec) {
      ARTS_ERROR("Failed to get current working directory: " << ec.message());
      configFileExists = false;
      return "";
    }
    return (cwd / "arts.cfg").string();
  };

  /// Check if config file is provided - if not, use default config path
  std::string path;
  if (!configFile.empty())
    path = configFile;
  else 
    path = getDefaultConfigPath();

  ARTS_DEBUG("Looking for configuration file at: " << path);
  if (!parseFromFile(path)) {
    ARTS_ERROR("No arts.cfg file found at " << path);
    configFileExists = false;
    return;
  }

  /// Validate the configuration
  if (!validateConfiguration())
    ARTS_WARN("Configuration validation failed - using defaults");
  else
    ARTS_INFO("Configuration loaded successfully");
  ARTS_DEBUG_FOOTER(AbstractMachine);
}

bool ArtsAbstractMachine::parseFromFile(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    ARTS_DEBUG("Could not open configuration file: " << path);
    return false;
  }

  ARTS_DEBUG("Successfully opened configuration file: " << path);

  configFileExists = true; 
  std::string line;
  bool inArtsSection = false;
  int lineNumber = 0;

  while (std::getline(in, line)) {
    lineNumber++;
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      if (!line.empty())
        ARTS_DEBUG("Line " << lineNumber << " (comment): " << line);
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      inArtsSection = (line == "[ARTS]");
      ARTS_DEBUG("Line " << lineNumber << " (section): " << line
                         << " - inArtsSection="
                         << (inArtsSection ? "true" : "false"));
      continue;
    }
    if (!inArtsSection) {
      ARTS_DEBUG("Line " << lineNumber << " (skipped): " << line);
      continue;
    }

    auto pos = line.find('=');
    if (pos == std::string::npos) {
      ARTS_DEBUG("Line " << lineNumber << " (no equals): " << line);
      continue;
    }
    std::string key = trim(line.substr(0, pos));
    std::string val = trim(line.substr(pos + 1));
    ARTS_DEBUG("Line " << lineNumber << " (config): " << key << "=" << val);

    /// Core Configuration
    if (key == "threads") {
      threads = parseInt(val);
      ARTS_DEBUG("Set threads=" << threads);
    } else if (key == "tMT") {
      tMT = parseInt(val, 0);
      ARTS_DEBUG("Set tMT=" << tMT);
    } else if (key == "nodeCount") {
      nodeCount = parseInt(val);
      ARTS_DEBUG("Set nodeCount=" << nodeCount);
    } else if (key == "nodes") {
      nodes = splitCSV(val);
      ARTS_DEBUG("Set nodes=" << val << " (parsed " << nodes.size()
                              << " nodes)");
    } else if (key == "masterNode") {
      masterNode = val;
      ARTS_DEBUG("Set masterNode=" << masterNode);
    } else if (key == "launcher") {
      launcher = val;
      ARTS_DEBUG("Set launcher=" << launcher);
    }

    /// GPU Configuration
    else if (key == "scheduler")
      scheduler = parseInt(val, 0);
    else if (key == "gpu")
      gpu = parseInt(val, 0);
    else if (key == "gpuRouteTableSize")
      gpuRouteTableSize = parseInt(val, 12);
    else if (key == "freeDbAfterGpuRun")
      freeDbAfterGpuRun = parseBool(val, false);
    else if (key == "deleteZerosGpuGc")
      deleteZerosGpuGc = parseBool(val, true);
    else if (key == "runGpuGcIdle")
      runGpuGcIdle = parseBool(val, true);
    else if (key == "runGpuGcPreEdt")
      runGpuGcPreEdt = parseBool(val, false);
    else if (key == "gpuLocality")
      gpuLocality = parseInt(val, 0);
    else if (key == "gpuFit")
      gpuFit = parseInt(val, 0);
    else if (key == "gpuLCSync")
      gpuLCSync = parseInt(val, 0);
    else if (key == "gpuBufferOn")
      gpuBufferOn = parseBool(val, true);
    else if (key == "gpuMaxMemory")
      gpuMaxMemory = parseInt(val, -1);
    else if (key == "gpuMaxEdts")
      gpuMaxEdts = parseInt(val, -1);
    else if (key == "gpuP2P")
      gpuP2P = parseBool(val, false);

    /// Network Configuration
    else if (key == "outgoing")
      outgoing = parseInt(val, 1);
    else if (key == "incoming")
      incoming = parseInt(val, 1);
    else if (key == "ports")
      ports = parseInt(val, 1);
    else if (key == "protocol")
      protocol = val;
    else if (key == "port")
      port = parseInt(val, 34739);
    else if (key == "netInterface")
      netInterface = val;

    /// Hardware Configuration
    else if (key == "pinStride")
      pinStride = parseInt(val, 1);
    else if (key == "printTopology")
      printTopology = parseBool(val, false);
    else if (key == "workerInitDequeSize")
      workerInitDequeSize = parseInt(val, 2048);
    else if (key == "routeTableSize")
      routeTableSize = parseInt(val, 16);
    else if (key == "coreDump")
      coreDump = parseBool(val, true);

    /// Performance Monitoring
    else if (key == "counterFolder")
      counterFolder = val;
    else if (key == "counterStartPoint")
      counterStartPoint = parseInt(val, 1);
    else if (key == "killMode")
      killMode = parseBool(val, false);

    /// Introspection
    else if (key == "introspectiveConf")
      introspectiveConf = val;
    else if (key == "introspectiveFolder")
      introspectiveFolder = val;
    else if (key == "introspectiveTraceLevel")
      introspectiveTraceLevel = parseInt(val, 0);
    else if (key == "introspectiveStartPoint")
      introspectiveStartPoint = parseInt(val, 1);
  }

  ARTS_DEBUG("Finished parsing configuration file");
  ARTS_INFO("Final configuration - threads=" << threads
                                             << ", nodeCount=" << nodeCount
                                             << ", launcher=" << launcher);

  return true;
}

std::string ArtsAbstractMachine::getMachineDescription() const {
  std::ostringstream oss;
  oss << "ARTS Abstract Machine Configuration:\n";
  oss << "  Execution Mode: " << (isLocalExecution() ? "Local" : "Distributed")
      << "\n";
  oss << "  Worker Threads: " << threads << " per node\n";
  oss << "  Total Nodes: " << nodeCount << "\n";
  oss << "  Total Worker Threads: " << getTotalWorkerThreads() << "\n";

  if (hasGpuSupport()) {
    oss << "  GPU Support: Enabled (" << gpu << " GPUs)\n";
    oss << "  GPU Scheduler: " << scheduler << "\n";
    oss << "  Total GPU Threads: " << getTotalGpuThreads() << "\n";
  } else {
    oss << "  GPU Support: Disabled\n";
  }

  oss << "  Temporal Multi-threading: " << tMT << "\n";
  oss << "  Launcher: " << launcher << "\n";
  oss << "  Master Node: " << masterNode << "\n";

  if (!nodes.empty()) {
    oss << "  Nodes: ";
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << nodes[i];
    }
    oss << "\n";
  }

  return oss.str();
}

std::string ArtsAbstractMachine::getConfigurationSummary() const {
  std::ostringstream oss;
  oss << "Configuration Summary:\n";
  oss << "  Threading: " << threads << " threads, tMT=" << tMT << "\n";
  oss << "  Networking: " << outgoing << " outgoing, " << incoming
      << " incoming threads\n";
  oss << "  Protocol: " << protocol << " on port " << port << "\n";
  oss << "  Hardware: pinStride=" << pinStride
      << ", dequeSize=" << workerInitDequeSize << "\n";
  oss << "  Monitoring: counters in " << counterFolder << "\n";

  if (hasGpuSupport()) {
    oss << "  GPU: " << gpu << " devices, locality=" << gpuLocality
        << ", fit=" << gpuFit << "\n";
  }

  return oss.str();
}

std::string ArtsAbstractMachine::trim(const std::string &s) {
  size_t i = 0, j = s.size();
  while (i < j && std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
  while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1])))
    --j;
  return s.substr(i, j - i);
}

bool ArtsAbstractMachine::startsWith(const std::string &s, const char *prefix) {
  size_t n = std::strlen(prefix);
  return s.size() >= n && std::equal(prefix, prefix + n, s.begin());
}

std::vector<std::string> ArtsAbstractMachine::splitCSV(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(trim(item));
  }
  if (out.empty() && !s.empty())
    out.push_back(trim(s));
  return out;
}

int ArtsAbstractMachine::parseInt(const std::string &value, int defaultValue) {
  if (value.empty())
    return defaultValue;
  return std::atoi(value.c_str());
}

bool ArtsAbstractMachine::parseBool(const std::string &value,
                                    bool defaultValue) {
  if (value.empty())
    return defaultValue;
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

bool ArtsAbstractMachine::validateConfiguration() {
  ARTS_DEBUG("Validating configuration...");

  bool isValid = true;

  /// Check core requirements
  if (threads <= 0) {
    ARTS_WARN("Invalid threads value: " << threads << " (must be > 0)");
    isValid = false;
  }

  if (nodeCount <= 0) {
    ARTS_WARN("Invalid nodeCount value: " << nodeCount << " (must be > 0)");
    isValid = false;
  }

  /// Check nodes consistency
  if (nodes.empty()) {
    ARTS_WARN("No nodes specified");
    isValid = false;
  } else if (static_cast<int>(nodes.size()) != nodeCount) {
    ARTS_WARN("Nodes count (" << nodes.size() << ") doesn't match nodeCount ("
                              << nodeCount << ")");
    isValid = false;
  }

  /// Check master node is in nodes list
  if (!nodes.empty() &&
      std::find(nodes.begin(), nodes.end(), masterNode) == nodes.end()) {
    ARTS_WARN("Master node '" << masterNode << "' not found in nodes list");
    isValid = false;
  }

  /// Check GPU configuration consistency
  if (hasGpuSupport()) {
    if (gpu <= 0) {
      ARTS_WARN("GPU support enabled but gpu count is " << gpu);
      isValid = false;
    }
    if (scheduler != 3) {
      ARTS_WARN("GPU support requires scheduler=3, got " << scheduler);
      isValid = false;
    }
  }

  /// Check network configuration
  // if (outgoing <= 0 || incoming <= 0) {
  //   ARTS_WARN("Invalid network thread counts: outgoing="
  //             << outgoing << ", incoming=" << incoming);
  //   isValid = false;
  // }

  // if (port <= 0 || port > 65535) {
  //   ARTS_WARN("Invalid port number: " << port);
  //   isValid = false;
  // }

  /// Set the validation flag
  isValidFlag = isValid;

  ARTS_DEBUG("Configuration validation " << (isValid ? "PASSED" : "FAILED"));
  return isValid;
}

} // namespace arts
} // namespace mlir
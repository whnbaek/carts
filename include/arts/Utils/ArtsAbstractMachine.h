///==========================================================================
/// File: ArtsAbstractMachine.h
///
/// This file defines the ARTS Abstract Machine representation that reads
/// configuration files and provides comprehensive information about
/// abstractMachine capabilities, threading, networking, GPU support, and
/// execution parameters.
///==========================================================================

#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace arts {

class ArtsAbstractMachine {
public:
  ArtsAbstractMachine(const std::string &configFile = "");

  /// Core Machine Configuration
  int getThreads() const { return threads; }
  int getTemporalMultiThreading() const { return tMT; }
  int getNodeCount() const { return nodeCount; }
  const std::vector<std::string> &getNodes() const { return nodes; }
  const std::string &getMasterNode() const { return masterNode; }
  const std::string &getLauncher() const { return launcher; }

  /// GPU Configuration
  int getScheduler() const { return scheduler; }
  int getGpuCount() const { return gpu; }
  int getGpuRouteTableSize() const { return gpuRouteTableSize; }
  int getGpuLocality() const { return gpuLocality; }
  int getGpuFit() const { return gpuFit; }
  int getGpuLCSync() const { return gpuLCSync; }
  bool isGpuBufferEnabled() const { return gpuBufferOn; }
  int getGpuMaxMemory() const { return gpuMaxMemory; }
  int getGpuMaxEdts() const { return gpuMaxEdts; }

  /// Network Configuration
  int getOutgoingThreads() const { return outgoing; }
  int getIncomingThreads() const { return incoming; }
  int getPorts() const { return ports; }
  const std::string &getProtocol() const { return protocol; }
  int getPort() const { return port; }
  const std::string &getNetInterface() const { return netInterface; }

  /// Hardware Configuration
  int getPinStride() const { return pinStride; }
  bool isPrintTopologyEnabled() const { return printTopology; }
  int getWorkerInitDequeSize() const { return workerInitDequeSize; }
  int getRouteTableSize() const { return routeTableSize; }
  bool isCoreDumpEnabled() const { return coreDump; }

  /// Performance Monitoring
  const std::string &getCounterFolder() const { return counterFolder; }
  int getCounterStartPoint() const { return counterStartPoint; }
  bool isKillModeEnabled() const { return killMode; }

  /// Machine Capability Queries
  bool hasGpuSupport() const { return scheduler == 3 && gpu > 0; }
  bool isDistributed() const { return nodeCount > 1; }
  bool isLocalExecution() const {
    return nodeCount == 1 && nodes.size() == 1 && nodes[0] == "localhost";
  }
  int getTotalWorkerThreads() const { return threads * nodeCount; }
  int getTotalGpuThreads() const { return hasGpuSupport() ? gpu : 0; }

  /// Get abstractMachine description as a formatted string
  std::string getMachineDescription() const;

  /// Get configuration summary
  std::string getConfigurationSummary() const;

  /// Configuration file status
  bool hasConfigFile() const { return configFileExists; }
  bool isValid() const { return isValidFlag; }

  /// Validation methods
  bool hasValidThreads() const { return threads > 0; }
  bool hasValidNodeCount() const { return nodeCount > 0; }
  bool validateConfiguration();

  /// Parse arts.cfg at the given path. Returns false on failure.
  bool parseFromFile(const std::string &path);

private:
  static std::string trim(const std::string &s);
  static bool startsWith(const std::string &s, const char *prefix);
  static std::vector<std::string> splitCSV(const std::string &s);
  static int parseInt(const std::string &value, int defaultValue = -1);
  static bool parseBool(const std::string &value, bool defaultValue = false);

  /// Core Configuration
  int threads = 1;
  int tMT = 0;
  int nodeCount = 1;

  std::vector<std::string> nodes = {"localhost"};
  std::string masterNode = "localhost";
  std::string launcher = "ssh";

  /// GPU Configuration
  int scheduler = 0;
  int gpu = 0;
  int gpuRouteTableSize = 12;
  bool freeDbAfterGpuRun = false;
  bool deleteZerosGpuGc = true;
  bool runGpuGcIdle = true;
  bool runGpuGcPreEdt = false;
  int gpuLocality = 0;
  int gpuFit = 0;
  int gpuLCSync = 0;
  bool gpuBufferOn = true;
  int gpuMaxMemory = -1;
  int gpuMaxEdts = -1;
  bool gpuP2P = false;

  /// Network Configuration
  int outgoing = 1;
  int incoming = 1;
  int ports = 1;
  std::string protocol = "tcp";
  int port = 34739;
  std::string netInterface = "";

  /// Hardware Configuration
  int pinStride = 1;
  bool printTopology = false;
  int workerInitDequeSize = 2048;
  int routeTableSize = 16;
  bool coreDump = true;

  /// Performance Monitoring
  std::string counterFolder = "./counters/";
  int counterStartPoint = 1;
  bool killMode = false;

  /// Introspection (optional)
  std::string introspectiveConf = "";
  std::string introspectiveFolder = "./introspective";
  int introspectiveTraceLevel = 0;
  int introspectiveStartPoint = 1;

  /// Configuration file status
  bool configFileExists = false;
  bool isValidFlag = false;
};

} // namespace arts
} // namespace mlir
